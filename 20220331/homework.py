import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import time
import tqdm
import os
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# //                              _ooOoo_
# //                             o8888888o
# //                             88" . "88
# //                             (| -_- |)
# //                              O\ = /O
# //                           ____/`---'\____
# //                        .   ' \\| |// `.
# //                         / \\||| : |||// \
# //                        / _||||| -:- |||||- \
# //                         | | \\\ - /// | |
# //                       | \_| ''\---/'' | |
# //                        \ .-\__ `-` ___/-. /
# //                    ___`. .' /--.--\ `. . __
# //                  ."" '< `.___\_<|>_/___.' >'"".
# //                 | | : `- \`.;`\ _ /`;.`/ - ` : | |
# //                    \ \ `-. \_ __\ /__ _/ .-` / /
# //           ======`-.____`-.___\_____/___.-`____.-'======
# //                              `=---='
# //
# //           .............................................
# //                     佛祖保佑             永无BUG

# 定义超参数
epoch_num=50
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
Learning_Rate=1e-2
LOSS=[]
ACC=[]

# 读取数据
data=pd.read_csv("diabetes.csv",index_col =['BloodPressure','SkinThickness'])

# 分析数据相关性
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True)
plt.savefig('corr.png', bbox_inches='tight')
plt.show()
plt.close()

# 转换为numpy
np_data=np.array(data)

# 数据归一化
for i in range(6):
    r=np_data[:,i]
    np_data[:,i]=(r - np.min(r)) / (np.max(r) - np.min(r))

# 打印数据
print(data)
print("the csv shape is {}".format(np_data.shape))

# 分为data/label
train_data=np_data[:,:6]
train_label=np_data[:,6]

# 查看数据大小
print("train_data.shape is {}".format(train_data.shape))
print("train_label.shape is {}".format(train_label.shape))

# np->tensor
train_data_tensor=torch.from_numpy(train_data).to(device).float()
train_label_tensor=torch.from_numpy(train_label).to(device).long()
print("the tensor of data is size of {}".format(train_data_tensor.shape))
print("the tensor of label is size of {}".format(train_label_tensor.shape))

# 划分训练集/验证集
torch_dataset = Data.TensorDataset(train_data_tensor,train_label_tensor)
train_size = int(len(torch_dataset) * 0.8)
val_size = len(torch_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [train_size, val_size])
train_loader=Data.DataLoader(train_dataset,batch_size=64,shuffle=True)
val_loader=Data.DataLoader(val_dataset,batch_size=64,shuffle=True)

# 查看数据集
print("train_loader length is {}".format(len(train_loader)))
print("val_loader length is {}".format(len(val_loader)))

# 定义网络
class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linear_0 = torch.nn.Linear(6, 16)
        self.linear_1 = torch.nn.Linear(16, 8)
        self.linear_2 = torch.nn.Linear(8, 4)
        self.linear_3 = torch.nn.Linear(4, 2)
    def forward(self,x):
        linear_0=self.linear_0(x)
        linear_1 = self.linear_1(linear_0)
        linear_2 = self.linear_2(linear_1)
        linear_3 = self.linear_3(linear_2)
        return linear_3

# 定义损失函数\优化器
Net=net().to(device)
criterion=torch.nn.CrossEntropyLoss()
opt=torch.optim.Adam(Net.parameters(),lr=Learning_Rate)

# 训练
for epoch in range(epoch_num):
    start=time.time()
    running_loss = 0.0
    for x,y in tqdm.tqdm(iter(train_loader)):
        pre=Net(x)
        loss=criterion(pre,y)
        running_loss=running_loss+loss/64
        opt.zero_grad()
        loss.backward()
        opt.step()
    LOSS.append(running_loss)
    end = time.time()
    print("train {} epoch ,total loss is {},time use {}s".format(epoch+1,running_loss,end-start))
    with torch.no_grad():
        total=0
        correct=0
        start = time.time()
        running_loss = 0.0
        for x, y in tqdm.tqdm(iter(val_loader)):
            pre = Net(x)
            _, predicted = torch.max(pre, -1)
            total += y.size(0)
            correct += (predicted == y.view(-1)).sum().item()
        acc=correct/total
        ACC.append(acc)
        end = time.time()
    print("val {} epoch ,total acc is {}%,time use {}s".format(epoch + 1, acc, end - start))

# 画图
plt.figure(1)
plt.plot(range(epoch_num), ACC, '.-b', label='Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend()
plt.savefig('Acc.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(2)
plt.plot(range(epoch_num), LOSS, '.-r', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png', bbox_inches='tight')
plt.show()
plt.close()
