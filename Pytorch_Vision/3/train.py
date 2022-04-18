import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torchvision.datasets import ImageFolder

import torchvision
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from net import ConvNet
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.1307, ],
#                                                      std=[0.3081, ])])

def train(console):
    transform = transforms.Compose([
        transforms.ToTensor()
                                    ])
    train_data = datasets.MNIST(
        root="./data/MNIST",
        train=True,
        transform=transform,
        download=False
    )
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True,
    )
    test_data = datasets.MNIST(
        root="./data/MNIST",
        transform=transform,
        train=False,
        download=False
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=128,
        shuffle=True,
    )

    MyConvnet = ConvNet().to("cuda")
    optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
    loss_func = nn.CrossEntropyLoss()
    epoch_num = 4
    print_step = 100
    acc_list = []
    loss_list = []
    for epoch in range(epoch_num):
        print("\nepoch:", epoch)
        train_loss = 0
        acc_total = 0
        print("Training")
        for b_x, b_y in tqdm(iter(train_loader)):
            b_x = b_x.to("cuda")
            b_y = b_y.to("cuda")
            output = MyConvnet(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss

        print("\nVerify the training set accuracy!")
        with torch.no_grad():
            for b_x, b_y in tqdm(iter(test_loader)):
                b_x = b_x.to("cuda")
                output = MyConvnet(b_x)
                _, pre_lab = torch.max(output, 1)
                acc = accuracy_score(b_y, pre_lab.to("cpu"))
                acc_total += acc
        acc_list.append(acc_total / len(test_loader))
        loss_list.append(train_loss / len(test_loader))
        print("\nThe training set accuracy is {}".format(acc_total / len(test_loader)))

    print("\nVerify the testing set accuracy!")
    with torch.no_grad():
        acc_total = 0
        for b_x, b_y in tqdm(iter(test_loader)):
            b_x = b_x.to("cuda")
            output = MyConvnet(b_x)
            _, pre_lab = torch.max(output, 1)
            acc = accuracy_score(b_y, pre_lab.to("cpu"))
            acc_total += acc
        acc_total /= len(test_loader)
        print("The testing set accuracy is {}".format(acc_total))
        print("predict is {}".format(pre_lab))
        print("label is {}".format(b_y.to("cpu")))
    torch.save(MyConvnet, 'Convnet.pt')
    console.log("Successfully save the model!")
    img = utils.make_grid(b_x.to("cpu"))
    img = img.numpy().transpose(1, 2, 0)
    cv2.imshow("test", img)

    plt.figure(1)
    plt.plot(range(epoch_num), acc_list, '.-b', label='Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('Acc.png', bbox_inches='tight')
    plt.show()

    plt.figure(2)
    plt.plot(range(epoch_num), loss_list, '.-r', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss.png', bbox_inches='tight')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()