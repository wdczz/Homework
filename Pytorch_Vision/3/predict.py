import torch
import cv2
from torchvision import datasets, transforms, utils
import PIL.Image as Image
def pre(console,net_choose,device):
    while True:
        path = console.input("Please give me the path of the picture -> ")
        img = cv2.imread(path, 1)
        try :
            if img.any() != None :
                break
        except:
            console.log("Oh the path maybe is not effective ,so give me another one !")
    transform = transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor()
                                    ])
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    input = transform(img_pil).unsqueeze(0).to(device)
    if net_choose=="1":
        net=torch.load("Convnet{}.pt".format(device)).to(device)
    else:
        net = torch.load("lenet{}.pt".format(device)).to(device)
    output=net(input)
    _, pre_lab = torch.max(output, 1)
    console.log("The result is {}".format(int(pre_lab.detach().cpu())))
    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

