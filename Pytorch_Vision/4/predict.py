import torch
import cv2
from torchvision import datasets, transforms, utils
import PIL.Image as Image


def pre(console,img,index,net_choose,device):
    transform = transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor()
                                    ])
    img_pil = Image.fromarray(img)
    input = transform(img_pil).unsqueeze(0).to(device)
    if net_choose=='1':
        net = torch.load("Convnet{}.pt".format(device))
    else:
        net=torch.load("lenet{}.pt".format(device))
    output=net(input)
    _, pre_lab = torch.max(output, 1)
    console.log("The result is {}".format(int(pre_lab.detach().cpu())))
    return int(pre_lab.detach().cpu())
