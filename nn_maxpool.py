import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./datasets',train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)

class Module(nn.Module):
    def __init__(self):
        super(Module,self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        output = self.maxpool1(x)
        return output

mode = Module()


writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = mode(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()