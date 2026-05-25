import torch
import torchvision
from sympy.tensor.array.arrayop import Flatten
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Linear
from torch.nn.modules import padding
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./datasets',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)

class module(nn.Module):
    def __init__(self):
        super(module,self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10),

        )

    def forward(self,x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
model = module()
optim = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        output = model(imgs)
        res_loss = loss(output,targets)
        optim.zero_grad()
        res_loss.backward()
        optim.step()
        running_loss += res_loss + running_loss
    print(running_loss)