import torch
from torch import nn

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))
print(input)



class Module(nn.Module):
    def __init__(self):
        super(Module,self).__init__()
        self.ReLU1 = nn.ReLU()

    def forward(self, x):
        output = self.ReLU1(x)
        return output

mode = Module()
output = mode(input)
print(output.shape)
print(output)