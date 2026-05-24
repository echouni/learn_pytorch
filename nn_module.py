import torch
from torch import nn


class fistModule(nn.Module):
    def __init__(self):
        super(fistModule, self).__init__()

    def forward(self, input):
        output = input + 1
        return output

model = fistModule()
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
output = model(x)
print(output)