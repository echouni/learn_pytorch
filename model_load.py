import torch
import torchvision
#方式1保存，加载模型
# model = torch.load("vgg16_method1.pth",weights_only=False)
# print(model)

#方式2，加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)