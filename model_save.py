import torch
import torchvision

from model_pretrain import vgg16_false

vgg16_false = torchvision.models.vgg16(weights=None)
# 保存方式1:模型结构+模型参数
torch.save(vgg16_false,"vgg16_method1.pth")


# 保存方式2：模型参数
torch.save(vgg16_false.state_dict(),"vgg16_method2.pth")


#陷阱
