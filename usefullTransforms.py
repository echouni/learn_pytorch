from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/5650366_e22b7e1065.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ants/5650366", img_tensor, 0)


# Normalize 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Norm", img_norm, 0)


writer.close()
