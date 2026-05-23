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

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)


# Compose - resize
trans_resize_2 = transforms.Resize(512)

trans_compose = transforms.Compose([trans_totensor,trans_resize_2])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Crop", img_crop, i)



writer.close()
