from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)

writer.add_image("test",img_array,1,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()