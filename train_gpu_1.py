import torch
# 准备数据集
import torchvision,os
from sympy import sequence
from torch import nn
# 让 MPS 不支持的算子自动回退到 CPU；在 Windows/Linux 上设置此变量也无害
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from torch.utils.tensorboard import SummaryWriter

from model import *

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


train_data = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练数据集长度：", train_data_size)
print(f"测试集长度：", test_data_size)

# 加载数据集
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x
model = Model()
model = model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
train_num = 0
# 记录测试的次数
test_num = 0
# 训练的轮数
epoch = 10

# 添加Tensorboard
writer = SummaryWriter("logs/")

for i in range(epoch):
    print(f"第{i + 1}轮训练开始了 ")
    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_num += 1
        if train_num % 100 == 0:
            print(f"训练次数{train_num},Loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), train_num)

    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, test_num)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, test_num)
    test_num = test_num + 1

    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()



