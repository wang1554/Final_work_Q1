import glob
import os.path as osp
import random
import numpy as np
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import models, transforms
# 设置CPU生成随机数的种子，方便下次复现实验结果
torch.manual_seed(2022)
# 每次生成的随机数都相同
np.random.seed(2022)
random.seed(2022)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

# transforms.Resize(size):将图片的短边缩放成size的比例，然后长边也跟着缩放，使得缩放后的图片相对于原图的长宽比不变
# transforms.CenterCrop(size):从图像的中心位置裁剪指定大小的图像
# ToTensor():将图像由PIL转换为Tensor
# transform.Normalize():把0-1变换到(-1,1)
# image = (image - mean) / std
# 其中mean和std分别通过(0.5, 0.5, 0.5)和(0.2, 0.2, 0.2)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.2, 0.2, 0.2))
])
train_dataset = torchvision.datasets.CIFAR10("../self_sup/dataset", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.CIFAR10("../self_sup/dataset", train=False, transform=trans, download=True)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
batch_size = 128
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
data_loaders = {
    "train": train_dataloader,
    "test": test_dataloader
}

net = torchvision.models.resnet18(pretrained=False)
net.fc = nn.Linear(512, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

#使用Tensorboard可视化工具
TIMESTAMP = f"{datetime.now():%Y-%m-%dT%H-%M-%S/}"
writer = SummaryWriter(f"/tf-logs/{TIMESTAMP}/")


def train_model(net,dataloaders, criterion, optimizer, num_epochs):
    net.to(device)

    # 网络加速
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ["train", "test"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            # 损失
            epoch_loss = 0.0
            # 正确答案的数量
            epoch_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # 设置梯度计算开启或关闭
                # 只在训练时开启梯度计算
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    # 计算损失
                    loss = criterion(outputs, labels)
                    # 预测标签
                    # 返回每一行的最大值，也就是所属的类别
                    _, preds = torch.max(outputs, 1)

                    # 反向传播
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # loss的总和
                    # loss计算的是平均值，所以要乘上batchsize，计算损失的总和
                    epoch_loss += loss.item() * inputs.size(0)
                    # 预测正确的答案的数量
                    epoch_corrects += torch.sum(preds == labels.data)

                    # 每个epoch的loss和正确率
                    epoch_loss = epoch_loss / len(data_loaders[phase].dataset)
                    epoch_acc = epoch_corrects.double() / len(data_loaders[phase].dataset)

                    epoch_loss = torch.tensor(epoch_loss)
                    epoch_acc = torch.tensor(epoch_acc)
                    print(f"epoch: {epoch + 1}/{num_epochs}; {phase} loss:{np.round(epoch_loss.item(), 5)}; acc:{np.round(epoch_acc.item() * 100, 2)}%")
            writer.add_scalar(f"{phase} loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase} accuracy", epoch_acc, epoch)


            # 保存模型
            if (epoch % 20 == 0 and phase == 'train'):
                save_path = f"./save_path/resnet_{epoch}_{epoch_acc}.pth"
                torch.save(net.state_dict(), save_path)
num_epochs = 200
train_model(net, data_loaders, criterion, optimizer, num_epochs)
