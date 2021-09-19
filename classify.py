import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torch.autograd import Variable

root = "CNN_synth_testset"


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数

        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 用来装从txt读取的图片路径及标签
        for line in fh:  # 按行循环txt文本中的每行内容
            words = line.split()  # 按空格分隔
            imgs.append((words[0], int(words[1])))  # words[0]是图片路径，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # xxxx.item()
        fn, label = self.imgs[index]  # fn是图片路径 ，label是标签 对应于刚才的word[0]和word[1]的信息
        img = Image.open(root + fn).resize((64, 64), Image.ANTIALIAS)  # 按照path读入图片，并将尺寸压缩到64*64像素
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


train_data = MyDataset(root=root, datatxt='/train.txt', transform=transforms.ToTensor())
test_data = MyDataset(root=root, datatxt='/test.txt', transform=transforms.ToTensor())

# 读取数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv1: Conv2d ->ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # conv2: Conv2d -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # fully connected layer
        self.fcs = nn.Sequential(
            nn.Linear(32 * (64 // 4) * (64 // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Net()

# training....
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam梯度优化器
device = torch.device("cpu")
epochs = 10
for i in range(epochs):
    losses = 0  # 总损失 ，可以求平均损失
    correct = 0  # 总正确数，可以求训练的准确度
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        losses += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ,accuracy: {:.6f} '.format(
                i, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), losses / (batch_idx + 1), correct / (batch_idx + 1)))

# model = Net()
# model.load_state_dict(torch.load('smarter_model_weights1.pth'))


# testing.....
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

torch.save(model.state_dict(),
           'smarter_model_weights2.pth')  # model = Net(); model.load_state_dict(torch.load('smarter_model_weights.pth'))
torch.save(model, 'smarter_model2.pth')  # model = torch.load("smarter_model.pth2")
