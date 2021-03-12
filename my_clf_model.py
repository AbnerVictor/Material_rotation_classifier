import torch
import torch.nn as nn
import torch.nn.functional as  func
import torch.optim as optim


def weight_reset(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    m.reset_parameters()


class My_Classify_Net(nn.Module):
  def __init__(self):
    super(My_Classify_Net, self).__init__()
    # 定义网络层
    self.conv1 = nn.Conv2d(1, 2, 5)  # 卷积核的大小为5x5，输入1层，输出2层
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(2, 16, 5)
    self.fc1 = nn.Linear(16*22*22, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 4)

  def forward(self, x):
    # 定义网络结构
    x = self.pool(func.relu(self.conv1(x))) # 卷积->激活函数->池化
    x = self.pool(func.relu(self.conv2(x)))
    x = x.view(-1, self.num_flat_features(x)) # 将高维Tensor拼成一行
    x = func.relu(self.fc1(x)) # 全连接->激活函数
    x = func.relu(self.fc2(x))
    x = self.fc3(x) # 输出层
    return x

  def num_flat_features(self, x):
    # 特征维度
    size = x.size()[1:] # x的第一个维度是该batch中data的数量
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


class My_Classify_Net_2(nn.Module):
  def __init__(self):
    super(My_Classify_Net_2, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
    self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
    self.maxpool = nn.MaxPool2d(2, 2)
    self.avgpool = nn.AvgPool2d(2, 2)
    self.globalavgpool = nn.AvgPool2d(8, 8)
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)
    self.dropout50 = nn.Dropout(0.5)
    self.dropout10 = nn.Dropout(0.1)
    self.fc = nn.Linear(256, 10)

  def forward(self, x):
    x = self.bn1(func.relu(self.conv1(x)))
    x = self.bn1(func.relu(self.conv2(x)))
    x = self.maxpool(x)
    x = self.dropout10(x)
    x = self.bn2(func.relu(self.conv3(x)))
    x = self.bn2(func.relu(self.conv4(x)))
    x = self.avgpool(x)
    x = self.dropout10(x)
    x = self.bn3(func.relu(self.conv5(x)))
    x = self.bn3(func.relu(self.conv6(x)))
    x = self.globalavgpool(x)
    x = self.dropout50(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x