import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel, self).__init__()

    # define layers
    # 28*28*1
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
    # self.conv2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.activation = nn.ReLU()
    # output from conv2 = 8*8*12
    # after pooling, it will be 4*4*12
    # therefore nb of parameters will be 12*4*4
    self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(in_features=120, out_features=84)
    self.out = nn.Linear(in_features=84, out_features=10)

  # define forward function
  def forward(self, x):
    # conv 1
    x = self.pool(self.activation(self.conv1(x)))
    x = self.pool(self.activation(self.conv2(x)))
    # fc1
    x = x.reshape(-1, 16 * 5 * 5)
    x = self.activation(self.fc1(x))
    x = self.dropout(x)
    # fc2
    x = self.activation(self.fc2(x))
    x = self.dropout(x)
    # output
    x = self.out(x)
    return x

class CNNModel2(nn.Module):
  def __init__(self):
    super(CNNModel2, self).__init__()

    # define layers
    # 28*28*1
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
    # self.conv2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.activation = nn.ReLU()
    # output from conv2 = 8*8*12
    # after pooling, it will be 4*4*12
    # therefore nb of parameters will be 12*4*4
    self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(in_features=120, out_features=84)
    self.out = nn.Linear(in_features=84, out_features=10)

  # define forward function
  def forward(self, x):
    # conv 1
    x = self.pool(self.activation(self.conv1(x)))
    x = self.pool(self.activation(self.conv2(x)))
    # fc1
    x = x.reshape(-1, 16 * 5 * 5)
    x = self.activation(self.fc1(x))
    x = self.dropout(x)
    # fc2
    x = self.activation(self.fc2(x))
    x = self.dropout(x)
    # output
    x = self.out(x)
    return x

