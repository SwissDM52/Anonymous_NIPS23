'''ResNet-18 Image classfication for cifar-10 with PyTorch


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        #self.relu = nn.ReLU()
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.constant_(self.fc3.bias, val=0)
        nn.init.constant_(self.fc4.bias, val=0)
        nn.init.constant_(self.fc5.bias, val=0)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        #out = self.relu(out)
        return out


def resnet10(num_classes=2):
    return Net(num_classes)

