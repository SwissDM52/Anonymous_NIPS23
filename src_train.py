import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import resnet as resn
import datetime
import numpy as np

EPOCH = 100
BATCH_SIZE = 512
LR = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=16)

trainset2 = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)  # 训练数据集


def get_loader(trainsets, low, high):
    test = trainsets
    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= low) & (targets_test < high))
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
                                              batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    return test_loader


def get_mapp(mapp):
    for i in range(10):
        mapp[i] = get_loader(trainset2, i * 10, i * 10 + 10)

    return mapp


mapp = dict()

mapp = get_mapp(mapp)


def trains1(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=5e-4)
    net.train()
    print("Start Training, Resnet-18!")
    for epoch in range(EPOCH):
        print("epoch = {}, time = {}".format(epoch, datetime.datetime.now()))
        for index, (inputs, labels) in enumerate(trainloader):
            labels.data = labels.data % 10
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch == 50:
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    return net


def trains(net, trainloaders):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=5e-4)
    net.train()
    print("Start Training, Resnet-18!")
    for epoch in range(EPOCH):
        print("epoch = {}, time = {}".format(epoch, datetime.datetime.now()))
        for index, (inputs, labels) in enumerate(trainloaders):
            labels.data = labels.data % 10
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch == 50:
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    return net


if __name__ == "__main__":
    net = resn.resnet18(10).cuda()
    model1 = trains1(net)
    torch.save(model1.state_dict(), "model1.pth")

    for i in range(10):
        model = resn.resnet18(10)
        model.load_state_dict(torch.load("model1.pth"))
        model.cuda()
        model2 = trains(model, mapp[i])
        torch.save(model2.state_dict(), "model1_{}.pth".format(i))

    for i in range(10):
        model = resn.resnet18(10)
        model.cuda()
        model2 = trains(model, mapp[i])
        torch.save(model2.state_dict(), "model2_{}.pth".format(i))
