# trainstage1.py
import torch, argparse
import net
import resnet as resn
import classifier
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

BATCH_SIZE = 300
# 准备数据集并预处理

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=16)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取


# train stage one
def train(args, train_data):
    DEVICE = torch.device("cuda:0")
    generator = net.SimCLRStage1()
    # generator.load_state_dict(torch.load("gan5_b/generator_100.pth"))
    # model.load_state_dict(torch.load("pth/model_stage1_epoch90.pth"))
    generator = generator.to(DEVICE)

    # lossLR = net.Loss().to(DEVICE)
    model1 = resn.resnet18()
    model1.load_state_dict(torch.load("model_source/model1.pth"))
    model1 = model1.to(DEVICE)

    model2_0 = resn.resnet18()
    model2_0.load_state_dict(torch.load("model_source/model1_3.pth"))
    model2_0 = model2_0.to(DEVICE)

    model2_1 = resn.resnet18()
    model2_1.load_state_dict(torch.load("model_source/model1_6.pth"))
    model2_1 = model2_1.to(DEVICE)

    model2_2 = resn.resnet18()
    model2_2.load_state_dict(torch.load("model_source/model1_8.pth"))
    model2_2 = model2_2.to(DEVICE)

    model2_3 = resn.resnet18()
    model2_3.load_state_dict(torch.load("model_source/model1_9.pth"))
    model2_3 = model2_3.to(DEVICE)

    model2_5 = resn.resnet18()
    model2_5.load_state_dict(torch.load("model_source/model1_5.pth"))
    model2_5 = model2_5.to(DEVICE)

    model3_0 = resn.resnet18()
    model3_0.load_state_dict(torch.load("model_source/model2_3.pth"))
    model3_0 = model3_0.to(DEVICE)

    model3_1 = resn.resnet18()
    model3_1.load_state_dict(torch.load("model_source/model2_6.pth"))
    model3_1 = model3_1.to(DEVICE)

    model3_2 = resn.resnet18()
    model3_2.load_state_dict(torch.load("model_source/model2_8.pth"))
    model3_2 = model3_2.to(DEVICE)

    model3_3 = resn.resnet18()
    model3_3.load_state_dict(torch.load("model_source/model2_9.pth"))
    model3_3 = model3_3.to(DEVICE)

    model3_5 = resn.resnet18()
    model3_5.load_state_dict(torch.load("model_source/model2_5.pth"))
    model3_5 = model3_5.to(DEVICE)

    dismodel = classifier.resnet10()
    # dismodel.load_state_dict(torch.load("gan5_b/dismodel_15.pth"))
    dismodel = dismodel.to(DEVICE)

    lossLR = net.Loss().to(DEVICE)

    optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer_D = torch.optim.Adam(dismodel.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    generator.train()
    model1.eval()
    model2_0.eval()
    model2_1.eval()
    model2_2.eval()
    model2_3.eval()
    model2_5.eval()
    model3_0.eval()
    model3_1.eval()
    model3_2.eval()
    model3_3.eval()
    model3_5.eval()
    dismodel.train()

    for epoch in range(1, args.max_epoch + 1):
        total_loss = 0
        for index, (inputs, labels) in enumerate(train_data):
            labels.data = labels.data % 10
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, pre = generator(inputs)
            outputs1 = model1(inputs)
            outputs2_0 = model2_0(inputs)
            outputs2_1 = model2_1(inputs)
            outputs2_2 = model2_2(inputs)
            outputs2_3 = model2_3(inputs)
            outputs2_5 = model2_5(inputs)
            outputs3_0 = model3_0(inputs)
            outputs3_1 = model3_1(inputs)
            outputs3_2 = model3_2(inputs)
            outputs3_3 = model3_3(inputs)
            outputs3_5 = model3_5(inputs)
            outs_pre = torch.cat([pre, pre, pre, pre, pre], dim=0)
            outs_1 = torch.cat([outputs1, outputs1, outputs1, outputs1, outputs1], dim=0)
            outs_2 = torch.cat([outputs2_0, outputs2_1, outputs2_2, outputs2_3, outputs2_5], dim=0)
            outs_3 = torch.cat([outputs3_0, outputs3_1, outputs3_2, outputs3_3, outputs3_5], dim=0)
            pre_1 = outs_pre + outs_1
            pre_2 = outs_pre + outs_2
            pre_3 = outs_pre + outs_3
            loss_G = lossLR(pre_1, pre_2, pre_3, len(inputs) * 5)
            # loss = lossLR(pre, pre, pre, args.batch_size)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            # print("epoch", epoch, "batch", index, "loss:", loss_G.detach().item())
            total_loss += loss_G.detach().item()
        # total_loss += loss.detach().item()
        print("epoch {} generator loss: {}".format(epoch, total_loss / (2 * len(trainset)) * args.batch_size))
        if epoch % 5 == 0:
            # torch.save(dismodel.state_dict(), "gan5_a/dismodel_{}.pth".format(epoch))
            torch.save(generator.state_dict(), "gan5_b/generator_{}.pth".format(epoch))

    for epoch in range(1, args.max_epoch + 1):
        # model.train()
        total_loss = 0
        # for batch, (imgL, imgR, labels) in enumerate(train_data):
        for index, (inputs, labels) in enumerate(train_data):
            labels.data = labels.data % 10
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, pre = generator(inputs)
            outputs1 = model1(inputs)
            outputs2_0 = model2_0(inputs)
            outputs2_1 = model2_1(inputs)
            outputs2_2 = model2_2(inputs)
            outputs2_3 = model2_3(inputs)
            outputs2_5 = model2_5(inputs)
            outputs3_0 = model3_0(inputs)
            outputs3_1 = model3_1(inputs)
            outputs3_2 = model3_2(inputs)
            outputs3_3 = model3_3(inputs)
            outputs3_5 = model3_5(inputs)
            pre_1 = pre + outputs1
            pre_20 = pre + outputs2_0
            pre_21 = pre + outputs2_1
            pre_22 = pre + outputs2_2
            pre_23 = pre + outputs2_3
            pre_25 = pre + outputs2_5
            pre_30 = pre + outputs3_0
            pre_31 = pre + outputs3_1
            pre_32 = pre + outputs3_2
            pre_33 = pre + outputs3_3
            pre_35 = pre + outputs3_5
            outs00 = torch.cat([pre_1, pre_30], dim=1)
            outs01 = torch.cat([pre_1, pre_31], dim=1)
            outs02 = torch.cat([pre_1, pre_32], dim=1)
            outs03 = torch.cat([pre_1, pre_33], dim=1)
            outs05 = torch.cat([pre_1, pre_35], dim=1)
            outs0 = torch.cat([outs00, outs01, outs02, outs03, outs05], dim=0)
            outs10 = torch.cat([pre_1, pre_20], dim=1)
            outs11 = torch.cat([pre_1, pre_21], dim=1)
            outs12 = torch.cat([pre_1, pre_22], dim=1)
            outs13 = torch.cat([pre_1, pre_23], dim=1)
            outs15 = torch.cat([pre_1, pre_25], dim=1)
            outs1 = torch.cat([outs10, outs11, outs12, outs13, outs15], dim=0)
            outs = torch.cat([outs0, outs1], dim=0)
            # outs = [outs0.to(DEVICE),outs1.to(DEVICE)]
            ous = dismodel(outs)
            labs0 = torch.zeros(len(inputs) * 5).to(DEVICE)
            labs1 = torch.ones(len(inputs) * 5).to(DEVICE)
            labs = torch.cat([labs0, labs1], dim=0)
            loss = criterion(ous, labs.long())
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()
            total_loss += loss.detach().item()
            # print("epoch", epoch, "batch", index, "loss:", loss.detach().item())
        if epoch % 5 == 0:
            torch.save(dismodel.state_dict(), "gan5_b/dismodel_{}.pth".format(epoch))
            # torch.save(generator.state_dict(), "gan5_a/generator_{}.pth".format(epoch))
        print("epoch {} dismodel loss: {}".format(epoch, total_loss / (2 * len(trainset)) * args.batch_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=300, type=int, help='')
    parser.add_argument('--max_epoch', default=100, type=int, help='')

    args = parser.parse_args()
    train(args, trainloader)
