# trainstage1.py
import torch
import net
import resnet as resn
import classifier as resn10
import torchvision.transforms as transforms
import torchvision

BATCH_SIZE = 100

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)  
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=16)  


def tests(model1, model2, model3, gen, dis, trainloaders, DEVICE, files):
    correct = 0
    total = 0
    for index, (inputs, labels) in enumerate(trainloaders):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        _, pre = gen(inputs)
        # pre = gen(inputs)
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        outputs3 = model3(inputs)
        pre_1 = pre + outputs1
        pre_2 = pre + outputs2
        pre_3 = pre + outputs3
        outs0 = torch.cat([pre_1, pre_3], dim=1)
        outs1 = torch.cat([pre_1, pre_2], dim=1)
        outs = torch.cat([outs0, outs1], dim=0)
        # outs = [outs0.to(DEVICE),outs1.to(DEVICE)]
        ous = dis(outs)
        
        _, predicted = torch.max(ous.data, 1)
        total += labels.size(0)
        labs0 = torch.zeros(BATCH_SIZE).to(DEVICE)
        labs1 = torch.ones(BATCH_SIZE).to(DEVICE)
        labs = torch.cat([labs0, labs1], dim=0)

        correct += (predicted == labs % 10).sum().item()
        # print(predicted)
    total = total * 2
    acc = 100 * (correct / total)

    print("total:{},test accuracy：{}%\n".format(total, acc))
    files.writelines('total pictures:{},test accuracy：{}%\n'.format(total, acc))
    return acc


def tests0(model1, model3, gen, dis, trainloaders, DEVICE, files):
    correct = 0
    total = 0

    for index, (inputs, labels) in enumerate(trainloaders):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        _, pre = gen(inputs)
        outputs1 = model1(inputs)
        outputs3 = model3(inputs)
        pre_1 = pre + outputs1
        pre_3 = pre + outputs3
        outs0 = torch.cat([pre_1, pre_3], dim=1)
        ous = dis(outs0)
        
        _, predicted = torch.max(ous.data, 1)
        total += labels.size(0)
        labs0 = torch.zeros(BATCH_SIZE).to(DEVICE)
        correct += (predicted == labs0 % 10).sum().item()
        # print(predicted)
    total = total
    acc = 100 * (correct / total)

    print("total:{},test0 accuracy：{}%\n".format(total, acc))
    files.writelines('total pictures:{},test0 accuracy：{}%\n'.format(total, acc))
    return acc


def tests1(model1, model2, gen, dis, trainloaders, DEVICE, files):
    correct = 0
    total = 0

    for index, (inputs, labels) in enumerate(trainloaders):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        _, pre = gen(inputs)
        
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        pre_1 = pre + outputs1
        pre_2 = pre + outputs2
        outs1 = torch.cat([pre_1, pre_2], dim=1)
        ous = dis(outs1)
        
        _, predicted = torch.max(ous.data, 1)
        total += labels.size(0)
        labs1 = torch.ones(BATCH_SIZE).to(DEVICE)

        correct += (predicted == labs1 % 10).sum().item()
        # print(predicted)
    total = total
    acc = 100 * (correct / total)

    print("total:{},test1 accuracy：{}%\n".format(total, acc))
    files.writelines('total pictures:{},test1 accuracy：{}%\n'.format(total, acc))
    return acc


if __name__ == '__main__':
    DEVICE = torch.device("cuda:0")
    mapp2 = []
    mapp3 = []

    model1 = resn.resnet18()
    model1.load_state_dict(torch.load("model_source/model1.pth"))
    model1 = model1.to(DEVICE)

    model2_0 = resn.resnet18()
    model2_0.load_state_dict(torch.load("model_source/model1_1.pth"))
    model2_0 = model2_0.to(DEVICE)

    model2_1 = resn.resnet18()
    model2_1.load_state_dict(torch.load("model_source/model1_2.pth"))
    model2_1 = model2_1.to(DEVICE)

    model2_2 = resn.resnet18()
    model2_2.load_state_dict(torch.load("model_source/model1_4.pth"))
    model2_2 = model2_2.to(DEVICE)

    model2_3 = resn.resnet18()
    model2_3.load_state_dict(torch.load("model_source/model1_7.pth"))
    model2_3 = model2_3.to(DEVICE)

    model2_5 = resn.resnet18()
    model2_5.load_state_dict(torch.load("model_source/model1_0.pth"))
    model2_5 = model2_5.to(DEVICE)

    mapp2.insert(len(mapp2), model2_0)
    mapp2.insert(len(mapp2), model2_1)
    mapp2.insert(len(mapp2), model2_2)
    mapp2.insert(len(mapp2), model2_3)
    mapp2.insert(len(mapp2), model2_5)

    model3_0 = resn.resnet18()
    model3_0.load_state_dict(torch.load("model_source/model2_1.pth"))
    model3_0 = model3_0.to(DEVICE)

    model3_1 = resn.resnet18()
    model3_1.load_state_dict(torch.load("model_source/model2_2.pth"))
    model3_1 = model3_1.to(DEVICE)

    model3_2 = resn.resnet18()
    model3_2.load_state_dict(torch.load("model_source/model2_4.pth"))
    model3_2 = model3_2.to(DEVICE)

    model3_3 = resn.resnet18()
    model3_3.load_state_dict(torch.load("model_source/model2_7.pth"))
    model3_3 = model3_3.to(DEVICE)

    model3_5 = resn.resnet18()
    model3_5.load_state_dict(torch.load("model_source/model2_0.pth"))
    model3_5 = model3_5.to(DEVICE)

    mapp3.insert(len(mapp3), model3_0)
    mapp3.insert(len(mapp3), model3_1)
    mapp3.insert(len(mapp3), model3_2)
    mapp3.insert(len(mapp3), model3_3)
    mapp3.insert(len(mapp3), model3_5)
    genmodel = net.SimCLRStage1()
    genmodel.load_state_dict(torch.load("gan5_b/generator_{}.pth".format(100)))
    genmodel = genmodel.to(DEVICE)

    dismodel = resn10.resnet10()
    dismodel.load_state_dict(torch.load("gan5_b/dismodel_{}.pth".format(100)))
    dismodel = dismodel.to(DEVICE)
    genmodel.eval()
    dismodel.eval()
    model1.eval()
    for i in range(5):
        model2 = mapp2[i]
        model3 = mapp3[i]
        model2.eval()
        model3.eval()
        files = open('gan5_b.txt', mode='a')
        files.writelines('gan5_b,i={}=====================================cifar10\n'.format(i))
        tests(model1, model2, model3, genmodel, dismodel, trainloader, DEVICE, files)
        tests0(model1, model3, genmodel, dismodel, trainloader, DEVICE, files)
        tests1(model1, model2, genmodel, dismodel, trainloader, DEVICE, files)
        files.close()
