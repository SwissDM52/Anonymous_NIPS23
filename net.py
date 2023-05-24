# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=10, grayscale=False):
        super(SimCLRStage1, self).__init__()
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, out_1, out_2, out_3, batch_size, temperature=30):
        def cal_loss(out_1, out_2, batch_size, temperature=2):
            out = torch.cat([out_1, out_2], dim=0)

            sim_matrix = torch.sigmoid(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            sim = torch.sigmoid(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            sim = torch.cat([sim, sim], dim=0)
            return sim, sim_matrix

        sim_pos, _ = cal_loss(out_1, out_2, batch_size, temperature)
        sim_neg, _ = cal_loss(out_1, out_3, batch_size, temperature)
        _, matrix1 = cal_loss(out_1, out_1, batch_size, temperature)
        _, matrix2 = cal_loss(out_2, out_2, batch_size, temperature)
        _, matrix3 = cal_loss(out_3, out_3, batch_size, temperature)

        loss = torch.mean(-torch.log(sim_pos / (sim_pos + sim_neg)) - torch.log(matrix1).mean(dim=-1) - torch.log(
            matrix2).mean(dim=-1) - torch.log(matrix3).mean(dim=-1))
        return loss


if __name__ == "__main__":
    for name, module in resnet50().named_children():
        print(name, module)
