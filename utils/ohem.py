import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.lovasz_softmax import LovaszSoftmaxV1


class OHEM_Loss(nn.CrossEntropyLoss):
    # 继承自nn.CrossEntropyLoss
    def __init__(self, ratio, weight):
        super(OHEM_Loss, self).__init__()
        self.ratio = ratio
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, x, y, ratio=None, weight=None):
        if ratio is not None:
            self.ratio = ratio
        if weight is not None:
            self.weight = weight
        # print(self.ratio)
        x = x[0]
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        inst_losses = self.criterion(x, y)
        value, idxs = inst_losses.topk(num_hns)

        loss = torch.mean(value)
        return loss


class OAELoss(nn.Module):
    def __init__(self, num_class=7, island_loss_weight=0.1, LS_loss_weight=1, lamda1=0.1, weight=None, seg_th=0.5):
        super(OAELoss, self).__init__()
        self.L_s = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.LovaszSoftmax_loss = LovaszSoftmaxV1(reduction='mean', ignore_index=255)
        self.num_class = num_class
        self.lamda1 = lamda1
        self.island_loss_weight = island_loss_weight
        self.LS_loss_weight = LS_loss_weight
        self.seg_th = seg_th

    def forward(self, x, y, mask):
        out, feature, x_seg = x
        y_seg = torch.where(mask > self.seg_th, 1, 0).to(x_seg.device)
        loss = torch.mean(self.L_s(out, y)) + self.island_loss_weight * self.island_loss(feature, y) + \
               self.LS_loss_weight * self.LovaszSoftmax_loss(x_seg, y_seg)[0]
        return loss

    def island_loss(self, x, y):
        n_cls = self.num_class
        gamma1 = self.lamda1
        x_mean = torch.zeros_like(x).cuda()
        class_means = torch.zeros(n_cls, x.size()[1]).cuda() + 0.000000001

        for k in range(n_cls):
            idx = Variable(torch.cuda.LongTensor([i for (i, x) in enumerate(y) if int(x.cpu().data.numpy()) == k]))

            if idx.size()[0] > 0:
                mean = torch.mean(x.index_select(0, idx), 0)
                x_mean[idx] = mean
                class_means[k] = mean

        loss = 0.0
        for i in range(n_cls):
            for j in range(n_cls):
                if i != j:
                    prod = (class_means[i].dot(class_means[j]) / (
                            torch.norm(class_means[i], 2) * torch.norm(class_means[j], 2))) + 1
                    loss += prod
        return F.mse_loss(x, x_mean) + gamma1 * loss
