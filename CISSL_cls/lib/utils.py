import torch
from torch import nn
import torch.nn.functional as F
import os, errno
import numpy as np
from scipy import linalg


def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    #x, y = x.numpy(), y.numpy()
    #mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        lr = 1.0
    else:
        lr = current / rampup_length

    # print (lr)
    return lr


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


class KLLoss(nn.Module):
    def __init__(self):

        super(KLLoss, self).__init__()

    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T = 3
        predict = F.log_softmax(pred/T, dim=1)
        target_data = F.softmax(label/T, dim=1)
        target_data = target_data+10**(-7)
        # target = Variable(target_data.data.cuda(),requires_grad=False)
        loss = T * T * ((target_data * (target_data.log() - predict)).sum(1))
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, max_len):
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if self.count < self.max_len:
            self.val.append(val)
            self.count += 1
        else:
            self.val.pop(0)
            self.val.append(val)

        self.sum = sum(self.val)
        self.avg = self.sum / float(self.count)
