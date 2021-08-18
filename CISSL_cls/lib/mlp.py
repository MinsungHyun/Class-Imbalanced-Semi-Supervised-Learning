import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils import mixup_data

## code for CNN13 from https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py
from torch.nn.utils import weight_norm


class MLP(nn.Module):

    def __init__(self, num_classes=2, transform_fn=None):
        super(MLP, self).__init__()

        self.activation = nn.LeakyReLU(0.1)
        # self.fc_in = weight_norm(nn.Linear(2, 100))
        # self.fc_hidden = weight_norm(nn.Linear(100, 100))
        # self.fc_out = weight_norm(nn.Linear(100, num_classes))
        n_hidden = 30
        self.fc_in = nn.Linear(2, n_hidden)
        self.fc_hidden = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, num_classes)

        self.transform_fn = transform_fn

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)

        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_hidden(x)
        x = self.activation(x)
        x = self.fc_hidden(x)
        x = self.activation(x)
        x = self.fc_hidden(x)
        f = self.activation(x)
        c = self.fc_out(f)
        if return_feature:
            return [c, f]
        else:
            return c

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


def mlp(num_classes=10, dropout=0.0):
    model = MLP3(num_classes=num_classes, dropout=dropout)
    return model
