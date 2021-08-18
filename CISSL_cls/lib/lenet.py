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


class CNN13(nn.Module):

    def __init__(self, num_classes=10, dropout=0.5, transform_fn=None):
        super(CNN13, self).__init__()

        # self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(dropout)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(dropout)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = weight_norm(nn.Linear(128, num_classes))

        self.transform_fn = transform_fn

    def forward(self, x, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)

        out = x
        ## layer 1-a###
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)

        ## layer 1-b###
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)

        ## layer 1-c###
        out = self.conv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)

        out = self.mp1(out)
        out = self.drop1(out)

        ## layer 2-a###
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)

        ## layer 2-b###
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)

        ## layer 2-c###
        out = self.conv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)

        out = self.mp2(out)
        out = self.drop2(out)

        ## layer 3-a###
        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)

        ## layer 3-b###
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)

        ## layer 3-c###
        out = self.conv3c(out)
        out = self.bn3c(out)
        out = self.activation(out)

        f = self.ap3(out)

        c = f.view(-1, 128)
        c = self.fc1(c)
        if return_feature:
            return [c, f]
        else:
            return c

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


def cnn13(num_classes=10, dropout=0.0):
    model = CNN13(num_classes=num_classes, dropout=dropout)
    return model


class CNN13Decoder(nn.Module):

    def __init__(self, dropout=0.5):
        super(CNN13Decoder, self).__init__()

        # 32 X 32
        # self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        # self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        # self.bn1a = nn.BatchNorm2d(128)
        # self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        # self.bn1b = nn.BatchNorm2d(128)
        # self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        # self.bn1c = nn.BatchNorm2d(128)
        # self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop1 = nn.Dropout(dropout)

        self.dconv1a = weight_norm(nn.ConvTranspose2d(128, 3, 3, padding=1))
        self.dbn1a = nn.BatchNorm2d(3)
        self.dconv1b = weight_norm(nn.ConvTranspose2d(128, 128, 3, padding=1))
        self.dbn1b = nn.BatchNorm2d(128)
        self.dconv1c = weight_norm(nn.ConvTranspose2d(128, 128, 3, padding=1))
        self.dbn1c = nn.BatchNorm2d(128)
        self.dmp1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.ddrop1 = nn.Dropout(dropout)

        # 16 X 16
        # self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        # self.bn2a = nn.BatchNorm2d(256)
        # self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        # self.bn2b = nn.BatchNorm2d(256)
        # self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        # self.bn2c = nn.BatchNorm2d(256)
        # self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop2 = nn.Dropout(dropout)

        self.dconv2a = weight_norm(nn.ConvTranspose2d(256, 128, 3, padding=1))
        self.dbn2a = nn.BatchNorm2d(128)
        self.dconv2b = weight_norm(nn.ConvTranspose2d(256, 256, 3, padding=1))
        self.dbn2b = nn.BatchNorm2d(256)
        self.dconv2c = weight_norm(nn.ConvTranspose2d(256, 256, 3, padding=1))
        self.dbn2c = nn.BatchNorm2d(256)
        self.dmp2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.ddrop2 = nn.Dropout(dropout)

        # 6 X 6
        # self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        # self.bn3a = nn.BatchNorm2d(512)
        # self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        # self.bn3b = nn.BatchNorm2d(256)
        # self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        # self.bn3c = nn.BatchNorm2d(128)
        # self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.dconv3a = weight_norm(nn.ConvTranspose2d(512, 256, 3, padding=0))
        self.dbn3a = nn.BatchNorm2d(256)
        self.dconv3b = weight_norm(nn.ConvTranspose2d(256, 512, 1, padding=0))
        self.dbn3b = nn.BatchNorm2d(512)
        self.dconv3c = weight_norm(nn.ConvTranspose2d(128, 256, 1, padding=0))
        self.dbn3c = nn.BatchNorm2d(256)
        self.dap3 = nn.Upsample(scale_factor=6, mode='nearest')

        # self.fc1 = weight_norm(nn.Linear(128, num_classes))
        # self.dfc1 = weight_norm(nn.Linear(num_classes, 128))

        # self.transform_fn = transform_fn

        # q
        self.q_mean = weight_norm(nn.Linear(128, 128))
        self.q_logvar = weight_norm(nn.Linear(128, 128))

        self.recon_loss = nn.MSELoss(reduction='sum')

    def forward(self, x):
        # x = x.view(-1, 128)
        # mean, logvar = self.q(x)
        # z = self.z(mean, logvar)
        # x = z.view(-1, 128, 1, 1)

        x = self.dap3(x)

        # layer 3
        x = self.dconv3c(x)
        x = self.dbn3c(x)
        x = self.activation(x)

        x = self.dconv3b(x)
        x = self.dbn3b(x)
        x = self.activation(x)

        x = self.dconv3a(x)
        x = self.dbn3a(x)
        x = self.activation(x)

        x = self.dmp2(x)
        x = self.ddrop2(x)

        # layer 2
        x = self.dconv2c(x)
        x = self.dbn2c(x)
        x = self.activation(x)

        x = self.dconv2b(x)
        x = self.dbn2b(x)
        x = self.activation(x)

        x = self.dconv2a(x)
        x = self.dbn2a(x)
        x = self.activation(x)

        x = self.dmp1(x)
        x = self.ddrop1(x)

        # layer 1
        x = self.dconv1c(x)
        x = self.dbn1c(x)
        x = self.activation(x)

        x = self.dconv1b(x)
        x = self.dbn1b(x)
        x = self.activation(x)

        x = self.dconv1a(x)

        return x

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, 128)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (torch.randn(std.size())).cuda()
        return eps.mul(std).add_(mean)

    # def reconstruction_loss(self, x_reconstructed, x):
    #     return nn.BCELoss(reduction='sum')(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)
