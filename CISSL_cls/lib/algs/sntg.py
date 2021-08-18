import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SNTGLoss():
    """ Smooth Neighbors on Teacher Graphs for Semi-Supervised Learning, https://arxiv.org/abs/1711.00258
            - Mini-batch size: n >> Total number of pairs: n x n (100 x 100)
            - Doubly Stochastic case, Total number of pairs: s(50)
    """
    def __init__(self, batch_size, sntg_cfg):
        # super(SNTGLoss, self).__init__(opt)
        # Initialize Variables
        self.n_batch = batch_size
        # self.mask = torch.zeros(self.n_batch).type('torch.cuda.LongTensor')
        self.coeff_embed = sntg_cfg["K"]  # @ https://github.com/xinmei9322/SNTG
        self.margin = sntg_cfg["margin"]  # @ Appendix A. Training details

    def loss(self, out, feat, label):
        # Update Mask
        # self.mask.fill_(0)
        # self.mask[:(self.n_batch // 2)] = 1.0
        self.mask = (label != -1).float()
        # Get Hard Targets
        target_hard = torch.argmax(out, dim=1).float()  # Hard Target (Prediction)
        merged_target = self.mask * label.float() + (1 - self.mask) * target_hard

        # shuffle data
        rand_idx = torch.randperm(self.n_batch)
        merged_target = merged_target[rand_idx]
        feat = feat[rand_idx]

        # Calculate SNTG Loss
        feat_distance_square = torch.mean(torch.pow(feat[:self.n_batch // 2] - feat[self.n_batch // 2:], exponent=2), dim=1)  # [50]
        feat_distance = torch.sqrt(feat_distance_square)  # [50]
        neighbor_var = torch.eq(merged_target[:self.n_batch // 2], merged_target[self.n_batch // 2:]).type('torch.cuda.FloatTensor')  # [50]
        pos = neighbor_var * feat_distance_square  # [50]
        neg = (1.0 - neighbor_var) * torch.pow(torch.max(torch.FloatTensor([0]).cuda(), self.margin - feat_distance), exponent=2)  # [50]
        loss_sntg = torch.mean(pos + neg)

        return loss_sntg
