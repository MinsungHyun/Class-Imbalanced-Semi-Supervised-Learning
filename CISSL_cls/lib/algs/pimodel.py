import torch
import torch.nn as nn
import torch.nn.functional as F

class PiModel(nn.Module):
    def __init__(self, scl_weight=None):
        super().__init__()
        self.scl_weight = scl_weight

    def forward(self, x, y, model, mask):
        # NOTE:
        # stochastic transformation is embeded in forward function
        # so, pi-model is just to calculate consistency between two outputs
        model.update_batch_stats(False)
        y_hat = model(x)
        model.update_batch_stats(True)

        if self.scl_weight is not None:
            target_weights = torch.stack(list(map(lambda t: self.scl_weight[t.data], y.max(1)[1])))
            return (target_weights * F.mse_loss(y_hat.softmax(1), y.softmax(1).detach(), reduction="none").mean(1) * mask).mean()
        else:
            return (F.mse_loss(y_hat.softmax(1), y.softmax(1).detach(), reduction="none").mean(1) * mask).mean()
