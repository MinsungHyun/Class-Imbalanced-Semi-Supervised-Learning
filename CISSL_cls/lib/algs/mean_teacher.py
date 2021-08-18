import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MT(nn.Module):
    def __init__(self, model, ema_factor, loss='mse', scl_weight=None):
        super().__init__()
        self.model = model
        self.model.train()
        self.ema_factor = ema_factor
        self.global_step = 0
        self.loss = loss

        # scl weight
        if scl_weight is not None:
            self.scl_weight = scl_weight
        else:
            self.scl_weight = None

    def forward(self, x, y, model, mask):
        self.global_step += 1
        y_hat = self.model(x)
        model.update_batch_stats(False)
        y = model(x) # recompute y since y as input of forward function is detached
        model.update_batch_stats(True)
        if self.loss == 'mse':
            if self.scl_weight is not None:
                target_weights = torch.stack(list(map(lambda t: self.scl_weight[t.data], y.max(1)[1])))
                return (target_weights * F.mse_loss(y.softmax(1), y_hat.softmax(1).detach(), reduction="none").mean(1) * mask).mean()
            else:
                return (F.mse_loss(y.softmax(1), y_hat.softmax(1).detach(), reduction="none").mean(1) * mask).mean()
        elif self.loss == 'kld':
            return (F.kl_div(y.softmax(1).log(), y_hat.softmax(1).detach(), reduction="none").sum(1) * mask).mean()
        else:
            raise ValueError("{} is unknown loss type".format(self.loss))

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
        for ema_p, p in zip(self.model.parameters(), parameters):
            ema_p.data = ema_factor * ema_p.data + (1 - ema_factor) * p.data
