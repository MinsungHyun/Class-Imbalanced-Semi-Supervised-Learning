import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np


class Mine(nn.Module):
    def __init__(self, input_size=20, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # nn.init.normal_(self.fc1.weight, std=0.02)
        # nn.init.constant_(self.fc1.bias, 0)
        # nn.init.normal_(self.fc2.weight, std=0.02)
        # nn.init.constant_(self.fc2.bias, 0)
        # nn.init.normal_(self.fc3.weight, std=0.02)
        # nn.init.constant_(self.fc3.bias, 0)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    # joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    # marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    #     loss = - mi_lb

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et


def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        # index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        index = torch.randperm(data.size(0))
        batch = data[index]
        batch = batch.view(batch.size(0), -1)
    else:
        # joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        # marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        joint_index = torch.randperm(data.size(0))
        marginal_index = torch.randperm(data.size(0))
        # batch = np.concatenate([data[joint_index][:,0].reshape(-1,1), data[marginal_index][:,1].reshape(-1,1)], axis=1)
        batch = torch.cat((data[joint_index][:,0,:].unsqueeze(1), data[marginal_index][:,0,:].unsqueeze(1)), dim=1)
        batch = batch.view(batch.size(0), -1)
    return batch


def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data, batch_size=batch_size, sample_mode='joint'), sample_batch(data,batch_size=batch_size,sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        # if (i+1)%(log_freq)==0:
        #     print(result[-1])
    return result


def eval(data, mine_net, batch_size=100):
    # data is x or y
    result = list()
    ma_et = 1.

    joint, marginal = sample_batch(data, batch_size=batch_size, sample_mode='joint'), sample_batch(data,batch_size=batch_size,sample_mode='marginal')
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)

    return mi_lb.detach().cpu().numpy()


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    x = np.random.multivariate_normal(mean=[0, 0],
                                      cov=[[1, 0], [0, 1]],
                                      size=300)
    y = np.random.multivariate_normal(mean=[0, 0],
                                      cov=[[1, 0.8], [0.8, 1]],
                                      size=300)

    joint_data = sample_batch(y, batch_size=100, sample_mode='joint')
    sns.scatterplot(x=joint_data[:, 0], y=joint_data[:, 1], color='red')
    marginal_data = sample_batch(y, batch_size=100, sample_mode='marginal')
    sns.scatterplot(x=marginal_data[:, 0], y=marginal_data[:, 1])

    mine_net_indep = Mine().cuda()
    mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-3)
    result_indep = train(x, mine_net_indep, mine_net_optim_indep)

    result_indep_ma = ma(result_indep)
    print(result_indep_ma[-1])
    plt.plot(range(len(result_indep_ma)), result_indep_ma)

    mine_net_cor = Mine().cuda()
    mine_net_optim_cor = optim.Adam(mine_net_cor.parameters(), lr=1e-3)
    result_cor = train(y, mine_net_cor, mine_net_optim_cor)

    result_cor_ma = ma(result_cor)
    print(result_cor_ma[-1])
    plt.plot(range(len(result_cor_ma)), result_cor_ma)

