#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse, math, time, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from lib import wrn, lenet, transform, utils, mlp
from config import config
from build_dataset import split_imbalance_toy_ul


def plot_decision_boundary(model, X_l, y_l, X_u, y_u, color="prob", colorbar=False):
    # Set min and max values and give it some padding
    x_min, x_max = X_u[:, 0].min() - .5, X_u[:, 0].max() + .5
    y_min, y_max = X_u[:, 1].min() - .5, X_u[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    xy_input = np.c_[xx.ravel(), yy.ravel()]
    z_output = model(torch.tensor(xy_input).cuda().float())
    m = nn.Softmax(dim=1)

    if color=="prob":
        Z = m(z_output).max(1)[0].cpu().numpy()  # argmax class prob.
        Z = Z.reshape(xx.shape)
        # Z = 1.0 - Z # reverse
        ct = plt.contourf(xx, yy, Z, cmap=plt.cm.hot, vmin=0., vmax=1.)
        cb = plt.cm.ScalarMappable(cmap=cm.hot)  # afmhot
        cb.set_array(Z)
        cb.set_clim(0., 1.)
        if colorbar:
            plt.colorbar(cb, boundaries=np.linspace(0, 1, 21))

    elif color == "argmax":
        Z = z_output.max(1)[1].cpu().numpy()  # argmax class
        Z = Z.reshape(xx.shape)
        ct = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, vmin=0., vmax=float(z_output.shape[1]), alpha=0.9)
        cb = plt.cm.ScalarMappable(cmap=cm.Spectral)
        cb.set_array(Z)
        cb.set_clim(0., float(z_output.shape[1]))
        if colorbar:
            plt.colorbar(cb, boundaries=np.linspace(0, float(z_output.shape[1])-1., z_output.shape[1]+1.))
        # plt.colorbar(ct)

    plt.scatter(X_u[:, 0], X_u[:, 1], s=1, c='gray', alpha=0.3)
    plt.scatter(X_l[:, 0], X_l[:, 1], s=50, c=y_l, cmap=cm.jet)
    plt.axis("off")
    

parser = argparse.ArgumentParser()
# data args
parser.add_argument("--dataset", default="twomoons", type=str, help="dataset name : [twomoons, fourspins]")
parser.add_argument("--validation", default=100, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--seed", default=0, type=int, help="random seed (default: 0)")
parser.add_argument("--root", default="data", type=str, help="dataset dir")
parser.add_argument("--output", default="./exp_res", type=str, help="output dir")
parser.add_argument("--db-output", default="./figs/decision_boundary", type=str, help="decision boundary output dir")
parser.add_argument("--save_dir", default="./save_models", type=str, help="save dir")
# algorithm args
parser.add_argument("--alg", default="supervised", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--nlabels", default=12, type=int, help="the number of labeled data")
parser.add_argument("--nunlabels", default=5000, type=int, help="the number of unlabeled data, default: None")
parser.add_argument("--sntg", default=0, type=float, help="coefficient of SNTG. 0.4 * consistency_loss_coef.")
parser.add_argument("--imb-factor", default=100, type=float, help="class imbalance factor (default: 10).")
parser.add_argument("--cb-beta", default=0, type=float, help="hyperparameter of class-balanced loss (default: 0.9999).")
parser.add_argument("--scl", default=0, type=float, help="hyperparameter of suppressed consistency loss (default: 0).")
# model args
parser.add_argument("--model", default="MLP", type=str, help="network model : [CNN13, WRN, MLP]")
parser.add_argument("--optimizer", default="nesterov", type=str, help="optimizer : [adam, nesterov]")
parser.add_argument("--dropout", default=0, type=float, help="dropout rate for CNN13 model")
parser.add_argument("--widen-factor", default=2, type=int, help="widen-factor for WRN (default: 2)")
parser.add_argument("--lr-schedule", default="decay", type=str, help="lr schedule : [decay, cos_rampdown]")
parser.add_argument('--gif', default=False, action='store_true', help='plot figure for gif')
parser.add_argument('--colorbar', default=False, action='store_true', help='plot with colorbar')

args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326, "extra": 531131},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
    "twomoons": {"train": 6000},
    "fourspins": {"train": 6000},
}

condition = {}
exp_name = ""

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
condition["model"] = args.model
condition["optimizer"] = args.optimizer
condition["lr-schedule"] = args.lr_schedule
condition["imbalance-factor"] = args.imb_factor
exp_name += str(args.dataset) + "_"
exp_name += str(args.nlabels) + "_"
if args.nunlabels != None:
    exp_name += str(args.nunlabels) + "_"
exp_name += 'imb_' + str(args.imb_factor) + "_"
exp_name += 'seed_' + str(args.seed) + "_"
if args.cb_beta > 0:
    exp_name += 'cb_loss_'
    condition["cb-beta"] = args.cb_beta
if args.scl > 0:
    exp_name += 'scl_' + str(args.scl) + '_'
    condition["scl"] = args.scl

if not os.path.exists(args.db_output):
    os.mkdir(args.db_output)

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)

train_set = dataset_cfg["dataset"](args.seed).dataset

# permute index of training set
rng = np.random.RandomState(args.seed)
indices = rng.permutation(len(train_set["images"]))
train_set["images"] = train_set["images"][indices]
train_set["labels"] = train_set["labels"][indices]

# unlabel imbalance
l_train_dataset, u_train_dataset, n_labels_per_class = split_imbalance_toy_ul(train_set, args.nlabels, args.imb_factor, args.seed, args.nunlabels)
exp_name += 'unlabel_imb_'

# major, minor class
major_label = torch.argmax(torch.tensor(n_labels_per_class)).item()
minor_label = torch.argmin(torch.tensor(n_labels_per_class)).item()


l_train_dataset = dataset_cfg["train_dataset"](l_train_dataset)
u_train_dataset = dataset_cfg["train_dataset"](u_train_dataset)
val_dataset = dataset_cfg["train_dataset"](train_set)

print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}".format(len(val_dataset)))

condition["number_of_data"] = {
    "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),"validation":len(val_dataset),
}

# class-balanced loss weight
n_labels_per_class = torch.Tensor(n_labels_per_class).cuda()
cb_weight = (1 - args.cb_beta) / (1 - args.cb_beta ** n_labels_per_class)
if args.scl > 0:
    label_weight = n_labels_per_class.max() / n_labels_per_class
    scl_weight = n_labels_per_class / n_labels_per_class.max()
else:
    scl_weight = None
    label_weight = n_labels_per_class.max() / n_labels_per_class


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


shared_cfg = config["toy"]
if args.alg == "supervised":
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
    )
else:
    # batch size = 0.5 x batch size
    l_loader = DataLoader(
        l_train_dataset, int(shared_cfg["batch_size"]*0.2), drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * int(shared_cfg["batch_size"]*0.2))
    )

print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg


if args.alg == 'MT': exp_name += str(alg_cfg['ema_factor']) + '_'
exp_name += str(args.model) + "_"
exp_name += str(args.optimizer) + "_"

u_loader = DataLoader(
    u_train_dataset, int(shared_cfg["batch_size"]*0.8), drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * int(shared_cfg["batch_size"]*0.8))
)

val_loader = DataLoader(val_dataset, 6000, shuffle=False, drop_last=False)

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

if args.em > 0:
    print("entropy minimization : {}".format(args.em))
    exp_name += "em_"
    condition["entropy_minimization"] = args.em

if args.model == 'CNN13':
    model = lenet.CNN13(dataset_cfg["num_classes"], args.dropout, transform_fn).to(device)
elif args.model == 'WRN':
    model = wrn.WRN(args.widen_factor, dataset_cfg["num_classes"], transform_fn).to(device)
elif args.model == 'MLP':
    model = mlp.MLP(dataset_cfg["num_classes"], transform_fn).to(device)
else:
    raise ValueError("{} is unknown model".format(args.model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=dataset_cfg["lr"], weight_decay=dataset_cfg["weight_decay"])
elif args.optimizer == 'nesterov':
    optimizer = optim.SGD(model.parameters(), lr=dataset_cfg["lr"], momentum=dataset_cfg["momentum"], weight_decay=dataset_cfg["weight_decay"], nesterov=True)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=dataset_cfg["lr"], momentum=dataset_cfg["momentum"], weight_decay=dataset_cfg["weight_decay"], nesterov=False)
else:
    raise ValueError("{} is unknown optimizer".format(args.optimizer))

trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT
    ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT
    if args.model == 'CNN13':
        t_model = lenet.CNN13(dataset_cfg["num_classes"], args.dropout, transform_fn).to(device)
    elif args.model == 'WRN':
        t_model = wrn.WRN(args.widen_factor, dataset_cfg["num_classes"], transform_fn).to(device)
    elif args.model == 'MLP':
        t_model = mlp.MLP(dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT(t_model, alg_cfg["ema_factor"], alg_cfg["loss"], scl_weight=scl_weight)
elif args.alg == "PI": # PI Model
    from lib.algs.pimodel import PiModel
    ssl_obj = PiModel(scl_weight=scl_weight)
elif args.alg == "ICT": # interpolation consistency training
    from lib.algs.ict import ICT
    if args.model == 'CNN13':
        t_model = lenet.CNN13(dataset_cfg["num_classes"], args.dropout, transform_fn).to(device)
    elif args.model == 'WRN':
        t_model = wrn.WRN(args.widen_factor, dataset_cfg["num_classes"], transform_fn).to(device)
    elif args.model == 'MLP':
        t_model = mlp.MLP(dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))

if args.sntg > 0:
    from lib.algs.sntg import SNTGLoss
    sntg_cfg = config["SNTG"]
    sntg_obj = SNTGLoss(shared_cfg["batch_size"], sntg_cfg)
    exp_name += 'sntg_'
    condition["SNTG"] = sntg_cfg
    print("sntg parameters : ", sntg_cfg)


print()
iteration = 0
maximum_val_acc = 0
start_time = str(time.time())
s = time.time()
for l_data, u_data in zip(l_loader, u_loader):
    iteration += 1
    l_input, target = l_data
    l_input, target = l_input.to(device).float(), target.to(device).long()
    l_target = target

    if args.alg != "supervised": # for ssl algorithm
        u_input, dummy_target = u_data
        u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()

        target = torch.cat([target, dummy_target], 0)
        unlabeled_mask = (target == -1).float()

        inputs = torch.cat([l_input, u_input], 0)
        outputs, features = model(inputs, return_feature=True)

        # ramp up exp(-5(1 - t)^2)
        coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
        ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) * coef

        # SNTG loss
        if args.sntg > 0:
            sntg_loss = sntg_obj.loss(outputs.detach(), features, target) * coef * args.sntg

    else:
        outputs, features = model(l_input, return_feature=True)
        coef = 0
        ssl_loss = torch.zeros(1).to(device)

    # supervised loss
    if args.cb_beta >0:
        # match class-balanced weights to labels
        target_weights = torch.stack(list(map(lambda t: scl_weight[t.data], target)))
        cls_loss = (target_weights * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()
    elif args.alg != "ICT":
        cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
    elif args.alg == "ICT":
        class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1).cuda()
        mixed_input, target_a, target_b, l_lam = utils.mixup_data_sup(l_input, l_target, alg_cfg["alpha"])
        loss_func = utils.mixup_criterion(target_a, target_b, l_lam)
        output_mixed_l = model(mixed_input)
        cls_loss = loss_func(class_criterion, output_mixed_l) / l_input.size(0)

    loss = cls_loss + ssl_loss

    if args.em > 0:
        loss -= args.em * ((outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1) * unlabeled_mask).mean()
    if args.sntg > 0 :
        loss += sntg_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.alg == "MT" or args.alg == "ICT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    # display
    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
        print("iteration [{}/{}] cls loss : {:.5f}, SSL loss : {:.5f}, coef : {:.1f}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {:.4f}".format(
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]),
            "\r", end="")
        s = time.time()

    # lr decay
    if args.lr_schedule == 'decay':
        if iteration == shared_cfg["lr_decay_iter"]:
            optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]
    elif args.lr_schedule == 'cos_rampdown':
        assert shared_cfg["lr_rampdown_iter"] >= iteration
        optimizer.param_groups[0]["lr"] = alg_cfg["lr"] * utils.cosine_rampdown(iteration, shared_cfg["lr_rampdown_iter"])
    else:
        raise ValueError("{} is unknown lr schedule".format(args.lr_schedule))

    with torch.no_grad():
        model.eval()
        if args.alg == 'MT': t_model.eval()

        # validation
        if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
            with torch.no_grad():
                model.eval()
                print()
                print("### validation ###")
                sum_acc = 0.
                major_sum_acc = 0.
                minor_sum_acc = 0.

                s = time.time()
                for j, data in enumerate(val_loader):
                    input, target = data
                    input, target = input.to(device).float(), target.to(device).long()

                    output = model(input)

                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                    major_idx = torch.tensor(val_dataset.dataset['labels'] == major_label)
                    minor_idx = torch.tensor(val_dataset.dataset['labels'] == minor_label)
                    major_sum_acc += (pred_label == target).float()[major_idx].sum()
                    minor_sum_acc += (pred_label == target).float()[minor_idx].sum()

                acc = sum_acc / float(len(val_dataset))
                major_acc = major_sum_acc / float(sum(major_idx.float()))
                minor_acc = minor_sum_acc / float(sum(minor_idx.float()))
                print()
                print("validation accuracy : {}".format(acc))
                print("major class validation accuracy : {}".format(major_acc))
                print("minor class validation accuracy : {}".format(minor_acc))

        # plot
        if iteration == shared_cfg["iteration"]:
            # decision boundary
            plot_decision_boundary(model, l_train_dataset.dataset["images"], l_train_dataset.dataset["labels"],
                                   u_train_dataset.dataset["images"], u_train_dataset.dataset["gt_labels"], colorbar=args.colorbar)
            # plt.show()
            plt.savefig(os.path.join(args.db_output, exp_name) + 'total_{:.4f}_major_{:.4f}_minor_{:.4f}.png'.format(acc, major_acc, minor_acc), bbox_inches='tight')
            plt.close()

        # save figs for decision boundary gif
        if args.gif == True:
            if not os.path.exists(os.path.join(args.db_output, exp_name)):
                os.mkdir(os.path.join(args.db_output, exp_name))
            if (iteration % 10) == 0 or iteration == shared_cfg["iteration"]:
                plot_decision_boundary(model, l_train_dataset.dataset["images"], l_train_dataset.dataset["labels"],
                                       u_train_dataset.dataset["images"], u_train_dataset.dataset["gt_labels"], colorbar=True)
                plt.title('{:04d} iteration'.format(iteration))
                plt.savefig(os.path.join(args.db_output, exp_name, '{:04d}'.format(iteration)) + '.png', bbox_inches='tight')
                plt.close()

    model.train()
    if args.alg == 'MT': t_model.train()



