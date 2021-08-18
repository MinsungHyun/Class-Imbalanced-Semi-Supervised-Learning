#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse, math, time, json, os
import numpy as np

from lib import wrn, lenet, transform, utils
from config import config
from build_dataset import split_imbalance_l_u, split_trainval

parser = argparse.ArgumentParser()
# data args
parser.add_argument("--dataset", default="cifar10", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--validation", default=25000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--seed", default=0, type=int, help="random seed (default: 0)")
parser.add_argument("--root", default="data", type=str, help="dataset dir")
parser.add_argument("--output", default="./exp_res", type=str, help="output dir")
parser.add_argument("--save_dir", default="./save_models", type=str, help="save dir")
# algorithm args
parser.add_argument("--alg", default="supervised", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--nlabels", default=4000, type=int, help="the number of labeled data")
parser.add_argument("--nunlabels", default=None, type=int, help="the number of unlabeled data, default: None")
parser.add_argument("--imb-unlabel", default=None, type=str, help="imbalance type of unlabeled samples: [None, uniform, same, half]")
parser.add_argument("--reweight", default='None',type=str, help="reweight type of labeled loss: [None, focal, inverse, cls_bal]")
parser.add_argument("--sntg", default=0, type=float, help="coefficient of SNTG. 0.4 * consistency_loss_coef.")
parser.add_argument("--imb-factor", default=100, type=float, help="class imbalance factor (default: 10).")
parser.add_argument("--cb-beta", default=0, type=float, help="hyperparameter of class-balanced loss (default: 0.9999).")
parser.add_argument("--scl", default=0, type=float, help="hyperparameter of suppressed consistency loss (default: 0).")
# model args
parser.add_argument("--model", default="WRN", type=str, help="network model : [CNN13, WRN, MLP]")
parser.add_argument("--optimizer", default="nesterov", type=str, help="optimizer : [adam, nesterov]")
parser.add_argument("--dropout", default=0, type=float, help="dropout rate for CNN13 model")
parser.add_argument("--widen-factor", default=2, type=int, help="widen-factor for WRN (default: 2)")
parser.add_argument("--lr-schedule", default="cos_rampdown", type=str, help="lr schedule : [decay, cos_rampdown]")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326, "extra": 531131},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0}
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

if args.reweight != "None":
    exp_name += args.reweight + "_"

if args.cb_beta > 0:
    exp_name += 'cb_loss_'
    condition["cb-beta"] = args.cb_beta
if args.scl > 0:
    exp_name += 'scl_' + str(args.scl) + '_'
    condition["scl"] = args.scl
if args.imb_unlabel is not None:
    exp_name += args.imb_unlabel + '_'

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)

train_set = dataset_cfg["dataset"](args.root, "train").dataset

# permute index of training set
rng = np.random.RandomState(args.seed)
indices = rng.permutation(len(train_set["images"]))
train_set["images"] = train_set["images"][indices]
train_set["labels"] = train_set["labels"][indices]

# split training set into training and validation
validation_count = COUNTS[args.dataset]["valid"]
train_set, val_dataset = split_trainval(train_set, validation_count)
l_train_dataset, u_train_dataset, n_labels_per_class, n_unlabels_per_class = \
            split_imbalance_l_u(train_set, args.nlabels, args.imb_factor, args.seed, args.nunlabels, args.imb_unlabel)
l_train_dataset = dataset_cfg["train_dataset"](l_train_dataset)
u_train_dataset = dataset_cfg["train_dataset"](u_train_dataset)
val_dataset = dataset_cfg["train_dataset"](val_dataset)
test_dataset = dataset_cfg["dataset"](args.root, "test")

print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {
    "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
    "validation":len(val_dataset), "test":len(test_dataset)
}

# class-balanced loss weight
n_labels_per_class = torch.Tensor(n_labels_per_class).cuda()
cb_weight = (1 - args.cb_beta) / (1 - args.cb_beta ** n_labels_per_class)

if args.scl > 0:
    scl_weight = args.scl ** (1.0 - n_labels_per_class / n_labels_per_class.max())
else:
    scl_weight = None


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


shared_cfg = config["shared"]
if args.alg == "supervised":
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
    )
else:
    # batch size = 0.5 x batch size
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
    )

print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

exp_name += str(args.model) + "_"
exp_name += str(args.optimizer) + "_"

u_loader = DataLoader(
    u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
)

val_loader = DataLoader(val_dataset, 100, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 100, shuffle=False, drop_last=False)

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

if args.em > 0:
    print("entropy minimization : {}".format(args.em))
    exp_name += "em_"
    condition["entropy_minimization"] = args.em

if args.model == 'CNN13':
    model = lenet.CNN13(dataset_cfg["num_classes"], args.dropout, transform_fn).to(device)
elif args.model == 'WRN':
    model = wrn.WRN(args.widen_factor, dataset_cfg["num_classes"], transform_fn).to(device)
else:
    raise ValueError("{} is unknown model".format(args.model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"], weight_decay=alg_cfg["weight_decay"])
elif args.optimizer == 'nesterov':
    optimizer = optim.SGD(model.parameters(), lr=alg_cfg["lr"], momentum=alg_cfg["momentum"], weight_decay=alg_cfg["weight_decay"], nesterov=True)
else:
    raise ValueError("{} is unknown optimizer".format(args.optimizer))

trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT
    ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1, scl_weight=scl_weight)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT
    if args.model == 'CNN13':
        t_model = lenet.CNN13(dataset_cfg["num_classes"], args.dropout, transform_fn).to(device)
    elif args.model == 'WRN':
        t_model = wrn.WRN(args.widen_factor, dataset_cfg["num_classes"], transform_fn).to(device)
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


iteration = 0
maximum_val_acc = 0
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
    if args.alg != "ICT":
        if args.reweight == "None":
            cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()

        elif args.reweight == "inverse":
            inverse_weight = torch.sum(n_labels_per_class)/n_labels_per_class
            re_inverse_weight = inverse_weight * len(n_labels_per_class)/sum(inverse_weight)

            target_weights = torch.stack(list(map(lambda t: re_inverse_weight[t.data], target)))
            cls_loss = (target_weights * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()

        elif args.reweight == "cls_bal":
            target_weights = torch.stack(list(map(lambda t: cb_weight[t.data], target)))
            cls_loss = (target_weights * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()

        elif args.reweight == "focal":
            softmax_value = F.softmax(outputs, dim=-1)
            pt = softmax_value[:, target]
            cls_loss = ((1-pt)**1 * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()

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

    # validation
    if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
        with torch.no_grad():
            model.eval()
            if args.alg == "MT" or args.alg == "ICT": t_model.eval()
            print()
            print("### validation ###")
            sum_acc = 0.
            t_sum_acc = 0.
            s = time.time()
            for j, data in enumerate(val_loader):
                input, target = data
                input, target = input.to(device).float(), target.to(device).long()

                output = model(input)

                pred_label = output.max(1)[1]
                sum_acc += (pred_label == target).float().sum()

                if args.alg == "MT" or args.alg == "ICT":
                    t_output = t_model(input)
                    t_pred_label = t_output.max(1)[1]
                    t_sum_acc += (t_pred_label == target).float().sum()

                if ((j+1) % 10) == 0:
                    d_p_s = 10/(time.time()-s)
                    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                        j+1, len(val_loader), d_p_s, (len(val_loader) - j-1)/d_p_s
                    ), "\r", end="")
                    s = time.time()
            acc = sum_acc/float(len(val_dataset))
            print()
            print("validation accuracy : {}".format(acc))
            if args.alg == "MT" or args.alg == "ICT":
                t_acc = t_sum_acc / float(len(val_dataset))
                print("teacher validation accuracy : {}".format(t_acc))

            # test
            if maximum_val_acc < acc:
                print("### test ###")
                maximum_val_acc = acc
                sum_acc = 0.
                t_sum_acc = 0.
                s = time.time()
                for j, data in enumerate(test_loader):
                    input, target = data
                    input, target = input.to(device).float(), target.to(device).long()
                    output = model(input)
                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()

                    if args.alg == "MT" or args.alg == "ICT":
                        t_output = t_model(input)
                        t_pred_label = t_output.max(1)[1]
                        t_sum_acc += (t_pred_label == target).float().sum()

                    if ((j+1) % 10) == 0:
                        d_p_s = 100/(time.time()-s)
                        print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                            j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s
                        ), "\r", end="")
                        s = time.time()
                print()
                test_acc = sum_acc / float(len(test_dataset))
                print("test accuracy : {}".format(test_acc))

                if args.alg == "MT" or args.alg == "ICT":
                    t_test_acc = t_sum_acc / float(len(test_dataset))
                    print("teacher test accuracy : {}".format(t_test_acc))

                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save(model.state_dict(), os.path.join(args.save_dir, exp_name + "_best_model.pth"))
        model.train()
        if args.alg == "MT" or args.alg == "ICT": t_model.train()
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


print("test acc : {}".format(test_acc))
condition["test_acc"] = test_acc.item()

exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
    json.dump(condition, f)
