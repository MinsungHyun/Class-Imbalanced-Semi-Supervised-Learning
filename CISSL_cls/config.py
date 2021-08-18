from lib.datasets import svhn, cifar10, twomoons, fourspins


shared_config = {
    "iteration" : 500000,
    "warmup" : 200000,
    "lr_decay_iter" : 400000,
    "lr_decay_factor" : 0.2,
    "batch_size" : 100,
    "lr_rampdown_iter" : 600000,
}
toy_config = {
    "iteration" : 5000,
    # "iteration" : 1000,
    "warmup" : 2000,
    "lr_decay_iter" : 4000,
    "lr_decay_factor" : 0.2,
    "batch_size" : 100,
    "lr_rampdown_iter" : 6000,
}
### dataset ###
svhn_config = {
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "train_dataset" : svhn.SVHNTrain,
    "num_classes" : 10,
}
cifar10_config = {
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "train_dataset" : cifar10.CIFAR10Train,
    "num_classes" : 10,
}
twomoons = {
    "transform" : [False, False, True],
    "dataset": twomoons.TWOMOONS,
    "train_dataset": twomoons.TWOMOONSTrain,
    "num_classes": 2,
    "lr": 0.1,
    "weight_decay": 0,
    "momentum":0.9,
}
fourspins = {
    "transform" : [False, False, True],
    "dataset": fourspins.FOURSPINS,
    "train_dataset": fourspins.FOURSPINSTrain,
    "num_classes": 4,
    "lr": 0.1,
    "weight_decay": 0,
    "momentum":0.9,
}
### algorithm ###
vat_config = {
    # virtual adversarial training
    "xi" : 1e-6,
    "eps" : {"cifar10":6, "svhn":1, "twomoons":1, "fourspins": 1, "cifar100":6,},
    "consis_coef" : 0.3,
    # "lr" : 3e-3,
    "lr" : 0.1,
    # "weight_decay" : 0,
    "weight_decay" : 1e-4,
    "momentum": 0.9,
}
pl_config = {
    # pseudo label
    "threashold" : 0.95,
    # "lr" : 3e-4,
    "lr" : 0.1,
    # "weight_decay" : 0,
    "weight_decay" : 1e-4,
    "consis_coef" : 1,
    "momentum": 0.9,
}
mt_config = {
    # mean teacher
    "ema_factor" : 0.95,
    # "lr" : 4e-4,
    "lr" : 0.1,
    # "weight_decay" : 0,
    "weight_decay" : 1e-4,
    "consis_coef" : 8,
    "momentum": 0.9,
    "loss": 'mse',  # mse, kld
    # "T": 1.0,   # softmax smoothing factor
}
pi_config = {
    # Pi Model
    # "lr" : 3e-4,
    "lr" : 0.1,
    "consis_coef" : 20.0,
    # "weight_decay" : 0,
    "weight_decay" : 1e-4,
    "momentum": 0.9,
}
ict_config = {
    # interpolation consistency training
    "ema_factor" : 0.999,
    # "lr" : 4e-4,
    "lr" : 0.1,
    "momentum" : 0.9,
    "weight_decay" : 1e-4,
    # "weight_decay" : 0,
    "consis_coef" : 100,
    "alpha" : 1.0,
}
supervised_config = {
    # "lr" : 3e-3,
    "lr" : 0.1,
    # "weight_decay" : 0,
    "weight_decay" : 1e-4,
    "momentum": 0.9,
}
sntg_config = {
    "K" : 0.4,
    "margin" : 1.0,
}

### master ###
config = {
    "shared" : shared_config,
    "svhn" : svhn_config,
    "cifar10" : cifar10_config,
    "VAT" : vat_config,
    "PL" : pl_config,
    "MT" : mt_config,
    "PI" : pi_config,
    "ICT" : ict_config,
    "supervised" : supervised_config,
    "SNTG" : sntg_config,
    "toy": toy_config,
    "twomoons": twomoons,
    "fourspins": fourspins,
}
