import torch
from torch.distributions.multinomial import Multinomial
from torchvision import datasets
import argparse, os
import numpy as np
import random
from math import floor


def split_trainval(train_set, num_val):
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = floor(num_val // len(classes))
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    for c in classes:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        valid_images += [c_images[:n_labels_per_cls]]
        valid_labels += [c_labels[:n_labels_per_cls]]
        train_images += [c_images[n_labels_per_cls:]]
        train_labels += [c_labels[n_labels_per_cls:]]
    valid_set = {"images": np.concatenate(valid_images, 0), "labels": np.concatenate(valid_labels, 0)}
    train_set = {"images": np.concatenate(train_images, 0), "labels": np.concatenate(train_labels, 0)}
    return train_set, valid_set


def split_imbalance_l_u(train_set, n_labels=None, imb_factor=1.0, seed=0, n_unlabels=None, imb_unlabel=None):
    # NOTE: this function assume that train_set is shuffled.
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)

    # imbalanced class distribution, total sum of samples is equal to 'n_labels'
    cls_num = len(classes)

    if imb_factor > 1.0:
        if n_labels is not None:
            mu = imb_factor ** (- 1.0 / (cls_num - 1.0))
            img_max = floor(min(n_labels * (1 - mu) / (1 - mu**cls_num), len(images) / cls_num))
        else:
            img_max = len(images) / cls_num

        if n_unlabels is None: n_unlabels = len(images) - n_labels


        n_labels_per_cls = []
        n_unlabels_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (- cls_idx / (cls_num - 1.0)))
            n_labels_per_cls.append(round(num))

        # unlabel imbalance
        for cls_idx in range(cls_num):
            if imb_unlabel is None:
                num_u = min(len(labels[labels == cls_idx]) - n_labels_per_cls[cls_idx], floor(n_unlabels / cls_num))
            elif imb_unlabel == 'uniform':
                num_u = min(floor(len(images) / cls_num) - max(n_labels_per_cls), floor(n_unlabels / cls_num))
            elif imb_unlabel == 'same':
                mu_u = imb_factor ** (- 1.0 / (cls_num - 1.0))
                img_max_u = min(floor(n_unlabels * (1 - mu_u) / (1 - mu_u ** cls_num)), floor(len(images) / cls_num) - max(n_labels_per_cls))
                num_u = img_max_u * (imb_factor ** (- cls_idx / (cls_num - 1.0)))
            elif imb_unlabel == 'half':
                btw = 0.5
                mu_u = (btw * imb_factor) ** (- 1.0 / (cls_num - 1.0))
                img_max_u = min(floor(n_unlabels * (1 - mu_u) / (1 - mu_u ** cls_num)), floor(len(images) / cls_num) - max(n_labels_per_cls))
                num_u = img_max_u * ((btw * imb_factor) ** (- cls_idx / (cls_num - 1.0)))
            else:
                raise ValueError("{} is unknown unlabel imbalance method".format(imb_unlabel))
            n_unlabels_per_cls.append(floor(num_u))

        random.Random(seed).shuffle(n_labels_per_cls)
        random.Random(seed).shuffle(n_unlabels_per_cls)

    else:
        n_labels_per_cls = []
        n_unlabels_per_cls = []
        for cls_idx in range(cls_num):
            num = floor(n_labels / cls_num)
            n_labels_per_cls.append(num)
            num_u = len(labels[labels == cls_idx]) - num
            n_unlabels_per_cls.append(num_u)

    l_images = []
    l_labels = []
    u_images = []
    u_labels = []

    for c, n_label, n_unlabel in zip(classes, n_labels_per_cls, n_unlabels_per_cls):
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_label]]
        l_labels += [c_labels[:n_label]]
        u_images += [c_images[n_label:n_label+n_unlabel]]
        u_labels += [np.zeros_like(c_labels[n_label:n_label+n_unlabel]) - 1]    # dummy label

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}

    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}

    return l_train_set, u_train_set, n_labels_per_cls, n_unlabels_per_cls


def split_imbalance_toy_ul(train_set, n_labels=None, imb_factor=1.0, seed=0, n_unlabels=None):
    # NOTE: this function assume that train_set is shuffled.
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)

    # imbalanced class distribution, total sum of samples is equal to 'n_labels'
    cls_num = len(classes)

    if imb_factor > 1.0:
        if n_labels is not None:
            mu = imb_factor ** (- 1.0 / (cls_num - 1.0))
            img_max = floor(n_labels * (1 - mu) / (1 - mu**cls_num))
        else:
            img_max = len(images) / cls_num

        if n_unlabels is None: n_unlabels = len(images) - n_labels
        mu_u = imb_factor ** (- 1.0 / (cls_num - 1.0))
        img_max_u = floor(min(n_unlabels * (1 - mu_u) / (1 - mu_u ** cls_num), floor(n_unlabels / cls_num)))

        n_labels_per_cls = []
        n_unlabels_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (- cls_idx / (cls_num - 1.0)))
            n_labels_per_cls.append(round(num))
            num_u = img_max_u * (imb_factor ** (- cls_idx / (cls_num - 1.0)))
            n_unlabels_per_cls.append(round(num_u))

    else:
        n_labels_per_cls = []
        num = floor(n_labels // cls_num)
        [n_labels_per_cls.append(num) for _ in range(cls_num)]

    random.Random(seed).shuffle(n_labels_per_cls)
    random.Random(seed).shuffle(n_unlabels_per_cls)

    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    u_gt_labels = []
    for c, n_label, n_unlabel in zip(classes, n_labels_per_cls, n_unlabels_per_cls):
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_label]]
        l_labels += [c_labels[:n_label]]
        u_images += [c_images[n_label:n_label+n_unlabel]]
        u_labels += [np.zeros_like(c_labels[n_label:n_label+n_unlabel]) - 1]    # dummy label
        u_gt_labels += [c_labels[n_label:n_label + n_unlabel]]

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0), "gt_labels": np.concatenate(u_gt_labels, 0)}

    return l_train_set, u_train_set, n_labels_per_cls

def _load_svhn():
    splits = {}
    for split in ["train", "test", "extra"]:
        tv_data = datasets.SVHN(_DATA_DIR, split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split] = data
    return splits.values()

def _load_cifar10():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR10(_DATA_DIR, train, download=True)
        data = {}
        if train:
            data["images"] = tv_data.data
            data["labels"] = np.array(tv_data.targets)
        else:
            data["images"] = tv_data.data
            data["labels"] = np.array(tv_data.targets)
        splits["train" if train else "test"] = data
    return splits

def _load_cifar100():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR100(_DATA_DIR, train, download=True)
        data = {}
        if train:
            data["images"] = tv_data.data
            data["labels"] = np.array(tv_data.targets)
        else:
            data["images"] = tv_data.data
            data["labels"] = np.array(tv_data.targets)
        splits["train" if train else "test"] = data
    return splits

def gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", default=0, type=int, help="random seed")
    parser.add_argument("--dataset", "-d", default="cifar10", type=str, help="dataset name : [svhn, cifar10]")
    args = parser.parse_args()

    COUNTS = {
        "svhn": {"train": 73257, "test": 26032, "valid": 7326, "extra": 531131},
        "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
        "cifar100": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
        "imagenet_32": {
            "train": 1281167,
            "test": 50000,
            "valid": 50050,
            "extra": 0,
        },
    }

    _DATA_DIR = "./data"
    if not os.path.exists(_DATA_DIR):
        os.mkdir(_DATA_DIR)

    rng = np.random.RandomState(args.seed)

    validation_count = COUNTS[args.dataset]["valid"]

    extra_set = None  # In general, there won't be extra data.
    if args.dataset == "svhn":
        train_set, test_set, extra_set = _load_svhn()
    elif args.dataset == "cifar10":
        data_cifar10 = _load_cifar10()
        train_set, test_set = data_cifar10["train"], data_cifar10["test"]
        train_set["images"] = gcn(train_set["images"])
        test_set["images"] = gcn(test_set["images"])
        mean, zca_decomp = get_zca_normalization_param(train_set["images"])
        train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
        test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
        # N x H x W x C -> N x C x H x W
        train_set["images"] = np.transpose(train_set["images"], (0,3,1,2))
        test_set["images"] = np.transpose(test_set["images"], (0,3,1,2))
    elif args.dataset == "cifar100":
        data_cifar100 = _load_cifar100()
        train_set, test_set = data_cifar100["train"], data_cifar100["test"]
        train_set["images"] = gcn(train_set["images"])
        test_set["images"] = gcn(test_set["images"])
        mean, zca_decomp = get_zca_normalization_param(train_set["images"])
        train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
        test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
        # N x H x W x C -> N x C x H x W
        train_set["images"] = np.transpose(train_set["images"], (0,3,1,2))
        test_set["images"] = np.transpose(test_set["images"], (0,3,1,2))

    if extra_set is not None:
        extra_indices = rng.permutation(len(extra_set["images"]))
        extra_set["images"] = extra_set["images"][extra_indices]
        extra_set["labels"] = extra_set["labels"][extra_indices]

    if not os.path.exists(os.path.join(_DATA_DIR, args.dataset)):
        os.mkdir(os.path.join(_DATA_DIR, args.dataset))

    # unshuffled train set
    np.save(os.path.join(_DATA_DIR, args.dataset, "train"), train_set)
    np.save(os.path.join(_DATA_DIR, args.dataset, "test"), test_set)
    if extra_set is not None:
        np.save(os.path.join(_DATA_DIR, args.dataset, "extra"), extra_set)

