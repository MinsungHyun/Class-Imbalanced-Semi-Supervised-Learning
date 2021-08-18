import numpy as np
import os

class CIFAR10:
    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "cifar10", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class CIFAR10Train:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])