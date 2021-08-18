import numpy as np


class FOURSPINS:
    def __init__(self, seed):
        X, y = fourspins(N=6000, num_spin=4, tangential_std=0.3, radial_std=0.1, rate=0.7, seed=seed, display=True)
        self.dataset = {"images": X, "labels": y}

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])

class FOURSPINSTrain:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


def fourspins(N=6000, num_spin=4, tangential_std=0.22, radial_std=0.1, rate=0.52, seed=0, display=True):
    rads = np.linspace(0, 2 * np.pi, num_spin + 1)
    rads = rads[:-1]

    num_per_spin = int(N / float(num_spin))

    np.random.seed(seed)
    data = np.random.randn(num_spin * num_per_spin, 2) * np.array([tangential_std, radial_std]) + np.array([1, 0])

    labels = []
    for idx in range(num_spin):
        labels.append(np.array([idx] * num_per_spin))
    labels = np.concatenate(labels)

    angles = rads[labels] + rate * np.exp(data[:, 0])

    for i in range(angles.shape[0]):
        trans_mat = [[np.cos(angles[i]), -np.sin(angles[i])],
                     [np.sin(angles[i]), np.cos(angles[i])]]
        data[i] = data[i].dot(trans_mat)

    return data, labels
