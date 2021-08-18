import sklearn.datasets


class TWOMOONS:
    def __init__(self, seed):
        X, y = sklearn.datasets.make_moons(6000, noise=0.15, random_state=seed)
        self.dataset = {"images": X, "labels": y}

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class TWOMOONSTrain:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])
