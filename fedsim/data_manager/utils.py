import numpy as np
from torch.utils import data


class SubsetWrapper(data.Dataset):

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        targets = np.array(self.subset.dataset.targets)
        self.targets = targets[self.subset.indices]
        # remove the transform function of the original dataset if transform
        # is provided avoiding double transform
        if transform is not None and self.subset.dataset.transform is not None:
            self.subset.dataset.transform = None

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class Subset(data.Subset):

    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)
        targets = np.array(dataset.targets)
        self.targets = targets[indices]
