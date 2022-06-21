import numpy as np
from torch.utils import data


class Subset(data.Dataset):
    r"""Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        if isinstance(indices, int) and indices == -1:
            self.indices = range(len(dataset))
        else:
            self.indices = indices
        self.transform = transform
        targets = np.array(dataset.targets)
        self.targets = targets[self.indices]
        # remove the transform function of the original dataset if transform
        # is provided avoiding double transform
        if transform is not None and self.dataset.transform is not None:
            self.dataset.transform = None

    def __getitem__(self, idx):
        if isinstance(idx, list):
            x, y = self.dataset[[self.indices[i] for i in idx]]
            if self.transform is None:
                return x, y
            return self.transform(x), y
        x, y = self.dataset[self.indices[idx]]
        if self.transform is None:
            return x, y
        return self.transform(x), y

    def __len__(self):
        return len(self.indices)
