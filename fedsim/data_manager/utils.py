from numpy import indices
from torch.utils.data import Dataset

class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.targets = self.subset.dataset.targets[self.subset.indices]
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
