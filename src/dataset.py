from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets

class GTSRBWithPaths(Dataset):
    """
    Wraps torchvision.datasets.GTSRB to also return the image file path.
    """

    def __init__(self, root: str, split: str, transform=None, download: bool = True):
        self.base_dataset = datasets.GTSRB(
            root=root,
            split=split,
            transform=transform,
            download=download
        )
        # base_dataset.samples should contain (path, label) tuples
        # If not, we can fall back to empty paths.
        self.has_samples = hasattr(self.base_dataset, "samples")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]
        if self.has_samples:
            path, _ = self.base_dataset.samples[idx]
        else:
            path = ""
        return img, label, path

    @property
    def classes(self):
        # Class names if provided by the base dataset
        if hasattr(self.base_dataset, "classes"):
            return self.base_dataset.classes
        return [str(i) for i in range(len(set(self.base_dataset.targets)))]
