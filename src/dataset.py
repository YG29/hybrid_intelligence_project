from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets

# Official GTSRB class names
gtsrb_class_names = {
     0: "Speed limit (20km/h)",
     1: "Speed limit (30km/h)",
     2: "Speed limit (50km/h)",
     3: "Speed limit (60km/h)",
     4: "Speed limit (70km/h)",
     5: "Speed limit (80km/h)",
     6: "End of speed limit (80km/h)",
     7: "Speed limit (100km/h)",
     8: "Speed limit (120km/h)",
     9: "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles > 3.5t",
}

class GTSRBWithPaths(Dataset):
    """
    Wraps torchvision.datasets.GTSRB to also return the image file path.
    Paths are kept exactly as torchvision provides them.
    Adds correct GTSRB .classes using official class names.
    Removes dependence on .targets attribute (not available in some versions).
    """

    def __init__(self, root: str, split: str, transform=None, download: bool = True):
        self.base_dataset = datasets.GTSRB(
            root=root,
            split=split,
            transform=transform,
            download=download
        )

        # torchvision only has .samples for the train split + some versions
        self.has_samples = hasattr(self.base_dataset, "samples")

        # Override .classes with real GTSRB class names
        self.classes = [gtsrb_class_names[i] for i in range(len(gtsrb_class_names))]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """
        Returns:
            img  : transformed image tensor
            label: integer class ID (0–42)
            path : image file path (string or "" if unavailable)
        """
        img, label = self.base_dataset[idx]

        if self.has_samples:
            # Most torchvision versions have this for the train split
            path, _ = self.base_dataset.samples[idx]
        else:
            # Fallback: no path available
            path = ""

        return img, label, path