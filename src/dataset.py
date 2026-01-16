import os
from torch.utils.data import Dataset
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

    - Uses torchvision for loading images + labels
    - Builds a 'paths' list by scanning the GTSRB directory, so we always
      have a non-empty path string.
    """

    def __init__(self, root: str, split: str, transform=None, download: bool = True):
        self.base_dataset = datasets.GTSRB(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )
        self.root = root
        self.split = split

        # Build paths in the same order as torchvision loads them
        self.paths = self._build_paths_list()

        # If you still want class names:
        self.classes = [gtsrb_class_names[i] for i in range(len(gtsrb_class_names))]

    def _build_paths_list(self):
        """
        Build a list of image file paths that matches the order of self.base_dataset.
        Torchvision loads train images from Final_Training/Images/<class_id>/*.ppm
        in sorted order, so we do the same.
        """
        img_root = os.path.join(self.root, "gtsrb", "GTSRB", "Training")

        all_paths = []
        # class folders: 00000, 00001, ..., 00042
        for class_id in sorted(os.listdir(img_root)):
            if not class_id.isdigit():
                continue
            class_folder = os.path.join(img_root, class_id)
            for fname in sorted(os.listdir(class_folder)):
                if fname.lower().endswith(".ppm"):
                    all_paths.append(os.path.join(class_folder, fname))

        if len(all_paths) != len(self.base_dataset):
            print("⚠️ Warning: number of paths != dataset length.")
            print("Paths:", len(all_paths), " | Dataset:", len(self.base_dataset))

        return all_paths

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]
        path = self.paths[idx]
        return img, label, path