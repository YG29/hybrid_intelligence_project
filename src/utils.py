import random
import csv
import pandas as pd
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm
from src import configure
from PIL import Image


# set seeds just in case
def set_seed(seed: int = 192):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# model setup
def create_model(num_classes: int) -> nn.Module:
    # Load a pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# training loop
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap dataloader with tqdm
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # update tqdm display
        current_loss = running_loss / total
        current_acc = correct / total
        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.4f}"
        )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# evaluation
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # update tqdm display
            current_loss = running_loss / total
            current_acc = correct / total
            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_acc:.4f}"
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# save top 5
def save_top5_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    csv_path: str,
    class_names: List[str],
):
    """
    Run model on dataloader and save, for each sample:
      - image_path
      - true label index + name
      - top-5 predicted indices, names, and probabilities
    """
    model.eval()

    fieldnames = [
        "image_path",
        "true_label_index",
        "true_label_name",
        "top1_index", "top1_name", "top1_prob",
        "top2_index", "top2_name", "top2_prob",
        "top3_index", "top3_name", "top3_prob",
        "top4_index", "top4_name", "top4_prob",
        "top5_index", "top5_name", "top5_prob",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for images, labels, paths in dataloader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                top5_probs, top5_indices = probs.topk(5, dim=1)

                labels = labels.cpu()
                top5_probs = top5_probs.cpu()
                top5_indices = top5_indices.cpu()

                for i in range(images.size(0)):
                    true_idx = int(labels[i].item())
                    # guard in case of any mismatch
                    if 0 <= true_idx < len(class_names):
                        true_name = class_names[true_idx]
                    else:
                        true_name = str(true_idx)

                    row = {
                        "image_path": paths[i],
                        "true_label_index": true_idx,
                        "true_label_name": true_name,
                    }

                    for k in range(5):
                        idx_k = int(top5_indices[i, k].item())
                        prob_k = float(top5_probs[i, k].item())

                        if 0 <= idx_k < len(class_names):
                            name_k = class_names[idx_k]
                        else:
                            name_k = str(idx_k)

                        row[f"top{k+1}_index"] = idx_k
                        row[f"top{k+1}_name"] = name_k
                        row[f"top{k+1}_prob"] = prob_k

                    writer.writerow(row)



# util functions and classes for the delegation system

# Training data prep
# Load and Merge the csv
def load_merged_dataframe() -> pd.DataFrame:
    ann_df = pd.read_csv(configure.ANNOTATIONS_CSV)
    pred_df = pd.read_csv(configure.PREDICTIONS_CSV)

    # Merge on image_path
    df = ann_df.merge(pred_df, on="image_path", suffixes=("_ann", "_pred"))

    # Rename prediction column for consistency
    if "top1_prob_pred" in df.columns:
        df = df.rename(columns={"top1_prob_pred": "top1_prob"})

    # Make sure accepted_top1 is boolean
    df["accepted_top1"] = (
        df["accepted_top1"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # Target: 0 = model, 1 = human_assist
    df["delegate_label"] = df["accepted_top1"].apply(lambda x: 0 if x else 1)

    # Keep only needed columns
    keep_cols = [
        "image_path",
        "delegate_label",
        "top1_prob", "top2_prob", "top3_prob", "top4_prob", "top5_prob",
        # we'll see if more features needed
    ]
    df = df[keep_cols].copy()

    print("Total annotated samples:", len(df))
    print("Label distribution (0=model, 1=human_assist):")
    print(df["delegate_label"].value_counts())

    return df

# Feature extractor
class ResNetFeatureExtractor(nn.Module):
    """
    Wrap ResNet18 so forward(x) returns a 512-dim feature vector
    (after avgpool, before final fc).
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 43):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        backbone.load_state_dict(state_dict)

        # Drop final fc, keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)           # [B, 512, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 512]
        return feat


# Delegation data
class DelegationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        probs = torch.tensor(
            [
                row["top1_prob"],
                row["top2_prob"],
                row["top3_prob"],
                row["top4_prob"],
                row["top5_prob"],
            ],
            dtype=torch.float32,
        )

        label = int(row["delegate_label"])
        return img, probs, label

# Delegation head
class DelegationHead(nn.Module):
    """
    Input: image features (512) + top-5 probs (5) = 517-dim
    Output: logits for 2 classes: 0=model, 1=human_assist
    """

    def __init__(
        self,
        feature_dim: int = 512,
        prob_dim: int = 5,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        input_dim = feature_dim + prob_dim  # 517

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 2),
        )

    def forward(self, feat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, probs], dim=1)
        return self.net(x)

# Train/ Eval Loops
def del_train_one_epoch(
    feat_extractor: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    head.train()
    feat_extractor.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training (delegation)", leave=False)
    for images, probs, labels in pbar:
        images = images.to(device)
        probs = probs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            feats = feat_extractor(images)

        outputs = head(feats, probs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{correct / total:.4f}",
        )

    return running_loss / total, correct / total


def del_evaluate(
    feat_extractor: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    head.eval()
    feat_extractor.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Evaluating (delegation)", leave=False)
    with torch.no_grad():
        for images, probs, labels in pbar:
            images = images.to(device)
            probs = probs.to(device)
            labels = labels.to(device)

            feats = feat_extractor(images)
            outputs = head(feats, probs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

    return running_loss / total, correct / total
