import random
import csv
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


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