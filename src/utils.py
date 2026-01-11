import random
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


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

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

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

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# save top 5
def save_top5_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    csv_path: str,
    class_names: List[str],
):
    model.eval()
    fieldnames = [
        "image_path",
        "true_label_index",
        "true_label_name",
        "top1_index",
        "top1_prob",
        "top2_index",
        "top2_prob",
        "top3_index",
        "top3_prob",
        "top4_index",
        "top4_prob",
        "top5_index",
        "top5_prob",
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
                    row = {
                        "image_path": paths[i],
                        "true_label_index": true_idx,
                        "true_label_name": class_names[true_idx]
                        if 0 <= true_idx < len(class_names)
                        else str(true_idx),
                    }
                    for k in range(5):
                        idx_k = int(top5_indices[i, k].item())
                        prob_k = float(top5_probs[i, k].item())
                        row[f"top{k+1}_index"] = idx_k
                        row[f"top{k+1}_prob"] = prob_k
                    writer.writerow(row)
