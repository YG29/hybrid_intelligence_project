import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import pandas as pd
from typing import Dict, List
from torchvision import transforms
from src import configure, dataset, utils

# create an empty model
def create_model(num_classes: int):
    model = utils.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = utils.nn.Linear(in_features, num_classes)
    return model

def main():
    print(f"Using device: {configure.DEVICE}")

    # load test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ImageNet
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = dataset.GTSRBWithPaths(
        root=configure.DATA_ROOT,
        split="test",
        transform=transform,
        download=True,
    )

    num_classes = len(test_dataset.classes)
    test_loader = utils.DataLoader(
        test_dataset,
        batch_size=configure.BATCH_SIZE,
        shuffle=False,
        num_workers=configure.NUM_WORKERS,
    )

    print(f"Number of test samples: {len(test_dataset)}")

    # load ResNet18 and evaluate
    model = create_model(num_classes=num_classes)
    model_path = os.path.join("..", configure.MODEL_PATH)
    model.load_state_dict(utils.torch.load(model_path, map_location=configure.DEVICE))
    model.to(configure.DEVICE)
    model.eval()

    all_true: List[int] = []
    all_pred: List[int] = []

    test_rows: List[Dict] = []

    with utils.torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Predicting on test"):
            images = images.to(configure.DEVICE)
            outputs = model(images)
            probs = utils.torch.softmax(outputs, dim=1)

            top1_probs, top1_indices = probs.max(dim=1)
            top5_probs, top5_indices = probs.topk(5, dim=1)

            labels = labels.cpu()
            top1_indices = top1_indices.cpu()
            top1_probs = top1_probs.cpu()
            top5_probs = top5_probs.cpu()
            top5_indices = top5_indices.cpu()

            for i in range(images.size(0)):
                true_idx = int(labels[i].item())
                pred_idx = int(top1_indices[i].item())
                all_true.append(true_idx)
                all_pred.append(pred_idx)

                row = {
                    "image_path": paths[i],
                    "true_label_index": true_idx,
                    "true_label_name": test_dataset.classes[true_idx],
                    "top1_index": pred_idx,
                    "top1_name": test_dataset.classes[pred_idx],
                    "top1_prob": float(top1_probs[i].item()),
                }

                # add top2..top5
                for k in range(5):
                    idx_k = int(top5_indices[i, k].item())
                    prob_k = float(top5_probs[i, k].item())
                    row[f"top{k+1}_index"] = idx_k
                    row[f"top{k+1}_name"] = test_dataset.classes[idx_k]
                    row[f"top{k+1}_prob"] = prob_k

                test_rows.append(row)

    # metrics: accuracy and F1
    acc = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average="macro")

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1_macro:.4f}")

    # Save predictions CSV for delegation
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(configure.TEST_PREDICTIONS_CSV, index=False)
    print(f"Saved test predictions to {configure.TEST_PREDICTIONS_CSV}")

    # ---- 4. Run delegation model on test samples ----
    print("\nRunning delegation model on test set...")

    # Feature extractor (frozen ResNet backbone)
    feat_extractor = utils.ResNetFeatureExtractor(model_path, num_classes=num_classes).to(configure.DEVICE)
    feat_extractor.eval()

    # Delegation head
    delegation_head = utils.DelegationHead(
        feature_dim=512,
        prob_dim=5,
        hidden_dim1=256,
        hidden_dim2=128,
        dropout=0.3,
    ).to(configure.DEVICE)
    delegation_head.load_state_dict(utils.torch.load(configure.DELEGATION_PATH, map_location=configure.DEVICE))
    delegation_head.eval()

    # We’ll need images + top5 probs again to feed delegation model.
    # Instead of re-running loader, we’ll re-loop test_loader with feature extractor + delegation_head.

    delegation_rows: List[Dict] = []

    with utils.torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Delegation on test"):
            images = images.to(configure.DEVICE)
            outputs = model(images)
            probs = utils.torch.softmax(outputs, dim=1)
            top5_probs, top5_indices = probs.topk(5, dim=1)

            feats = feat_extractor(images)

            top5_probs_sub = top5_probs[:, :5]  # [B,5]
            delegation_logits = delegation_head(feats, top5_probs_sub)
            delegation_pred = delegation_logits.argmax(dim=1)  # 0=model, 1=human_assist

            labels = labels.cpu()
            paths = list(paths)
            top5_probs = top5_probs.cpu()
            top5_indices = top5_indices.cpu()
            delegation_pred = delegation_pred.cpu()

            for i in range(images.size(0)):
                true_idx = int(labels[i].item())
                row = {
                    "image_path": paths[i],
                    "true_label_index": true_idx,
                    "true_label_name": test_dataset.classes[true_idx],
                }

                # top-5 info
                for k in range(5):
                    idx_k = int(top5_indices[i, k].item())
                    prob_k = float(top5_probs[i, k].item())
                    row[f"top{k+1}_index"] = idx_k
                    row[f"top{k+1}_name"] = test_dataset.classes[idx_k]
                    row[f"top{k+1}_prob"] = prob_k

                row["delegate_label_pred"] = int(delegation_pred[i].item())  # 0=model, 1=human_assist
                delegation_rows.append(row)

    delegation_df = pd.DataFrame(delegation_rows)

    # Split into two CSVs: one for human, one for model
    human_df = delegation_df[delegation_df["delegate_label_pred"] == 1].copy()
    model_df = delegation_df[delegation_df["delegate_label_pred"] == 0].copy()

    human_df.to_csv(configure.TEST_HUMAN_CSV, index=False)
    model_df.to_csv(configure.TEST_MODEL_CSV, index=False)

    print(f"Saved human-assist test cases to: {configure.TEST_HUMAN_CSV} (n={len(human_df)})")
    print(f"Saved model-handled test cases to: {configure.TEST_MODEL_CSV} (n={len(model_df)})")


if __name__ == "__main__":
    main()