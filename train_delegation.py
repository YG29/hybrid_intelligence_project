import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold

from src import configure
from src import utils


def main():
    utils.set_seed(configure.RANDOM_SEED)

    # 1) dataframe with labels + top-5 probs
    df = utils.load_merged_dataframe()

    # 2) dataset transform to ResNet ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = utils.DelegationDataset(df, transform=transform)
    num_samples = len(full_dataset)
    print(f"Total samples for delegation model: {num_samples}")

    # 3) feature extractor (frozen, reused across folds)
    feat_extractor = utils.ResNetFeatureExtractor(configure.MODEL_PATH, num_classes=43).to(configure.DEVICE)

    # 4) K-fold setup
    K = getattr(configure, "DELEGATION_K_FOLDS", 2)
    kf = KFold(n_splits=K, shuffle=True, random_state=configure.RANDOM_SEED)

    fold_results = []
    best_overall_state = None
    best_overall_acc = 0.0

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(num_samples)), start=1):
        print(f"\n===== Fold {fold_idx}/{K} =====")

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=configure.BATCH_SIZE,
            shuffle=True,
            num_workers=configure.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=configure.BATCH_SIZE,
            shuffle=False,
            num_workers=configure.NUM_WORKERS,
            pin_memory=True,
        )

        # New delegation head per fold
        head = utils.DelegationHead(
            feature_dim=512,
            prob_dim=5,
            hidden_dim1=256,
            hidden_dim2=128,
            dropout=0.3,
        ).to(configure.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(head.parameters(), lr=configure.DEL_LEARNING_RATE)

        PATIENCE = getattr(configure, "DELEGATION_PATIENCE", 3)
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, configure.DEL_NUM_EPOCHS + 1):
            train_loss, train_acc = utils.del_train_one_epoch(
                feat_extractor, head, train_loader, criterion, optimizer, configure.DEVICE
            )
            val_loss, val_acc = utils.del_evaluate(
                feat_extractor, head, val_loader, criterion, configure.DEVICE
            )

            print(
                f"[Fold {fold_idx}] Epoch [{epoch}/{configure.DEL_NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )


            # add in patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = head.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == PATIENCE:
                    print(f"[Fold {fold_idx}] Early stopping triggered at epoch {epoch}")
                    break

        print(f"[Fold {fold_idx}] Best val acc: {best_val_acc:.4f}")
        fold_results.append(best_val_acc)

        # Track best fold model overall (optional)
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_overall_state = best_state

    # 5) Summary of k-fold performance
    mean_acc = sum(fold_results) / len(fold_results)
    print("\n===== K-fold results =====")
    for i, acc in enumerate(fold_results, start=1):
        print(f"Fold {i}: {acc:.4f}")
    print(f"Mean val acc over {K} folds: {mean_acc:.4f}")

    # 6) Save best fold's delegation head
    os.makedirs("checkpoints", exist_ok=True)
    out_path = os.path.join("checkpoints", "delegation_head_kfold_best.pth")
    if best_overall_state is not None:
        torch.save(best_overall_state, out_path)
        print(
            f"Saved best delegation head across folds "
            f"(val acc={best_overall_acc:.4f}) to: {out_path}"
        )
    else:
        print("Warning: no best_overall_state found; nothing saved.")


if __name__ == "__main__":
    main()