import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from src import configure
from src import utils
from src import dataset
# Tidy up the imports

def main():
    utils.set_seed(configure.RANDOM_SEED)

    device = utils.torch.device("cuda" if utils.torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms: resize & normalize like ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],   # ImageNet stds
        ),
    ])

    # Load the full train split of GTSRB with paths
    full_dataset = dataset.GTSRBWithPaths(
        root=configure.DATA_ROOT,
        split="train",
        transform=transform,
        download=False, ## since I already downloaded but if not change to True
    )

    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Total images in train split: {len(full_dataset)}")

    # Split into 80% finetuning pool, 20% first-round test
    total_len = len(full_dataset)
    finetune_len = int(0.8 * total_len)
    first_test_len = total_len - finetune_len

    gen = utils.torch.Generator().manual_seed(configure.RANDOM_SEED)
    finetune_dataset, first_test_dataset = random_split(
        full_dataset,
        [finetune_len, first_test_len],
        generator=gen,
    )

    # From the 80% finetune dataset, split 80/20 into train/val
    train_len = int(0.8 * finetune_len)
    val_len = finetune_len - train_len

    train_dataset, val_dataset = random_split(
        finetune_dataset,
        [train_len, val_len],
        generator=gen,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"First-round test size: {len(first_test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configure.BATCH_SIZE,
        shuffle=True,
        num_workers=configure.NUM_WORKERS,
        # pin_memory=True, # using mps so commented out
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configure.BATCH_SIZE,
        shuffle=False,
        num_workers=configure.NUM_WORKERS,
        # pin_memory=True,
    )
    first_test_loader = DataLoader(
        first_test_dataset,
        batch_size=configure.BATCH_SIZE,
        shuffle=False,
        num_workers=configure.NUM_WORKERS,
        # pin_memory=True,
    )

    # Create model
    model = utils.create_model(num_classes=num_classes)
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configure.LEARNING_RATE)

    best_val_acc = 0.0
    best_state_dict = None

    # Training loop
    for epoch in range(1, configure.NUM_EPOCHS + 1):
        train_loss, train_acc = utils.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = utils.evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{configure.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Keep best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # Save the trained model
    if best_state_dict is not None:
        utils.torch.save(best_state_dict, configure.MODEL_PATH)
        print(f"Saved best model (val acc={best_val_acc:.4f}) to: {configure.MODEL_PATH}")
        # load best weights back into model for test / CSV
        model.load_state_dict(best_state_dict)
    else:
        # Fallback: save last state
        utils.torch.save(model.state_dict(), configure.MODEL_PATH)
        print(f"Saved model (last epoch) to: {configure.MODEL_PATH}")

    # Evaluate on the first-round test set
    test_loss, test_acc = utils.evaluate(model, first_test_loader, criterion, device)
    print(
        f"\nFirst-round test results -> "
        f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}"
    )

    # Save top-5 predictions & probabilities for human auditing
    # Get class names from the underlying dataset
    class_names = full_dataset.classes
    utils.save_top5_predictions(
        model=model,
        dataloader=first_test_loader,
        device=device,
        csv_path=configure.OUTPUT_CSV,
        class_names=class_names,
    )
    print(f"Saved first-round predictions to: {configure.OUTPUT_CSV}")


if __name__ == "__main__":
    main()