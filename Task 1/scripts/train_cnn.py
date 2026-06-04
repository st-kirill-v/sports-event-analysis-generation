from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class SportsCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 128),
            nn.MaxPool2d(2),
            self._block(128, 256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.03),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def class_counts(dataset: datasets.ImageFolder, indices: list[int] | np.ndarray) -> np.ndarray:
    labels = np.array(dataset.targets)
    return np.bincount(labels[indices], minlength=len(dataset.classes))


def make_loaders(
    train_dir: Path,
    test_dir: Path,
    image_size: int,
    batch_size: int,
    val_size: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor]:
    train_tfms, eval_tfms = get_transforms(image_size)
    train_aug_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    train_eval_dataset = datasets.ImageFolder(train_dir, transform=eval_tfms)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tfms)

    if train_aug_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Train and test classes differ. "
            f"Train={train_aug_dataset.classes}, test={test_dataset.classes}"
        )

    labels = np.array(train_aug_dataset.targets)
    all_indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    train_counts = class_counts(train_aug_dataset, train_idx)
    weights = train_counts.sum() / np.maximum(train_counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(
        Subset(train_aug_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_eval_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, train_aug_dataset.classes, class_weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        logits = model(images)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                int(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN for Sports-1M frame classification.")
    parser.add_argument("--train-dir", default="data/frames/train")
    parser.add_argument("--test-dir", default="data/frames/test")
    parser.add_argument("--output-dir", default="runs/cnn")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes, class_weights = make_loaders(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SportsCNN(num_classes=len(classes), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    print(f"Device: {device}")
    print(f"Classes ({len(classes)}): {classes}")

    best_val_acc = 0.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_model_path,
            )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after epoch {epoch}. Best epoch: {best_epoch}.")
            break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    y_true, y_pred = predict(model, test_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "classes": classes,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, zero_division=0),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")

    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved best model and reports to {output_dir}")


if __name__ == "__main__":
    main()
