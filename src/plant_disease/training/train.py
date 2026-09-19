"""MobileNetV2 training pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from plant_disease.data.class_map import load_class_map
from plant_disease.model import _select_device

logger = logging.getLogger(__name__)

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
UNFREEZE_RATIO = 0.30


def build_dataloaders(
    data_dir: Path, batch_size: int, num_workers: int = 4
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf = transforms.Compose(
        [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomAffine(0, shear=0.1, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int) -> nn.Module:
    """Construct MobileNetV2 with top-30% backbone unfrozen for fine-tuning."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone = list(model.features.children())
    start_unfreeze = int(len(backbone) * (1 - UNFREEZE_RATIO))
    for i, layer in enumerate(backbone):
        unfreeze = i >= start_unfreeze and not isinstance(layer, nn.BatchNorm2d)
        for p in layer.parameters():
            p.requires_grad = unfreeze
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_loss / max(len(loader), 1), 100 * correct / max(total, 1)


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item()
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss / max(len(loader), 1), 100 * correct / max(total, 1)


def evaluate_full(model, loader, device, target_names: list[str]) -> None:
    """Print classification_report. scikit-learn imported lazily."""
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        logger.warning("scikit-learn 未安装，跳过 classification_report")
        return

    model.eval()
    preds: list[int] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)
            _, pred = torch.max(out, 1)
            preds.extend(pred.cpu().tolist())
            labels_all.extend(labels.tolist())
    print(classification_report(labels_all, preds, target_names=target_names, digits=4))


def plot_history(train_losses, train_accs, val_losses, val_accs, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, train_m, val_m in [
        ("loss", train_losses, val_losses),
        ("accuracy", train_accs, val_accs),
    ]:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_m) + 1), train_m, label=f"train {name}")
        plt.plot(range(1, len(val_m) + 1), val_m, label=f"val {name}")
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def main(args: argparse.Namespace) -> int:
    device = _select_device()
    logger.info("training on %s", device)

    data_dir = Path(args.data_dir)
    train_loader, val_loader, _ = build_dataloaders(data_dir, args.batch_size)

    num_classes = len(train_loader.dataset.classes)
    model = build_model(num_classes).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    early_stop = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    ckpt_path = Path(args.ckpt_out)

    for epoch in range(args.epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        logger.info(
            "epoch %d/%d  train_loss=%.4f acc=%.2f%%  val_loss=%.4f acc=%.2f%%",
            epoch + 1,
            args.epochs,
            tl,
            ta,
            vl,
            va,
        )

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), ckpt_path)
            logger.info("saved best checkpoint to %s", ckpt_path)
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= args.patience:
            logger.info("early stopping triggered")
            break

    rows = load_class_map(Path("resources/actual_classed_v2.txt"))
    target_names = [r.disease_name for r in rows] if rows else [str(i) for i in range(num_classes)]
    evaluate_full(model, val_loader, device, target_names)
    plot_history(train_losses, train_accs, val_losses, val_accs, Path("artifacts"))
    return 0
