import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import create_dataloaders
from model import OrangeNetV1
from utils import accuracy, save_checkpoint, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OrangeNetV1 for citrus quality classification")
    parser.add_argument("--data_root", type=Path, required=True, help="Root directory containing train/ and val/ folders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--cbam_reduction", type=int, default=16)
    parser.add_argument("--use_mid_fc", action="store_true")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets) * images.size(0)

    total = len(loader.dataset)
    return {"loss": running_loss / total, "acc": running_acc / total}


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, targets) * images.size(0)

    total = len(loader.dataset)
    return {"loss": running_loss / total, "acc": running_acc / total}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, num_classes = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
    )

    model = OrangeNetV1(
        num_classes=num_classes,
        dropout_rate=args.dropout,
        cbam_reduction_ratio=args.cbam_reduction,
        use_mid_fc=args.use_mid_fc,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = {"train": [], "val": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f}")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(model, optimizer, epoch, args.output_dir / "best.pth")

    save_metrics(history, args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
