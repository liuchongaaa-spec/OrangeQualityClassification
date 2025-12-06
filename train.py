from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from dataset import create_dataloaders
from model import OrangeNetV1
from utils import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    save_metrics,
)


def validate_comprehensive(model, loader, criterion, device, desc="Val"):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    total_loss = 0.0
    num_samples = 0
    start = time.time()

    val_bar = tqdm(loader, desc=desc, ncols=80)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            num_samples += len(labels)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    fps = num_samples / (time.time() - start)

    val_loss = total_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        val_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    except:
        val_auc = 0.0

    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "loss": val_loss,
        "acc": val_acc,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "auc": val_auc,
        "fps": fps,
        "cm": cm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OrangeNetV1 for citrus quality classification")

    parser.add_argument("--data-path", type=str, default="D:\\GraduateProject\\Data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--cbam_reduction", type=int, default=16)
    parser.add_argument("--use_mid_fc", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


from pathlib import Path  # 顶部已经有的话就不要重复加

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"using {device} device.")

    # 把字符串路径转成 Path 对象
    data_root = Path(args.data_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = args.output_dir / "checkpoint_last.pth"
    ckpt_best = args.output_dir / "best.pth"
    metrics_path = args.output_dir / "metrics.json"
    csv_log = args.output_dir / "training_log.csv"

    tb_writer = SummaryWriter(log_dir=str(args.output_dir / "runs"))

    # 这里用 data_root，而不是 args.data_path
    train_loader, val_loader, num_classes = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        pin_memory=True,
    )

    print(f"using {len(train_loader.dataset)} images for training, "
          f"{len(val_loader.dataset)} images for validation.")


    model = OrangeNetV1(
        num_classes=num_classes,
        dropout_rate=args.dropout,
        cbam_reduction_ratio=args.cbam_reduction,
        use_mid_fc=args.use_mid_fc,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.0
    history = {"train": [], "val": []}


    if args.resume and ckpt_last.exists():
        last_epoch = load_checkpoint(model, optimizer, ckpt_last, map_location=device)
        start_epoch = last_epoch + 1
        print(f"Resuming from epoch {start_epoch}, best_acc={best_acc:.4f}")


    for epoch in range(start_epoch, args.epochs + 1):
        # -------- Train --------
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc=f"Train Epoch {epoch}"
        )

        # -------- Validate --------
        val_metrics = validate_comprehensive(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Val Epoch {epoch}"
        )

        scheduler.step()

        # Print (完全恢复旧版样式)
        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss: {train_metrics['loss']:.4f} train_acc: {train_metrics['acc']:.4f} "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['acc']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} "
            f"FPS: {val_metrics['fps']:.2f}"
        )

        # TensorBoard
        tb_writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
        tb_writer.add_scalar("Train/Acc", train_metrics["acc"], epoch)
        tb_writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
        tb_writer.add_scalar("Val/Acc", val_metrics["acc"], epoch)
        tb_writer.add_scalar("Val/F1", val_metrics["f1"], epoch)
        tb_writer.add_scalar("Val/AUC", val_metrics["auc"], epoch)

        # CSV log
        with open(csv_log, "a", encoding="utf-8") as f:
            if epoch == start_epoch:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_precision,val_recall,val_f1,val_auc,fps\n")
            f.write(
                f"{epoch},{train_metrics['loss']},{train_metrics['acc']},"
                f"{val_metrics['loss']},{val_metrics['acc']},"
                f"{val_metrics['precision']},{val_metrics['recall']},"
                f"{val_metrics['f1']},{val_metrics['auc']},{val_metrics['fps']}\n"
            )

        # Checkpoint
        save_checkpoint(model, optimizer, epoch, ckpt_last)

        # Best model
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(model, optimizer, epoch, ckpt_best)
            print(f"🔥 New Best Model Saved! Acc={best_acc:.4f}")

    # Save metrics
    save_metrics(
        {
            "best_acc": best_acc,
            "history": history,
            "args": vars(args),
        },
        metrics_path,
    )

    print("\nTraining Completed.")


if __name__ == "__main__":
    main()
