import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    extra: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    map_location: str | torch.device = "cpu",
) -> int:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from {path}")
    return int(checkpoint.get("epoch", 0))


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.numel()


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {output_path}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    desc: str = "Train",
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, targets) * batch_size

    total = len(loader.dataset)
    return {"loss": running_loss / total, "acc": running_acc / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, targets) * batch_size

    total = len(loader.dataset)
    return {"loss": running_loss / total, "acc": running_acc / total}
