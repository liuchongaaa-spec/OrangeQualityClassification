import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from {path}")
    return int(checkpoint.get("epoch", 0))


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        return correct / target.numel()


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {output_path}")
