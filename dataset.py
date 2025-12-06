import json
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool = True, input_size: int = 224) -> transforms.Compose:
    if train:
        aug = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25),
        ]
    else:
        aug = [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]
    return transforms.Compose(aug)


def save_class_indices(dataset: datasets.ImageFolder, output_dir: Path) -> Path:
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "class_indices.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    return json_path


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
) -> Tuple[DataLoader, DataLoader, int]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    train_set = datasets.ImageFolder(train_dir, transform=build_transforms(train=True, input_size=input_size))
    val_set = datasets.ImageFolder(val_dir, transform=build_transforms(train=False, input_size=input_size))

    class_map_path = save_class_indices(train_set, data_root)
    print(f"Saved class indices to {class_map_path}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, len(train_set.classes)
