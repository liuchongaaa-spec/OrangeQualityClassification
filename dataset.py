import json
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OrangeDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def save_class_indices(class_names: List[str], save_path="class_indices.json"):
    index_dict = {str(i): name for i, name in enumerate(class_names)}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=4)
    print(f"class_indices.json saved: {index_dict}")


def split_dataset(data_root: Path, val_rate=0.2):

    assert data_root.exists(), f"{data_root} not found."

    # 自动读取类别目录
    class_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    print(f"Detected classes: {class_names}")

    # 自动保存 class_indices.json（与之前项目一致）
    save_class_indices(class_names)

    train_paths, val_paths = [], []
    train_labels, val_labels = [], []

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_root / class_name

        image_files = sorted([
            p for p in class_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ])

        random.seed(0)
        val_count = int(len(image_files) * val_rate)
        val_samples = set(random.sample(image_files, val_count))

        for img_path in image_files:
            if img_path in val_samples:
                val_paths.append(img_path)
                val_labels.append(class_idx)
            else:
                train_paths.append(img_path)
                train_labels.append(class_idx)

    return train_paths, train_labels, val_paths, val_labels, len(class_names)

def create_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    input_size: int,
    pin_memory=True,
):
    data_root = Path(data_root)

    train_paths, train_labels, val_paths, val_labels, num_classes = split_dataset(
        data_root, val_rate=0.2
    )

    # 数据增强（与当前使用版本一致）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])

    train_dataset = OrangeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = OrangeDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, num_classes
