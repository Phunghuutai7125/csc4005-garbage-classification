"""
dataset.py – Load TrashNet / Garbage Classification dataset
Hỗ trợ: thư mục theo lớp (cardboard/, glass/, metal/, paper/, plastic/, trash/)
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def get_transforms(img_size: int = 224, augment: bool = True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = val_tf

    return train_tf, val_tf


class GarbageDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    augment: bool = True,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
    max_per_class: int = 0,
) -> Tuple[Dict[str, DataLoader], List[str], Dict[str, int]]:

    root = Path(data_dir)
    # Tìm các class folder
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"[dataset] Tìm thấy {len(class_names)} lớp: {class_names}")

    # Build samples
    all_samples = []
    rng = np.random.default_rng(seed)
    for cls_dir in class_dirs:
        imgs = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.jpeg")) + \
               list(cls_dir.rglob("*.png")) + list(cls_dir.rglob("*.JPG"))
        if max_per_class > 0 and len(imgs) > max_per_class:
            imgs = list(rng.choice(imgs, max_per_class, replace=False))
        for img in imgs:
            all_samples.append((img, class_to_idx[cls_dir.name]))

    rng.shuffle(all_samples)
    n = len(all_samples)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    print(f"[dataset] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # Lưu splits
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        df = pd.DataFrame([(str(p), l) for p, l in split], columns=["path", "label"])
        df.to_csv(splits_dir / f"{split_name}.csv", index=False)

    train_tf, val_tf = get_transforms(img_size, augment)
    train_ds = GarbageDataset(train_samples, transform=train_tf)
    val_ds   = GarbageDataset(val_samples,   transform=val_tf)
    test_ds  = GarbageDataset(test_samples,  transform=val_tf)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    loaders = {
        "train": DataLoader(train_ds, shuffle=True,  **kw),
        "val":   DataLoader(val_ds,   shuffle=False, **kw),
        "test":  DataLoader(test_ds,  shuffle=False, **kw),
    }
    return loaders, class_names, class_to_idx
