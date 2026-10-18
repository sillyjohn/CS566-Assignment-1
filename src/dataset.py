"""Dataset and transform utilities for CSCI 566 FashionMNIST subset."""

import csv
from pathlib import Path
import random
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import FashionMNIST


class IndependentRandomHorizontalFlip:
    """Random horizontal flip with an independent random number generator stream.

    Ensures that augmentation decisions do not alter the main PyTorch RNG stream
    or DataLoader sample shuffling order.
    """

    def __init__(self, p: float = 0.5, seed: int = 42) -> None:
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.rng.random() < self.p:
            return TF.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"IndependentRandomHorizontalFlip(p={self.p}, seed={self.seed})"


class FashionMNISTSubset(Dataset):
    """Course subset of FashionMNIST defined by a manifest CSV file.

    Each item returns a tuple of:
        (image_tensor, target_label, sample_id)
    where image_tensor is scaled to [0.0, 1.0], target_label is int (0..9),
    and sample_id is formatted as 'train:000123' or 'test:000123'.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Read manifest records
        self.records: list[tuple[str, int]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append((row["sample_id"], int(row["label"])))

        # Cache underlying raw datasets
        self.raw_train = FashionMNIST(root=str(self.data_dir), train=True, download=False)
        self.raw_test = FashionMNIST(root=str(self.data_dir), train=False, download=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        sample_id, label = self.records[idx]
        partition, raw_idx_str = sample_id.split(":")
        raw_idx = int(raw_idx_str)

        if partition == "train":
            img_tensor = self.raw_train.data[raw_idx]
        elif partition == "test":
            img_tensor = self.raw_test.data[raw_idx]
        else:
            raise ValueError(f"Unknown partition in sample_id: {sample_id}")

        img = Image.fromarray(img_tensor.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, label, sample_id


def get_transforms(condition: str, seed: int) -> tuple[Callable, Callable]:
    """Get train and evaluation transforms.

    Args:
        condition: 'baseline' or 'flip'.
        seed: Random seed for independent flip stream.

    Returns:
        (train_transform, eval_transform)
    """
    eval_transform = T.ToTensor()
    if condition == "flip":
        train_transform = T.Compose([
            IndependentRandomHorizontalFlip(p=0.5, seed=seed + 1000),
            T.ToTensor(),
        ])
    else:
        train_transform = T.ToTensor()

    return train_transform, eval_transform


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create DataLoader with deterministic generator when shuffle is True."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
