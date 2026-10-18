"""Public interface and numerical correctness tests for CSCI 566 starter package.

Students can run these tests to verify compatibility and basic numerical correctness.
Tests that depend on student implementation tasks will skip until the corresponding
TODO functions are implemented.
"""

import copy
import csv
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.dataset import FashionMNISTSubset, create_dataloader, get_transforms
from src.models import SmallCNN
from src.training import evaluate, train_one_epoch
from src.utils import compute_actual_id_digest, export_predictions

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


class ToyDataset(Dataset):
    """Toy dataset returning (image, label, sample_id)."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, sample_ids: list[str]) -> None:
        self.images = images
        self.labels = labels
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx], self.sample_ids[idx]


def test_manifests_disjointness_and_counts() -> None:
    """Verify manifest row counts, class distributions, and train/val disjointness."""
    train_path = MANIFEST_DIR / "train.csv"
    val_path = MANIFEST_DIR / "validation.csv"
    test_path = MANIFEST_DIR / "test.csv"

    for p in [train_path, val_path, test_path]:
        assert p.is_file(), f"Manifest file missing: {p}"

    def read_manifest(p: Path) -> list[tuple[str, int]]:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [(r["sample_id"], int(r["label"])) for r in reader]

    train_data = read_manifest(train_path)
    val_data = read_manifest(val_path)
    test_data = read_manifest(test_path)

    # 1. Total counts
    assert len(train_data) == 10000, f"Expected 10000 train samples, got {len(train_data)}"
    assert len(val_data) == 2000, f"Expected 2000 val samples, got {len(val_data)}"
    assert len(test_data) == 2000, f"Expected 2000 test samples, got {len(test_data)}"

    # 2. Split disjointness: train and validation share zero IDs
    train_ids = set(sid for sid, _ in train_data)
    val_ids = set(sid for sid, _ in val_data)
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, f"Train and validation splits must be disjoint, found overlap: {overlap}"

    # 3. ID namespace conventions
    assert all(sid.startswith("train:") for sid in train_ids), "Train IDs must start with 'train:'"
    assert all(sid.startswith("train:") for sid in val_ids), "Validation IDs must start with 'train:'"
    test_ids = set(sid for sid, _ in test_data)
    assert all(sid.startswith("test:") for sid in test_ids), "Test IDs must start with 'test:'"

    # 4. Class balance: 1000/class train, 200/class val, 200/class test
    for c in range(10):
        c_train = sum(1 for _, lbl in train_data if lbl == c)
        c_val = sum(1 for _, lbl in val_data if lbl == c)
        c_test = sum(1 for _, lbl in test_data if lbl == c)
        assert c_train == 1000, f"Class {c} in train should have 1000 samples, got {c_train}"
        assert c_val == 200, f"Class {c} in val should have 200 samples, got {c_val}"
        assert c_test == 200, f"Class {c} in test should have 200 samples, got {c_test}"


def test_repeatable_pairing() -> None:
    """Verify that paired seeds produce identical initial weights and DataLoader sample ordering."""
    seed = 17

    # 1. Identical model weights for baseline and flip under the same seed
    torch.manual_seed(seed)
    model_baseline = SmallCNN()

    torch.manual_seed(seed)
    model_flip = SmallCNN()

    for p_base, p_flip in zip(model_baseline.parameters(), model_flip.parameters()):
        assert torch.equal(p_base, p_flip), "Initial weights must be identical across conditions for same seed"

    # 2. Identical DataLoader shuffling order
    dummy_x = torch.zeros(100, 1, 28, 28)
    dummy_y = torch.arange(100)
    ds = TensorDataset(dummy_x, dummy_y)

    loader1 = create_dataloader(ds, batch_size=16, shuffle=True, seed=seed)
    loader2 = create_dataloader(ds, batch_size=16, shuffle=True, seed=seed)

    order1 = [b[1].tolist() for b in loader1]
    order2 = [b[1].tolist() for b in loader2]
    assert order1 == order2, "DataLoader sample ordering must be identical for identical seed"


def test_model_forward_shape_and_range() -> None:
    """Verify SmallCNN output shape and finite values (Exercise 1)."""
    model = SmallCNN()
    x = torch.randn(4, 1, 28, 28)

    try:
        out = model(x)
    except NotImplementedError:
        pytest.skip("Exercise 1 not implemented: SmallCNN.forward")

    assert out.shape == (4, 10), f"Expected output shape (4, 10), got {out.shape}"
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf"


def test_evaluation_preserves_weights() -> None:
    """Verify evaluate() runs in eval mode and strictly preserves model parameters (Exercise 3)."""
    model = SmallCNN()
    try:
        # Check if forward is implemented
        _ = model(torch.zeros(1, 1, 28, 28))
    except NotImplementedError:
        pytest.skip("Exercise 1 or 3 not implemented")

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    sids = [f"train:{i:06d}" for i in range(8)]
    ds = ToyDataset(x, y, sids)
    loader = DataLoader(ds, batch_size=4)

    # Save initial weights
    weights_before = [p.clone() for p in model.parameters()]
    criterion = nn.CrossEntropyLoss()

    try:
        loss, acc, preds = evaluate(model, loader, criterion, torch.device("cpu"))
    except NotImplementedError:
        pytest.skip("Exercise 3 not implemented: evaluate")

    # Verify model is in eval mode
    assert not model.training, "evaluate() must set model to eval mode"

    # Verify weights are strictly unmodified
    for p_before, p_after in zip(weights_before, model.parameters()):
        assert torch.equal(p_before, p_after), "evaluate() must not mutate model weights"

    assert len(preds) == 8, f"Expected 8 prediction records, got {len(preds)}"


def test_evaluation_weighted_partial_batches() -> None:
    """Verify that sample-weighted mean loss and accuracy properly account for partial batches (Exercise 3)."""
    model = SmallCNN()
    try:
        _ = model(torch.zeros(1, 1, 28, 28))
    except NotImplementedError:
        pytest.skip("Exercise 1 or 3 not implemented")

    # 7 samples evaluated in 1 batch vs batch_size=5 (batches of 5 and 2)
    torch.manual_seed(42)
    x = torch.randn(7, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    sids = [f"test:{i:06d}" for i in range(7)]
    ds = ToyDataset(x, y, sids)

    loader_single = DataLoader(ds, batch_size=7, shuffle=False)
    loader_split = DataLoader(ds, batch_size=5, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    try:
        loss_single, acc_single, _ = evaluate(model, loader_single, criterion, torch.device("cpu"))
        loss_split, acc_split, _ = evaluate(model, loader_split, criterion, torch.device("cpu"))
    except NotImplementedError:
        pytest.skip("Exercise 3 not implemented: evaluate")

    assert math.isclose(loss_single, loss_split, rel_tol=1e-5), (
        f"Sample-weighted loss must match between single-batch ({loss_single:.6f}) "
        f"and multi-batch ({loss_split:.6f})"
    )
    assert math.isclose(acc_single, acc_split, rel_tol=1e-5), (
        f"Accuracy must match between single-batch ({acc_single:.6f}) and multi-batch ({acc_split:.6f})"
    )


def test_train_one_epoch_updates_weights_and_reports_metrics() -> None:
    """Verify that train_one_epoch performs parameter updates and returns valid metrics (Exercise 2)."""
    model = SmallCNN()
    try:
        _ = model(torch.zeros(1, 1, 28, 28))
    except NotImplementedError:
        pytest.skip("Exercise 1 or 2 not implemented")

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    sids = [f"train:{i:06d}" for i in range(8)]
    ds = ToyDataset(x, y, sids)
    loader = DataLoader(ds, batch_size=4)

    weights_before = [p.clone() for p in model.parameters()]
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    try:
        mean_loss, acc, count = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    except NotImplementedError:
        pytest.skip("Exercise 2 not implemented: train_one_epoch")

    assert count == 8, f"Expected count 8, got {count}"
    assert mean_loss > 0.0 and math.isfinite(mean_loss), f"Invalid mean_loss: {mean_loss}"
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range [0, 1]: {acc}"

    # Weights must be updated
    weights_changed = any(not torch.equal(p_b, p_a) for p_b, p_a in zip(weights_before, model.parameters()))
    assert weights_changed, "train_one_epoch must update model parameters"
