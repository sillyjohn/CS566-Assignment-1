"""Training and evaluation routines for CSCI 566 image classification assignment."""

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, int]:
    """Train the model for one epoch over the given dataloader.

    The model should be set to training mode.
    Loss must be sample-weighted mean cross-entropy across all examples,
    properly accounting for any partial last batch.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding batches.
        optimizer: PyTorch optimizer (e.g. Adam).
        criterion: PyTorch loss criterion (e.g. CrossEntropyLoss).
        device: Device to train on.

    Returns:
        tuple of (mean_loss, accuracy, total_examples):
            - mean_loss: float, sample-weighted mean loss.
            - accuracy: float, accuracy in [0.0, 1.0].
            - total_examples: int, total count of examples processed.
    """

    # Exercise 2: Implement one training epoch
    # 1. Put model in train mode
    # 2. Iterate through dataloader batches
    # 3. Compute forward pass, loss, backward pass, optimizer step
    # 4. Track total loss and total correct predictions weighted by batch size
    # 5. Return (sample_weighted_mean_loss, accuracy, total_examples)
    # Set to training mode
    model.train()
    total_loss = 0
    total_sample = 0
    total_correct = 0
    accuracy = 0.0

    for img,label, sample_id in dataloader:

        img = img.to(device)
        label = label.to(device)

        #reset gradient for each epoch
        optimizer.zero_grad()
        output = model.forward(img)

        loss = criterion(output, label)

        batch_size = img.size(0)
        total_loss += loss.item()*batch_size
        total_sample+=batch_size

        with torch.no_grad():
            predictions = output.argmax(dim=1)
            total_correct += (predictions == label).sum().item()

        loss.backward()
        optimizer.step()
    mean_loss = total_loss / total_sample
    accuracy = total_correct / total_sample

    return mean_loss, accuracy, total_sample

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Evaluate the model over the given dataloader without updating parameters.

    The model should be set to evaluation mode and gradients disabled.
    Loss must be sample-weighted mean cross-entropy across all examples,
    properly accounting for any partial last batch.
    Collect prediction records for each example evaluated.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding batches.
        criterion: PyTorch loss criterion (e.g. CrossEntropyLoss).
        device: Device to evaluate on.

    Returns:
        tuple of (mean_loss, accuracy, predictions):
            - mean_loss: float, sample-weighted mean loss.
            - accuracy: float, accuracy in [0.0, 1.0].
            - predictions: list of dicts, each containing:
                'sample_id': str (actual sample_id from the batch)
                'true_label': int
                'predicted_label': int
    """
    # Exercise 3: Implement evaluation
    # 1. Put model in eval mode
    # 2. Disable gradient calculation (torch.no_grad())
    # 3. Iterate through dataloader batches
    # 4. Compute forward pass, loss, and predictions
    # 5. Track total loss, correct predictions, and prediction records
    # 6. Return (sample_weighted_mean_loss, accuracy, predictions)
    model.eval()
    total_loss = 0
    total_sample = 0
    total_correct = 0
    prediction_records = []
    with torch.no_grad():
        for img, label, sample_id in dataloader:
            img = img.to(device)
            label = label.to(device)
            output = model.forward(img)
            loss = criterion(output, label)
            batch_size = img.size(0)
            predictions = output.argmax(dim=1)
            total_loss += loss.item()*batch_size
            total_sample += batch_size
            total_correct += (predictions == label).sum().item()
            for sid, true, pred in zip(sample_id, label.tolist(), predictions.tolist()):
                prediction_records.append({
                    "sample_id": sid,
                    "true_label": int(true),
                    "predicted_label": int(pred),
                })
        mean_loss = total_loss / total_sample
        accuracy = total_correct / total_sample

    return mean_loss, accuracy, prediction_records

