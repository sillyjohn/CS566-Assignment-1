"""Utility functions for hashing, metrics, and artifact export."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def compute_file_sha256(path: str | Path) -> str:
    """Compute raw bytes SHA256 of a file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_source_sha256(
    base_dir: str | Path,
    entry_scripts: Sequence[str] = ("run_experiment.py",),
) -> dict[str, str]:
    """Compute SHA256 for all Python source files and executed entry scripts.

    Hash scheme:
    Maps relative POSIX path from the package root (e.g. 'run_experiment.py', 'src/models.py')
    to the SHA-256 hex digest of the raw file bytes.
    """
    base_dir = Path(base_dir)
    results: dict[str, str] = {}

    src_dir = base_dir / "src" if (base_dir / "src").is_dir() else base_dir
    package_root = src_dir.parent if src_dir.name == "src" else base_dir

    if src_dir.is_dir():
        for p in sorted(src_dir.rglob("*.py")):
            if p.is_file() and not p.name.startswith("."):
                rel_path = p.relative_to(package_root).as_posix()
                results[rel_path] = compute_file_sha256(p)

    for script_name in sorted(entry_scripts):
        script_path = package_root / script_name
        if script_path.is_file():
            rel_path = script_path.relative_to(package_root).as_posix()
            results[rel_path] = compute_file_sha256(script_path)

    return results


def compute_actual_id_digest(sample_ids: Sequence[str]) -> str:
    """Compute SHA256 of lexicographically sorted actual sample_id strings joined with LF.

    Includes a trailing LF.
    If sample_ids is empty, returns the SHA256 digest of empty bytes.
    """
    if not sample_ids:
        return hashlib.sha256(b"").hexdigest()
    sorted_ids = sorted(sample_ids)
    encoded = "\n".join(sorted_ids).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def export_predictions(
    output_path: str | Path,
    run_id: str,
    course_split: str,
    predictions: Sequence[Mapping[str, Any] | Sequence[Any]],
) -> None:
    """Export prediction records to CSV.

    Columns: run_id,course_split,sample_id,true_label,predicted_label
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "course_split", "sample_id", "true_label", "predicted_label"])
        for p in predictions:
            if isinstance(p, Mapping):
                writer.writerow([
                    run_id,
                    course_split,
                    p["sample_id"],
                    int(p["true_label"]),
                    int(p["predicted_label"]),
                ])
            else:
                # tuple or list: (sample_id, true_label, predicted_label)
                writer.writerow([
                    run_id,
                    course_split,
                    p[0],
                    int(p[1]),
                    int(p[2]),
                ])


def export_history(
    output_path: str | Path,
    history_records: Sequence[Mapping[str, Any]],
) -> None:
    """Export per-epoch training and validation history to CSV.

    Columns: epoch,train_loss,train_accuracy,validation_loss,validation_accuracy,train_count,validation_count
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "train_count",
            "validation_count",
        ])
        for row in history_records:
            writer.writerow([
                int(row["epoch"]),
                f"{float(row['train_loss']):.6f}" if row.get("train_loss") is not None else "",
                f"{float(row['train_accuracy']):.6f}" if row.get("train_accuracy") is not None else "",
                f"{float(row['validation_loss']):.6f}" if row.get("validation_loss") is not None else "",
                f"{float(row['validation_accuracy']):.6f}" if row.get("validation_accuracy") is not None else "",
                int(row["train_count"]) if row.get("train_count") is not None else "",
                int(row["validation_count"]) if row.get("validation_count") is not None else "",
            ])


def update_summary(
    summary_path: str | Path,
    row: Mapping[str, Any],
) -> None:
    """Update or append a row in results/summary.csv.

    Columns: run_id,condition,seed,epochs,evaluation_split,n_examples,loss,accuracy,status
    """
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "condition",
        "seed",
        "epochs",
        "evaluation_split",
        "n_examples",
        "loss",
        "accuracy",
        "status",
    ]

    existing_rows: list[dict[str, Any]] = []
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)

    # Format new row
    new_entry = {
        "run_id": str(row.get("run_id", "")),
        "condition": str(row.get("condition", "")),
        "seed": str(row.get("seed", "")) if row.get("seed") is not None else "",
        "epochs": str(row.get("epochs", "")) if row.get("epochs") is not None else "",
        "evaluation_split": str(row.get("evaluation_split", "")),
        "n_examples": str(row.get("n_examples", "")) if row.get("n_examples") is not None else "",
        "loss": f"{float(row['loss']):.6f}" if row.get("loss") is not None and row["loss"] != "" else "",
        "accuracy": f"{float(row['accuracy']):.6f}" if row.get("accuracy") is not None and row["accuracy"] != "" else "",
        "status": str(row.get("status", "")),
    }

    # Replace existing run_id if present, else append
    replaced = False
    for i, r in enumerate(existing_rows):
        if r["run_id"] == new_entry["run_id"]:
            existing_rows[i] = new_entry
            replaced = True
            break
    if not replaced:
        existing_rows.append(new_entry)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)


def save_config(output_path: str | Path, config_data: Mapping[str, Any]) -> None:
    """Save configuration dictionary as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
