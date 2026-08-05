"""Deterministic child-level dataset splits shared by every ML entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SPLIT_SCHEMA_VERSION = 1
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "calibration": 0.15, "test": 0.15}


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Fingerprint identity and labels without hashing sensitive image data."""
    required = df[["child_id", "label"]].astype(str).sort_values("child_id")
    payload = required.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_split_manifest(df: pd.DataFrame, *, seed: int = SPLIT_SEED) -> dict:
    """Create disjoint splits at child granularity, stratified by child label."""
    from sklearn.model_selection import train_test_split

    if "child_id" not in df or "label" not in df:
        raise ValueError("dataset must contain child_id and label columns")
    child_labels = df.groupby("child_id", sort=True)["label"].agg(
        lambda values: values.mode().iloc[0]
    )
    child_ids = child_labels.index.to_numpy()
    labels = child_labels.to_numpy()
    train_ids, remaining_ids, _, remaining_labels = train_test_split(
        child_ids,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    calibration_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_labels,
    )
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "strategy": "stratified_child_70_15_15",
        "dataset_fingerprint": dataset_fingerprint(df),
        "splits": {
            "train": sorted(map(str, train_ids)),
            "calibration": sorted(map(str, calibration_ids)),
            "test": sorted(map(str, test_ids)),
        },
    }


def write_split_manifest(
    df: pd.DataFrame, path: Path, *, seed: int = SPLIT_SEED
) -> dict:
    manifest = create_split_manifest(df, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_split_manifest(df: pd.DataFrame, path: Path) -> dict:
    """Load and strictly validate a previously generated split manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(df):
        raise ValueError("split manifest does not match this dataset")
    split_sets = {name: set(values) for name, values in manifest["splits"].items()}
    if set(split_sets) != set(SPLIT_FRACTIONS):
        raise ValueError("split manifest must define train, calibration, and test")
    if any(
        split_sets[a] & split_sets[b] for a in split_sets for b in split_sets if a < b
    ):
        raise ValueError("child IDs overlap between splits")
    dataset_children = set(df["child_id"].astype(str))
    if set().union(*split_sets.values()) != dataset_children:
        raise ValueError(
            "split manifest does not cover every dataset child exactly once"
        )
    return manifest


def rows_for_split(df: pd.DataFrame, manifest: dict, split: str) -> pd.DataFrame:
    if split not in SPLIT_FRACTIONS:
        raise ValueError(f"unknown split: {split}")
    child_ids = set(manifest["splits"][split])
    return df[df["child_id"].astype(str).isin(child_ids)].copy()
