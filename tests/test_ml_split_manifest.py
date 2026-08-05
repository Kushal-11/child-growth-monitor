import pandas as pd
import pytest

from ml.splits import (
    create_split_manifest,
    load_split_manifest,
    rows_for_split,
    write_split_manifest,
)


def _dataset():
    labels = ["SAM", "MAM", "Normal", "Risk_Overweight", "Overweight"]
    return pd.DataFrame(
        [
            {
                "child_id": f"child-{i:03d}",
                "label": labels[i % len(labels)],
                "frame": frame,
            }
            for i in range(100)
            for frame in range(2)
        ]
    )


def test_split_is_deterministic_disjoint_and_child_level():
    df = _dataset()
    first = create_split_manifest(df)
    second = create_split_manifest(df.sample(frac=1, random_state=9))
    assert first == second
    child_sets = [set(ids) for ids in first["splits"].values()]
    assert not any(a & b for i, a in enumerate(child_sets) for b in child_sets[i + 1 :])
    for name, ids in first["splits"].items():
        rows = rows_for_split(df, first, name)
        assert set(rows["child_id"]) == set(ids)
        assert all(rows.groupby("child_id")["frame"].count() == 2)


def test_manifest_rejects_dataset_drift(tmp_path):
    df = _dataset()
    path = tmp_path / "splits.json"
    write_split_manifest(df, path)
    changed = df.copy()
    changed.loc[0, "label"] = "Normal"
    with pytest.raises(ValueError, match="does not match"):
        load_split_manifest(changed, path)
