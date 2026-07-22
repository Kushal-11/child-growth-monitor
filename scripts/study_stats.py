"""
Agreement statistics for the app-vs-manual comparison study.

Bland & Altman (Lancet 1986) for continuous agreement; Wilson score CIs
for proportions; linearly weighted Cohen's kappa for the ordered
SAM > MAM > Normal scale.
"""
import math
from typing import Optional

import numpy as np
from sklearn.metrics import cohen_kappa_score


def bland_altman(actual: list[float], predicted: list[float]) -> dict:
    """Mean bias and 95% limits of agreement (bias ± 1.96·SD, ddof=1)."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("need equal, non-empty actual/predicted lists")
    diffs = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return {
        "n": len(diffs),
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mae": float(np.mean(np.abs(diffs))),
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        raise ValueError("n must be > 0")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def confusion_binary(
    actual: list[str], pred: list[str], positive: set[str],
) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) treating membership of `positive` as the positive class."""
    tp = fp = tn = fn = 0
    for a, p in zip(actual, pred):
        a_pos, p_pos = a in positive, p in positive
        if a_pos and p_pos:
            tp += 1
        elif not a_pos and p_pos:
            fp += 1
        elif not a_pos and not p_pos:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _rate(k: int, n: int) -> Optional[tuple[float, float, float]]:
    if n == 0:
        return None
    lo, hi = wilson_ci(k, n)
    return (k / n, lo, hi)


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Sensitivity/specificity/PPV/NPV, each (value, ci_low, ci_high) or None."""
    return {
        "sensitivity": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "ppv": _rate(tp, tp + fp),
        "npv": _rate(tn, tn + fn),
    }


def weighted_kappa(
    actual: list[str], pred: list[str], categories: list[str],
) -> float:
    """Linearly weighted Cohen's kappa over an ordered category scale."""
    return float(
        cohen_kappa_score(actual, pred, labels=categories, weights="linear")
    )
