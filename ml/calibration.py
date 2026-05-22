"""
Probability calibration and abstention helpers for the wasting classifier.

Two pieces:

1. Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural
   Networks") — fit a single scalar T on a held-out set so that the softmax
   becomes well-calibrated under negative log-likelihood. Applied
   post-hoc; no retraining required.

2. Mondrian (class-conditional) conformal prediction
   (Vovk 2005; Angelopoulos & Bates 2021) — for a target coverage 1-α,
   produce a *set* of plausible classes per prediction with a calibrated
   coverage guarantee that holds *within each class*. When the set
   has size > 1, the inference layer can route to "manual measurement
   needed" instead of a silent guess.

Both are saved to disk as JSON so inference does not need TensorFlow loaded
just to apply calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    n_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """
    Fit a single scalar T > 0 on `logits`/`labels` to minimize NLL.

    Uses simple gradient descent in log-T space (so T stays positive).
    `logits` should be the pre-softmax outputs of the classifier on a
    held-out calibration split.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    log_t = 0.0  # T = 1.0 initially

    for _ in range(n_iter):
        T = np.exp(log_t)
        scaled = logits / T
        # Numerically-stable softmax + NLL gradient w.r.t. log T
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        Z = exp.sum(axis=1, keepdims=True)
        probs = exp / Z

        # d NLL / d log_T
        # NLL = -mean( logits[y]/T - logsumexp(logits/T) )
        # d/d log_T of (-logits[y]/T + logsumexp) =
        #   logits[y]/T - sum_k probs[k]*logits[k]/T
        N = len(labels)
        idx = np.arange(N)
        true_logits = logits[idx, labels]
        weighted_logits = (probs * logits).sum(axis=1)
        grad = ((true_logits - weighted_logits) / T).mean()
        log_t -= lr * grad

    return float(np.exp(log_t))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return softmax(logits / T)."""
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Mondrian conformal prediction
# ---------------------------------------------------------------------------

@dataclass
class ConformalCalibrator:
    """
    Class-conditional conformal calibrator.

    Stores a per-class threshold q[c] such that, for a true class c, with
    probability ≥ 1-α the *non-conformity score* of the calibration sample
    is ≤ q[c]. At test time we include every class whose non-conformity
    score is ≤ q[c'] in the prediction set.

    Non-conformity score used: 1 - p(true class).
    """
    alpha: float
    classes: List[str]
    thresholds: Dict[str, float]   # class label → quantile threshold

    @classmethod
    def fit(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        class_names: Sequence[str],
        alpha: float = 0.10,
    ) -> "ConformalCalibrator":
        """
        Fit class-conditional 1-α quantile of (1 - p_true) on calibration set.
        """
        thresholds: Dict[str, float] = {}
        for c, name in enumerate(class_names):
            mask = labels == c
            if not mask.any():
                # Fallback: very loose threshold so the class is always included
                thresholds[name] = 1.0
                continue
            scores = 1.0 - probs[mask, c]
            n = mask.sum()
            # Conformal correction: ceil((n+1)*(1-alpha)) / n quantile
            q = np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0)
            thresholds[name] = float(np.quantile(scores, q))
        return cls(alpha=alpha, classes=list(class_names), thresholds=thresholds)

    def predict_set(self, probs: np.ndarray) -> List[List[str]]:
        """
        For each row, return the set of class names whose non-conformity
        score does NOT exceed their threshold. The set may be empty in
        pathological cases — callers should treat empty as abstain.
        """
        sets: List[List[str]] = []
        for row in probs:
            members: List[str] = []
            for c, name in enumerate(self.classes):
                score = 1.0 - row[c]
                if score <= self.thresholds[name]:
                    members.append(name)
            sets.append(members)
        return sets

    def to_json(self) -> str:
        return json.dumps({
            "alpha":      self.alpha,
            "classes":    self.classes,
            "thresholds": self.thresholds,
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ConformalCalibrator":
        d = json.loads(s)
        return cls(alpha=d["alpha"], classes=d["classes"], thresholds=d["thresholds"])


def save_calibration(
    out_path: Path,
    temperature: float,
    conformal: ConformalCalibrator,
    classifier_kind: str = "mlp",
) -> None:
    payload = {
        "classifier_kind": classifier_kind,
        "temperature":     temperature,
        "conformal":       {
            "alpha":      conformal.alpha,
            "classes":    conformal.classes,
            "thresholds": conformal.thresholds,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())
