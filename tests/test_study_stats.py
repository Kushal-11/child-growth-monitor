"""Hand-computed examples — a formula bug must not misreport the study."""
import pytest

from scripts.study_stats import (
    bland_altman,
    binary_metrics,
    confusion_binary,
    weighted_kappa,
    wilson_ci,
)


def test_bland_altman_hand_computed():
    # diffs (pred - actual) = [+1, -1]: bias 0, sd(ddof=1) = sqrt(2)
    r = bland_altman(actual=[80.0, 90.0], predicted=[81.0, 89.0])
    assert r["n"] == 2
    assert r["bias"] == pytest.approx(0.0)
    assert r["loa_high"] == pytest.approx(1.96 * 2 ** 0.5, abs=1e-6)
    assert r["loa_low"] == pytest.approx(-1.96 * 2 ** 0.5, abs=1e-6)
    assert r["mae"] == pytest.approx(1.0)


def test_wilson_ci_hand_computed():
    # k=8, n=10, z=1.96 -> (0.490, 0.943) (standard worked example)
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)


def test_confusion_binary():
    actual = ["SAM", "SAM", "MAM", "Normal", "Normal"]
    pred = ["SAM", "Normal", "SAM", "Normal", "SAM"]
    tp, fp, tn, fn = confusion_binary(actual, pred, positive={"SAM"})
    assert (tp, fp, tn, fn) == (1, 2, 1, 1)


def test_binary_metrics_hand_computed():
    m = binary_metrics(tp=8, fp=2, tn=88, fn=2)
    assert m["sensitivity"][0] == pytest.approx(0.8)
    assert m["specificity"][0] == pytest.approx(88 / 90)
    assert m["ppv"][0] == pytest.approx(0.8)
    assert m["npv"][0] == pytest.approx(88 / 90)


def test_binary_metrics_zero_denominator_is_none():
    m = binary_metrics(tp=0, fp=0, tn=10, fn=0)
    assert m["sensitivity"] is None
    assert m["ppv"] is None


CATS = ["SAM", "MAM", "Normal"]


def test_weighted_kappa_perfect_agreement():
    y = ["SAM", "MAM", "Normal", "SAM"]
    assert weighted_kappa(y, y, CATS) == pytest.approx(1.0)


def test_weighted_kappa_orders_error_severity():
    # SAM misread as Normal must cost more than SAM misread as MAM
    actual = ["SAM", "MAM", "Normal", "SAM", "MAM", "Normal"]
    near = ["MAM", "MAM", "Normal", "SAM", "MAM", "Normal"]   # SAM->MAM
    far = ["Normal", "MAM", "Normal", "SAM", "MAM", "Normal"]  # SAM->Normal
    assert weighted_kappa(actual, far, CATS) < weighted_kappa(actual, near, CATS)


def test_weighted_kappa_single_category_is_none_not_nan():
    # Entirely realistic for a subgroup (e.g. an age band where all
    # children are Normal): kappa's formula divides 0/0. Must report
    # None, never a silent nan that could pass unnoticed in a report.
    y = ["Normal", "Normal", "Normal", "Normal"]
    assert weighted_kappa(y, y, CATS) is None


def test_confusion_binary_mismatched_lengths_raises():
    actual = ["SAM", "MAM", "Normal"]
    pred = ["SAM", "MAM"]
    with pytest.raises(ValueError, match=r"3.*2|2.*3"):
        confusion_binary(actual, pred, positive={"SAM"})


def test_bland_altman_single_pair_loa_is_none():
    # With n=1 the sample SD is undefined; a zero-width interval would
    # misleadingly read as perfect agreement. bias/mae ARE computable.
    r = bland_altman(actual=[80.0], predicted=[82.0])
    assert r["n"] == 1
    assert r["bias"] == pytest.approx(2.0)
    assert r["mae"] == pytest.approx(2.0)
    assert r["loa_low"] is None
    assert r["loa_high"] is None
