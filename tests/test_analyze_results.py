"""Tests for scripts/analyze_results.py — analysis and coverage accounting."""
from scripts.analyze_results import analyze, coverage, render_report


def _row(**over) -> dict:
    row = {
        "child_name": "001", "age_months": "30.0", "sex": "M",
        "actual_height_cm": "85.0", "pred_height_cm": "86.0",
        "actual_weight_kg": "10.0", "pred_weight_ml_kg": "9.5",
        "actual_combined_status": "Normal", "pred_status_final": "Normal",
        "actual_whz_status": "Normal", "actual_muac_status": "Normal",
        "actual_oedema": "",
        "error": "",
    }
    row.update(over)
    return row


def test_analyze_height_and_weight_pairs():
    a = analyze([_row(), _row(child_name="002", actual_height_cm="90.0",
                            pred_height_cm="89.0")])
    assert a["height"]["n"] == 2
    assert a["weight"]["n"] == 2


def test_analyze_skips_rows_missing_values():
    a = analyze([_row(), _row(child_name="002", actual_height_cm="",
                              pred_height_cm="")])
    assert a["height"]["n"] == 1


def test_analyze_sam_confusion():
    rows = [
        _row(actual_combined_status="SAM", pred_status_final="SAM"),
        _row(child_name="002", actual_combined_status="SAM",
             pred_status_final="Normal"),
        _row(child_name="003"),
    ]
    a = analyze(rows)
    assert a["sam"]["tp"] == 1 and a["sam"]["fn"] == 1 and a["sam"]["tn"] == 1


def test_analyze_subgroups_partition_rows():
    rows = [
        _row(sex="M", age_months="12.0"),
        _row(child_name="002", sex="F", age_months="30.0"),
    ]
    a = analyze(rows)
    assert a["subgroups"]["sex=M"]["status_n"] == 1
    assert a["subgroups"]["sex=F"]["status_n"] == 1
    assert a["subgroups"]["age 6-23m"]["status_n"] == 1
    assert a["subgroups"]["age 24-59m"]["status_n"] == 1


def test_coverage_buckets_sum_to_total():
    intake = [{"child_id": c} for c in ("001", "002", "003", "004")]
    qc = [
        {"child_id": "001", "verdict": "ok"},
        {"child_id": "002", "verdict": "ok"},
        {"child_id": "003", "verdict": "failed"},
    ]
    results = [_row(child_name="001"), _row(child_name="002", error="boom")]
    cov = coverage(intake, qc, results)
    assert cov["total"] == 4
    assert cov["assessed"] == 1        # 001
    assert cov["qc_failed"] == 1       # 003
    assert cov["missing_data"] == 2    # 002 errored, 004 never cleaned
    assert cov["assessed"] + cov["qc_failed"] + cov["missing_data"] == cov["total"]
    assert cov["discrepancy"] == ""


def test_render_report_contains_headline_sections():
    rows = [_row(actual_combined_status="SAM", pred_status_final="SAM")]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], rows),
    )
    assert "## Coverage" in text
    assert "## Height agreement" in text
    assert "## Weight agreement" in text
    assert "## Status agreement" in text
    assert "SAM sensitivity" in text


# --- Extension: report SAM sensitivity against two gold standards ---------
#
# `actual_combined_status` is the full WHO OR-rule (oedema OR MUAC<11.5 OR
# WHZ<-3). `pred_status_final` is derived only from the app's photo-based
# WHZ estimate — it structurally cannot see oedema- or MUAC-only SAM cases.
# Reporting a single sensitivity number against the full OR-rule, with no
# explanation, would misread a structural blind spot as an ML failure.

def _sam_via_muac_only():
    """Gold-standard SAM reached via MUAC alone; WHZ arm says Normal, and
    the app (which only ever sees the WHZ arm) correctly says Normal too."""
    return _row(
        child_name="001",
        actual_combined_status="SAM", pred_status_final="Normal",
        actual_whz_status="Normal", actual_muac_status="SAM",
    )


def _sam_via_whz():
    """Gold-standard SAM reached via WHZ; the app correctly flags it."""
    return _row(
        child_name="002",
        actual_combined_status="SAM", pred_status_final="SAM",
        actual_whz_status="SAM", actual_muac_status="Normal",
    )


def test_sam_sensitivity_two_framings_diverge():
    rows = [_sam_via_muac_only(), _sam_via_whz()]
    a = analyze(rows)

    # Framing A: against the full gold standard, the MUAC-only case is an
    # unavoidable false negative -> sensitivity 1/2.
    assert a["sam"]["tp"] == 1 and a["sam"]["fn"] == 1
    v, _, _ = a["sam"]["sensitivity"]
    assert v == 0.5

    # Framing B: against the WHZ arm alone, the app is correct on both rows
    # it could ever have seen -> sensitivity 1/1, tn 1/1.
    assert a["sam_whz"]["tp"] == 1 and a["sam_whz"]["fn"] == 0
    assert a["sam_whz"]["tn"] == 1
    v_whz, _, _ = a["sam_whz"]["sensitivity"]
    assert v_whz == 1.0


def test_sam_detectability_counts():
    rows = [_sam_via_muac_only(), _sam_via_whz()]
    a = analyze(rows)
    det = a["sam_detectability"]
    assert det["total_actual_sam"] == 2
    assert det["whz_detectable"] == 1
    assert det["muac_or_oedema_only"] == 1


def test_render_report_explains_framing_gap():
    rows = [_sam_via_muac_only(), _sam_via_whz()]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}, {"child_id": "002"}],
                 [{"child_id": "001", "verdict": "ok"},
                  {"child_id": "002", "verdict": "ok"}], rows),
    )
    assert "Framing A" in text
    assert "Framing B" in text
    assert "WHZ arm alone" in text
    # The explanatory note must carry the actual counts, not just prose.
    assert (
        "1 were detectable in principle from the "
        "WHZ arm and 1 were reachable only via MUAC or oedema"
    ) in text


def test_render_report_none_loa_renders_without_crashing():
    rows = [_row()]  # single pair -> bland_altman loa_low/loa_high are None
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], rows),
    )
    assert "n/a (need at least 2 pairs to estimate)" in text


def test_render_report_none_kappa_renders_without_crashing():
    rows = [_row()]  # single actual category -> weighted_kappa is None
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], rows),
    )
    assert "n/a (needs ≥2 distinct actual categories)" in text
