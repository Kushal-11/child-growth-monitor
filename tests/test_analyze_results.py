"""Tests for scripts/analyze_results.py — analysis and coverage accounting."""
import csv
import sys
from pathlib import Path

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


# --- Finding 1: unverdicted gold-standard SAM rows must not silently -------
# vanish from the confusion matrix and inflate sensitivity.
#
# batch_assess.py catches an ML/pose failure, prints a [WARN], and leaves
# `error` blank while `pred_status_final` ends up blank too. Reproduces the
# verified failure: 3 gold-standard SAM children, 2 unverdicted.

def test_unverdicted_gold_sam_rows_excluded_from_matrix_and_counted():
    rows = [
        _row(child_name="001", actual_combined_status="SAM",
             pred_status_final="SAM"),   # correctly verdicted
        _row(child_name="002", actual_combined_status="SAM",
             pred_status_final=""),      # ML failed silently -> no verdict
        _row(child_name="003", actual_combined_status="SAM",
             pred_status_final=""),      # ML failed silently -> no verdict
    ]
    a = analyze(rows)

    # Only the one verdicted SAM row can participate in the confusion matrix.
    assert a["sam"]["tp"] == 1 and a["sam"]["fn"] == 0
    v, _, _ = a["sam"]["sensitivity"]
    assert v == 1.0

    # The other two must be counted as excluded, not silently dropped.
    assert a["excluded_unverdicted"] == 2
    assert a["excluded_unverdicted_sam"] == 2

    # sam_detectability must reconcile with the matrix: it is scoped to the
    # same paired rows, so its total equals tp+fn of the SAM confusion, not
    # all 3 gold-standard SAM children in the batch.
    assert a["sam_detectability"]["total_actual_sam"] == 1


def test_report_never_shows_bare_sam_sensitivity_without_exclusion_count():
    rows = [
        _row(child_name="001", actual_combined_status="SAM",
             pred_status_final="SAM"),
        _row(child_name="002", actual_combined_status="SAM",
             pred_status_final=""),
        _row(child_name="003", actual_combined_status="SAM",
             pred_status_final=""),
    ]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": c} for c in ("001", "002", "003")], [], rows),
    )
    assert "SAM sensitivity: 1.000" in text

    start = text.index("Framing A")
    end = text.index("Framing B")
    section = text[start:end]
    # The bare "1.000" must not appear without the exclusion count sitting
    # right beside it in the same Framing A block.
    assert "SAM sensitivity: 1.000" in section
    assert "excluded" in section.lower()
    assert "2" in section


# --- Finding 2 & 3: coverage() must flag results rows absent from intake ---

def test_coverage_flags_results_ids_absent_from_intake():
    intake = [{"child_id": "001"}]
    results = [_row(child_name="001"), _row(child_name="999")]
    cov = coverage(intake, [], results)

    # Denominator stays keyed off the intake manifest...
    assert cov["assessed"] == 1
    # ...but the mismatch with the statistics' denominator (status_n == 2)
    # must be visible, not silently absorbed.
    assert cov["discrepancy"] != ""
    assert "999" in cov["discrepancy"]
    assert "1" in cov["discrepancy"]
    assert cov["unknown_ids"] == ["999"]

    a = analyze(results)
    # Both rows have a full Normal/Normal gold-standard/verdict pair, so
    # the statistics' denominator (2) disagrees with coverage's (1) — that
    # disagreement is exactly what must be surfaced, not silently absorbed.
    assert a["status_n"] == 2
    assert cov["assessed"] != a["status_n"]


def test_render_report_surfaces_unknown_result_ids():
    intake = [{"child_id": "001"}]
    results = [_row(child_name="001"), _row(child_name="999")]
    text = render_report(analyze(results), coverage(intake, [], results))
    assert "999" in text


def test_coverage_discrepancy_is_not_dead_code_for_normal_input():
    """20,000 randomized well-formed inputs never triggered the old
    'BUCKET SUM MISMATCH' check because it was structurally unreachable.
    The real integrity failure (results not in intake) must be reachable."""
    intake = [{"child_id": "001"}, {"child_id": "002"}]
    results = [_row(child_name="001"), _row(child_name="002")]
    cov = coverage(intake, [], results)
    assert cov["discrepancy"] == ""  # well-formed input stays clean


# --- Finding 4: child_id must be stripped consistently in coverage() -------

def test_coverage_strips_whitespace_from_intake_child_id():
    intake = [{"child_id": " 001"}]  # leading space, e.g. dirty CSV export
    results = [_row(child_name="001")]
    cov = coverage(intake, [], results)
    assert cov["assessed"] == 1
    assert cov["missing_data"] == 0


# --- Finding 5: report write must not crash on a non-UTF-8 locale ----------

def test_main_writes_report_with_utf8_encoding(tmp_path, monkeypatch):
    results_csv = tmp_path / "results.csv"
    intake_csv = tmp_path / "intake.csv"
    out_path = tmp_path / "report.md"

    row = _row()
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    with open(intake_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["child_id"])
        w.writeheader()
        w.writerow({"child_id": "001"})

    captured: dict = {}
    orig_write_text = Path.write_text

    def spy_write_text(self, data, *args, **kwargs):
        if self == out_path:
            captured["encoding"] = kwargs.get("encoding")
        return orig_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(sys, "argv", [
        "analyze_results.py",
        "--results", str(results_csv),
        "--intake", str(intake_csv),
        "--qc-report", str(tmp_path / "no_such_qc.csv"),
        "--out", str(out_path),
    ])

    from scripts.analyze_results import main
    main()

    assert captured["encoding"] == "utf-8"


# --- Finding 6: blank/infant ages must not land in the "6-23m" bucket ------

def test_age_subgroups_exclude_blank_and_under_6_month_rows():
    rows = [
        _row(child_name="001", age_months="12.0"),  # genuine 6-23m
        _row(child_name="002", age_months=""),       # blank age
        _row(child_name="003", age_months="3.0"),    # genuine 3-month-old
        _row(child_name="004", age_months="30.0"),   # 24-59m
    ]
    a = analyze(rows)
    assert a["subgroups"]["age 6-23m"]["status_n"] == 1
    assert a["subgroups"]["age 24-59m"]["status_n"] == 1
    assert a["age_excluded_from_subgroups"] == 2
    # Whole-study statistics still include all 4 rows.
    assert a["status_n"] == 4


def test_render_report_shows_age_exclusion_count():
    rows = [
        _row(child_name="001", age_months="3.0"),
        _row(child_name="002", age_months="12.0"),
    ]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}, {"child_id": "002"}], [], rows),
    )
    assert "## Subgroups" in text
    assert "excluded from age stratification" in text.lower()


# --- Review round: Finding 1 — subgroup sensitivities are undenominated ---
#
# `_analyze_block` already computes `excluded_unverdicted` (and the
# gold-standard-SAM breakout) for every subgroup, but `render_report` never
# emitted it in `## Subgroups` — each subgroup printed a bare sensitivity
# figure with only `Height n=<N>` above it, inviting the reader to take the
# height n as the denominator. Reproduces the verified probe: 3
# gold-standard SAM children, 2 unverdicted, all male.

def test_subgroup_sensitivity_shows_exclusion_counts_not_bare():
    rows = [
        _row(child_name="001", sex="M", actual_combined_status="SAM",
             pred_status_final="SAM"),   # correctly verdicted
        _row(child_name="002", sex="M", actual_combined_status="SAM",
             pred_status_final=""),      # ML failed silently -> no verdict
        _row(child_name="003", sex="M", actual_combined_status="SAM",
             pred_status_final=""),      # ML failed silently -> no verdict
    ]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": c} for c in ("001", "002", "003")], [], rows),
    )

    start = text.index("### sex=M")
    end = text.index("### sex=F")
    section = text[start:end]

    # The subgroup's SAM sensitivity is 1.000 (1/1 on the one verdicted
    # row), but it must never appear without the exclusion counts sitting
    # right beside it in the same subgroup block.
    assert "SAM sensitivity" in section
    assert "1.000" in section
    assert "excluded" in section.lower()
    # 2 unverdicted rows dropped, both gold-standard SAM — the SAM
    # breakout must not be omitted.
    assert "2" in section


# --- Review round: Finding 2 — Assessed overstates capture -----------------
#
# `coverage()` keyed `assessed_ids` on `error` alone, so a child whose ML
# step failed silently (blank `pred_status_final`, blank `error`) was
# reported as fully "Assessed". Coverage is the first section a clinician
# reads; its numbers must reconcile with `status_n`, not silently disagree
# with it.

def test_coverage_distinguishes_assessed_with_and_without_verdict():
    intake = [{"child_id": str(i)} for i in range(1, 19)]  # 18 children
    results = []
    for i in range(1, 16):  # 15 fully verdicted
        results.append(_row(child_name=str(i)))
    for i in range(16, 19):  # 3 processed without error, but no app verdict
        results.append(_row(child_name=str(i), pred_status_final=""))

    cov = coverage(intake, [], results)
    a = analyze(results)

    assert cov["assessed"] == 18
    assert cov["missing_data"] == 0
    assert cov["assessed_with_verdict"] == 15
    assert cov["assessed_without_verdict"] == 3
    assert (cov["assessed_with_verdict"] + cov["assessed_without_verdict"]
            == cov["assessed"])
    # The whole point: Coverage's verdict count must reconcile with the
    # denominator the statistics section actually uses.
    assert cov["assessed_with_verdict"] == a["status_n"]

    # Bucket-sum invariant is unchanged — this is a breakdown WITHIN the
    # assessed bucket, not a new top-level bucket.
    assert cov["assessed"] + cov["qc_failed"] + cov["missing_data"] == cov["total"]


def test_render_report_coverage_reconciles_with_status_n():
    intake = [{"child_id": str(i)} for i in range(1, 19)]
    results = []
    for i in range(1, 16):
        results.append(_row(child_name=str(i)))
    for i in range(16, 19):
        results.append(_row(child_name=str(i), pred_status_final=""))

    text = render_report(analyze(results), coverage(intake, [], results))
    start = text.index("## Coverage")
    end = text.index("## Height agreement")
    section = text[start:end]

    assert "18" in section
    assert "15" in section
    assert "3" in section
    assert "verdict" in section.lower()


# --- Review round: Finding 3 — blank child_name unreadable banner ----------
#
# A results row with a blank `child_name` put `""` into `unknown_ids`,
# firing the banner as `1 result row(s) reference child_id(s) ... (e.g. )`
# and rendering `not in intake manifest: 1 ()`. The signal is genuine but
# unreadable, and conflates "no child id at all" with "child id not in the
# manifest".

def test_coverage_blank_child_name_reported_distinctly():
    intake = [{"child_id": "001"}]
    results = [_row(child_name="001"), _row(child_name="")]
    cov = coverage(intake, [], results)

    assert cov["blank_child_id_rows"] == 1
    assert "" not in cov["unknown_ids"]
    assert cov["unknown_ids"] == []


def test_render_report_blank_child_id_renders_readably():
    intake = [{"child_id": "001"}]
    results = [_row(child_name="001"), _row(child_name="")]
    text = render_report(analyze(results), coverage(intake, [], results))

    assert "<blank>" in text
    assert "(e.g. )" not in text
    assert "not in intake manifest: 1 ()" not in text


# --- Review round: Finding 4 — the age floor fix is one-sided --------------
#
# The prior round added a `>= 6` month floor so "age 6-23m" became
# accurate, but there is no ceiling: a 72-month-old lands in "age 24-59m"
# and is not counted as excluded. WHO growth standards run 0-60 months and
# the assessment stage already rejects ages outside 0-60, so this should be
# rare in practice, but the label must be true regardless.

def test_age_subgroups_exclude_over_59_month_rows():
    rows = [
        _row(child_name="001", age_months="30.0"),  # genuine 24-59m
        _row(child_name="002", age_months="72.0"),  # over the ceiling
    ]
    a = analyze(rows)
    assert a["subgroups"]["age 24-59m"]["status_n"] == 1
    assert a["age_excluded_from_subgroups"] == 1
    # Whole-study statistics still include both rows.
    assert a["status_n"] == 2


def test_errored_gold_standard_sam_child_is_counted_not_discarded():
    """An errored row still carries its ground truth. Dropping it before the
    exclusion accounting let a child with unambiguous gold-standard SAM
    leave the sensitivity denominator entirely.

    Cohort below: 3 gold-standard SAM children — one detected, one assessed
    with no verdict, one errored on a garbled DOB. The headline figure is
    still computed over paired rows only (that is correct), but the report
    must now disclose BOTH excluded SAM children, so a reader can bound the
    true value rather than reading 1.000 and stopping.
    """
    rows = [
        _row(child_name="001", actual_combined_status="SAM",
             pred_status_final="SAM"),                      # detected
        _row(child_name="002", actual_combined_status="SAM",
             pred_status_final=""),                         # no verdict
        _row(child_name="003", actual_combined_status="SAM",
             pred_status_final="", error="unparseable date_of_birth: 'xx'"),
        _row(child_name="004"),                             # normal control
    ]
    a = analyze(rows)

    assert a["errored"] == 1
    assert a["errored_sam"] == 1, (
        "an errored child with gold-standard SAM must not vanish silently"
    )
    assert a["excluded_unverdicted_sam"] == 1

    # Headline stays paired-only, and still reads 1.000 on its own terms...
    assert a["sam"]["tp"] == 1 and a["sam"]["fn"] == 0

    # ...but the report must now surface both excluded SAM children and the
    # worst-case bound, so 1.000 can't be read in isolation.
    text = render_report(a, coverage([], [], rows))
    assert "row errored" in text
    assert "Worst-case sensitivity" in text
    # tp=1 over denominator 1 paired + 1 unverdicted + 1 errored = 0.333
    assert "0.333" in text


def test_errored_row_with_blank_gold_standard_still_flagged_via_raw_muac():
    """A garbled DOB leaves the age unknown, so `_muac_status` correctly
    declines to classify and `actual_combined_status` comes back blank.

    Blank means indeterminate, not negative. Counting only `== "SAM"`
    would read it as negative and let a child with a 10.5 cm arm vanish
    into a bare exclusion tally — the exact scenario the errored-row
    accounting exists to prevent. The raw tape reading survives the error,
    so the count keys on that.
    """
    rows = [
        _row(child_name="001", actual_combined_status="SAM",
             pred_status_final="SAM"),
        _row(child_name="002", actual_combined_status="", muac_cm="10.5",
             error="unparseable date_of_birth: 'xx'"),
        _row(child_name="003", actual_combined_status="", actual_oedema="yes",
             error="unparseable date_of_birth: 'yy'"),
        _row(child_name="004", actual_combined_status="", muac_cm="13.0",
             error="pose failed"),          # errored but no sign of SAM
    ]
    a = analyze(rows)

    assert a["errored"] == 3
    assert a["errored_sam"] == 0, "gold standard is genuinely indeterminate"
    assert a["errored_possible_sam"] == 2, (
        "the 10.5cm arm and the oedema tick must both still be counted"
    )

    text = render_report(a, coverage([], [], rows))
    assert "SAM or possibly SAM: 2" in text
    # tp=1 over 1 paired + 2 possibly-SAM = 0.333
    assert "0.333" in text


def test_errored_row_from_pipeline_exception_still_carries_muac_and_oedema():
    """A pipeline exception routes through `_error_row`, which used to drop
    the ground truth entirely — so a child whose pose step threw with a
    10.5 cm arm recorded vanished from the study. Same blind spot as the
    garbled-DOB case, reached by a different route."""
    from scripts.batch_assess import _error_row

    row = _error_row(
        fname="c.jpg", child_name="007", age_months=24.0, sex="M",
        actual_height=None, actual_weight=None,
        manual_muac=10.5, oedema="", error_msg="pose model raised",
    )
    assert row["muac_cm"] == 10.5

    a = analyze([_row(child_name="001"), {**_row(child_name="007"), **row}])
    assert a["errored_possible_sam"] == 1, (
        "an exception row with a SAM-range arm must still be counted"
    )


# --- Stunting (height-for-age) analysis -------------------------------------
#
# Mirrors the SAM/MAM wasting analysis above: `actual_haz_status` (gold
# standard, from measured height) vs `pred_haz_status` (from the app's
# estimated height). The four-level scale (Severely Stunted / Stunted /
# Normal / Tall) collapses to a clinical positive/negative question for
# sensitivity/specificity, with severe stunting broken out on its own —
# exactly like SAM gets its own headline separate from SAM+MAM — because a
# missed severely-stunted case is the dangerous direction.

from scripts.analyze_results import HAZ_CATS  # noqa: E402
from scripts.study_stats import weighted_kappa  # noqa: E402


def test_stunting_collapses_four_level_scale_to_positive_negative():
    rows = [
        _row(child_name="1", actual_haz_status="Stunted",
             pred_haz_status="Severely Stunted"),          # TP (pos/pos)
        _row(child_name="2", actual_haz_status="Tall",
             pred_haz_status="Normal"),                    # TN (neg/neg)
        _row(child_name="3", actual_haz_status="Normal",
             pred_haz_status="Stunted"),                   # FP
        _row(child_name="4", actual_haz_status="Severely Stunted",
             pred_haz_status="Tall"),                       # FN
    ]
    a = analyze(rows)
    st = a["stunting"]
    assert (st["tp"], st["fp"], st["tn"], st["fn"]) == (1, 1, 1, 1)
    v, _, _ = st["sensitivity"]
    assert v == 0.5


def test_severe_stunting_sensitivity_counted_separately_from_all_stunting():
    # One Severely Stunted child correctly flagged, one Stunted child
    # correctly flagged: "any stunting" sensitivity is 2/2, but severe
    # stunting sensitivity must only be computed over the 1 severe case.
    rows = [
        _row(child_name="1", actual_haz_status="Severely Stunted",
             pred_haz_status="Severely Stunted"),
        _row(child_name="2", actual_haz_status="Stunted",
             pred_haz_status="Stunted"),
    ]
    a = analyze(rows)
    v_all, _, _ = a["stunting"]["sensitivity"]
    assert v_all == 1.0
    assert a["severe_stunting"]["tp"] == 1 and a["severe_stunting"]["fn"] == 0
    v_severe, _, _ = a["severe_stunting"]["sensitivity"]
    assert v_severe == 1.0
    # The "Stunted" (non-severe) row must not have entered the severe
    # confusion matrix as a TP/FN at all — only as a TN/FP slot.
    assert a["severe_stunting"]["tp"] + a["severe_stunting"]["fn"] == 1


def test_severe_stunting_sensitivity_none_when_no_severe_cases():
    rows = [
        _row(child_name="1", actual_haz_status="Normal",
             pred_haz_status="Normal"),
        _row(child_name="2", actual_haz_status="Stunted",
             pred_haz_status="Stunted"),
    ]
    a = analyze(rows)
    assert a["severe_stunting"]["sensitivity"] is None
    assert a["severe_stunting"]["ppv"] is None


def test_haz_cats_ordering_scores_far_errors_worse():
    # Severely Stunted misread as Normal must cost more than Severely
    # Stunted misread as Stunted — this is what makes HAZ_CATS's order
    # (not, say, alphabetical) load-bearing for the weighted kappa call.
    actual = ["Severely Stunted", "Stunted", "Normal", "Tall"]
    near = ["Stunted", "Stunted", "Normal", "Tall"]   # SevStunted -> Stunted
    far = ["Normal", "Stunted", "Normal", "Tall"]     # SevStunted -> Normal
    assert weighted_kappa(actual, far, HAZ_CATS) < weighted_kappa(actual, near, HAZ_CATS)


def test_stunting_kappa_via_analyze_orders_severity():
    rows_near = [
        _row(child_name="1", actual_haz_status="Severely Stunted",
             pred_haz_status="Stunted"),
        _row(child_name="2", actual_haz_status="Stunted",
             pred_haz_status="Stunted"),
        _row(child_name="3", actual_haz_status="Normal",
             pred_haz_status="Normal"),
        _row(child_name="4", actual_haz_status="Tall",
             pred_haz_status="Tall"),
    ]
    rows_far = [
        _row(child_name="1", actual_haz_status="Severely Stunted",
             pred_haz_status="Normal"),
        _row(child_name="2", actual_haz_status="Stunted",
             pred_haz_status="Stunted"),
        _row(child_name="3", actual_haz_status="Normal",
             pred_haz_status="Normal"),
        _row(child_name="4", actual_haz_status="Tall",
             pred_haz_status="Tall"),
    ]
    a_near = analyze(rows_near)
    a_far = analyze(rows_far)
    assert a_far["stunting_kappa"] < a_near["stunting_kappa"]


def test_stunting_kappa_none_for_single_category():
    rows = [_row(child_name="1", actual_haz_status="Normal",
                  pred_haz_status="Normal")]
    a = analyze(rows)
    assert a["stunting_kappa"] is None


def test_unverdicted_gold_stunted_rows_excluded_from_matrix_and_counted():
    rows = [
        _row(child_name="001", actual_haz_status="Severely Stunted",
             pred_haz_status="Severely Stunted"),   # correctly verdicted
        _row(child_name="002", actual_haz_status="Severely Stunted",
             pred_haz_status=""),                   # height estimate missing
        _row(child_name="003", actual_haz_status="Severely Stunted",
             pred_haz_status=""),                   # height estimate missing
    ]
    a = analyze(rows)

    assert a["severe_stunting"]["tp"] == 1 and a["severe_stunting"]["fn"] == 0
    v, _, _ = a["severe_stunting"]["sensitivity"]
    assert v == 1.0

    # The other two must be counted as excluded, not silently dropped.
    assert a["excluded_unverdicted_stunting"] == 2
    assert a["excluded_unverdicted_severe_stunting"] == 2


def test_stunting_status_n_excludes_rows_with_no_gold_standard():
    # A row with no actual_haz_status at all (e.g. no measured height) must
    # not enter paired or excluded counts for stunting.
    rows = [
        _row(child_name="1", actual_haz_status="Normal",
             pred_haz_status="Normal"),
        _row(child_name="2"),  # no actual_haz_status / pred_haz_status set
    ]
    a = analyze(rows)
    assert a["stunting_status_n"] == 1
    assert a["excluded_unverdicted_stunting"] == 0


def test_render_report_contains_stunting_sections():
    rows = [
        _row(actual_haz_status="Severely Stunted",
             pred_haz_status="Severely Stunted"),
    ]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], rows),
    )
    assert "Stunting" in text
    assert "Severe stunting" in text
    assert "Weighted" in text


def test_render_report_stunting_sensitivity_never_shown_without_exclusion_count():
    rows = [
        _row(child_name="001", actual_haz_status="Severely Stunted",
             pred_haz_status="Severely Stunted"),
        _row(child_name="002", actual_haz_status="Severely Stunted",
             pred_haz_status=""),
        _row(child_name="003", actual_haz_status="Severely Stunted",
             pred_haz_status=""),
    ]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": c} for c in ("001", "002", "003")], [], rows),
    )
    start = text.index("Stunting")
    section = text[start:]
    assert "excluded" in section.lower()
    assert "2" in section


def test_render_report_subgroup_stunting_sensitivity():
    rows = [
        _row(child_name="001", sex="M", age_months="30.0",
             actual_haz_status="Severely Stunted",
             pred_haz_status="Severely Stunted"),
        _row(child_name="002", sex="F", age_months="30.0",
             actual_haz_status="Normal", pred_haz_status="Normal"),
    ]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}, {"child_id": "002"}], [], rows),
    )
    start = text.index("### sex=M")
    end = text.index("### sex=F")
    section = text[start:end]
    assert "stunting" in section.lower()
