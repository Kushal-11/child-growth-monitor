"""
Stage 4b — turn batch_assess results into the comparison-study report.

Reads the batch results CSV (+ QC report and intake manifest for coverage
accounting) and writes field_data/reports/study_report.md with
Bland-Altman agreement, SAM/MAM sensitivity-specificity with Wilson CIs,
weighted kappa, subgroup breakdowns, and strict coverage buckets.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/analyze_results.py
    PYTHONPATH=. .venv/bin/python scripts/analyze_results.py \
        --results field_data/reports/batch_results.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.study_stats import (  # noqa: E402
    bland_altman, binary_metrics, confusion_binary, weighted_kappa,
)

STATUS_CATS = ["SAM", "MAM", "Normal"]

# Published yardsticks (see docs/ml_pipeline_improvement_and_feedback_loop.md)
SMART_HEIGHT_TOLERANCE_CM = 1.4
WHO_TEM_HEIGHT_CM = 0.7


def _f(row: dict, key: str) -> Optional[float]:
    try:
        return float((row.get(key) or "").strip())
    except (TypeError, ValueError):
        return None


def _pairs(rows: list[dict], a_key: str, p_key: str) -> tuple[list, list]:
    actual, pred = [], []
    for r in rows:
        a, p = _f(r, a_key), _f(r, p_key)
        if a is not None and p is not None:
            actual.append(a)
            pred.append(p)
    return actual, pred


def _status_pairs(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Full gold standard (`actual_combined_status`) vs the app's verdict."""
    actual, pred = [], []
    for r in rows:
        a = (r.get("actual_combined_status") or "").strip()
        p = (r.get("pred_status_final") or "").strip()
        if a in STATUS_CATS and p in STATUS_CATS:
            actual.append(a)
            pred.append(p)
    return actual, pred


def _collapse_whz(status: Optional[str]) -> Optional[str]:
    """Collapse the 5-class WHZ-only scale to SAM/MAM/Normal.

    A blank/unknown WHZ status is `None` (unknown), not "Normal" — this
    matters for the detectability count below, where "no WHZ status" means
    the gold-standard SAM verdict could not have come from the WHZ arm.
    """
    s = (status or "").strip()
    if not s:
        return None
    return s if s in ("SAM", "MAM") else "Normal"


def _status_pairs_whz(rows: list[dict]) -> tuple[list[str], list[str]]:
    """WHZ arm alone (`actual_whz_status`, collapsed) vs the app's verdict.

    Isolates the app's own height/weight-estimation quality from the
    structural gap caused by having no oedema or MUAC input.
    """
    actual, pred = [], []
    for r in rows:
        a = _collapse_whz(r.get("actual_whz_status"))
        p = (r.get("pred_status_final") or "").strip()
        if a is not None and p in STATUS_CATS:
            actual.append(a)
            pred.append(p)
    return actual, pred


def _confusion_block(actual: list[str], pred: list[str]) -> dict:
    """SAM / SAM+MAM confusion matrices and weighted kappa for one framing."""
    out: dict = {"status_n": len(actual)}
    for name, positive in (("sam", {"SAM"}), ("sam_mam", {"SAM", "MAM"})):
        tp, fp, tn, fn = confusion_binary(actual, pred, positive)
        out[name] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     **binary_metrics(tp, fp, tn, fn)}
    out["kappa"] = weighted_kappa(actual, pred, STATUS_CATS) if len(set(actual)) > 1 else None
    return out


def _sam_detectability(rows: list[dict]) -> dict:
    """Among gold-standard SAM cases, how many were reachable via the WHZ
    arm (the only arm the app can see) versus only via MUAC or oedema (arms
    the app has no input for in this pipeline).

    This is what makes the gap between the two SAM-sensitivity framings
    interpretable rather than looking like an unexplained ML failure.
    """
    whz_detectable = 0
    muac_or_oedema_only = 0
    for r in rows:
        if (r.get("actual_combined_status") or "").strip() != "SAM":
            continue
        if _collapse_whz(r.get("actual_whz_status")) == "SAM":
            whz_detectable += 1
        else:
            muac_or_oedema_only += 1
    return {
        "total_actual_sam": whz_detectable + muac_or_oedema_only,
        "whz_detectable": whz_detectable,
        "muac_or_oedema_only": muac_or_oedema_only,
    }


def _analyze_block(rows: list[dict]) -> dict:
    """Metrics for one set of rows (whole study or one subgroup)."""
    out: dict = {}
    ah, ph = _pairs(rows, "actual_height_cm", "pred_height_cm")
    aw, pw = _pairs(rows, "actual_weight_kg", "pred_weight_ml_kg")
    out["height"] = bland_altman(ah, ph) if ah else {"n": 0}
    out["weight"] = bland_altman(aw, pw) if aw else {"n": 0}

    # Framing A: full WHO OR-rule gold standard (oedema OR MUAC OR WHZ).
    sa, sp = _status_pairs(rows)
    cb = _confusion_block(sa, sp)
    out["status_n"] = cb["status_n"]
    out["sam"] = cb["sam"]
    out["sam_mam"] = cb["sam_mam"]
    out["kappa"] = cb["kappa"]

    # Framing B: WHZ arm alone — isolates the app's own estimation quality
    # from the structural blind spot to oedema/MUAC-only SAM cases.
    sa_whz, sp_whz = _status_pairs_whz(rows)
    cb_whz = _confusion_block(sa_whz, sp_whz)
    out["status_n_whz"] = cb_whz["status_n"]
    out["sam_whz"] = cb_whz["sam"]
    out["sam_mam_whz"] = cb_whz["sam_mam"]
    out["kappa_whz"] = cb_whz["kappa"]

    return out


def analyze(results: list[dict]) -> dict:
    """Pure analysis over assessed rows (rows with error are excluded)."""
    rows = [r for r in results if not (r.get("error") or "").strip()]
    out = _analyze_block(rows)
    out["sam_detectability"] = _sam_detectability(rows)
    out["subgroups"] = {}
    partitions = {
        "sex=M": [r for r in rows if (r.get("sex") or "").upper() == "M"],
        "sex=F": [r for r in rows if (r.get("sex") or "").upper() == "F"],
        "age 6-23m": [r for r in rows
                      if (_f(r, "age_months") or 0) < 24],
        "age 24-59m": [r for r in rows
                       if (_f(r, "age_months") or 0) >= 24],
    }
    for name, part in partitions.items():
        out["subgroups"][name] = _analyze_block(part)
    return out


def coverage(
    intake_rows: list[dict], qc_rows: list[dict], results: list[dict],
) -> dict:
    """Every child in exactly one bucket; buckets must sum to total.

    Rows with a non-empty `error` column are treated as not-assessed here
    too (they land in missing_data), matching the exclusion in `analyze()` —
    an errored row's ground truth/prediction cannot be trusted for coverage
    accounting either.
    """
    total_ids = {r["child_id"] for r in intake_rows}
    qc_failed_ids = {
        r["child_id"] for r in qc_rows if r.get("verdict") == "failed"
    }
    assessed_ids = {
        (r.get("child_name") or "").strip()
        for r in results if not (r.get("error") or "").strip()
    } & total_ids
    qc_failed_ids &= total_ids - assessed_ids
    missing_ids = total_ids - assessed_ids - qc_failed_ids

    cov = {
        "total": len(total_ids),
        "assessed": len(assessed_ids),
        "qc_failed": len(qc_failed_ids),
        "missing_data": len(missing_ids),
        "missing_ids": sorted(missing_ids),
        "discrepancy": "",
    }
    if cov["assessed"] + cov["qc_failed"] + cov["missing_data"] != cov["total"]:
        cov["discrepancy"] = (
            "BUCKET SUM MISMATCH — investigate before trusting this report"
        )
    return cov


def _fmt_rate(r: Optional[tuple]) -> str:
    if r is None:
        return "n/a (zero denominator)"
    v, lo, hi = r
    return f"{v:.3f} (95% CI {lo:.3f}-{hi:.3f})"


def _fmt_kappa(kappa: Optional[float]) -> str:
    return f"{kappa:.3f}" if kappa is not None else "n/a (needs ≥2 distinct actual categories)"


def _fmt_ba(b: dict, unit: str) -> list[str]:
    if b.get("n", 0) == 0:
        return ["No paired values."]
    lines = [
        f"- n = {b['n']}",
        f"- Mean bias (pred − actual): {b['bias']:+.2f} {unit}",
    ]
    if b.get("loa_low") is None or b.get("loa_high") is None:
        lines.append(
            "- 95% limits of agreement: n/a (need at least 2 pairs to estimate)"
        )
    else:
        lines.append(
            f"- 95% limits of agreement: {b['loa_low']:+.2f} to "
            f"{b['loa_high']:+.2f} {unit}"
        )
    lines.append(f"- MAE: {b['mae']:.2f} {unit}")
    return lines


def render_report(analysis: dict, cov: dict) -> str:
    lines: list[str] = ["# App vs Manual Measurement — Comparison Study", ""]

    lines += ["## Coverage", ""]
    lines += [
        f"- Total children (intake manifest): {cov['total']}",
        f"- Assessed: {cov['assessed']}",
        f"- QC-failed (recapture list): {cov['qc_failed']}",
        f"- Missing data (no ground truth / not cleaned / errored): "
        f"{cov['missing_data']}",
    ]
    if cov["missing_ids"]:
        lines += [f"- Missing-data IDs: {', '.join(cov['missing_ids'])}"]
    if cov["discrepancy"]:
        lines += ["", f"**{cov['discrepancy']}**"]
    lines += [""]

    lines += ["## Height agreement (Bland–Altman)", ""]
    lines += _fmt_ba(analysis["height"], "cm")
    lines += [
        "",
        f"Yardsticks: SMART tolerance {SMART_HEIGHT_TOLERANCE_CM} cm; "
        f"WHO TEM {WHO_TEM_HEIGHT_CM} cm.",
        "",
    ]

    lines += ["## Weight agreement (Bland–Altman)", ""]
    lines += _fmt_ba(analysis["weight"], "kg")
    lines += [""]

    lines += ["## Status agreement", ""]
    lines += [
        "The app has no oedema input and no MUAC tape reading in this "
        "pipeline — `pred_status_final` is derived only from the app's "
        "photo-based WHZ estimate. It is therefore reported against two "
        "different gold standards below; see the explanatory note for why "
        "they diverge.",
        "",
    ]

    sam = analysis["sam"]
    lines += [
        "### SAM sensitivity — Framing A: full gold standard "
        "(oedema OR MUAC<11.5 OR WHZ<-3)",
        "",
        "Answers: can photo screening replace a full manual assessment? "
        "This is the honest headline number for deployment decisions.",
        "",
        f"- Paired statuses: n = {analysis['status_n']}",
        f"- Confusion: TP {sam['tp']}, FN {sam['fn']}, FP {sam['fp']}, "
        f"TN {sam['tn']}",
        f"- SAM sensitivity: {_fmt_rate(sam['sensitivity'])}",
        f"- SAM specificity: {_fmt_rate(sam['specificity'])}",
        f"- PPV: {_fmt_rate(sam['ppv'])}   NPV: {_fmt_rate(sam['npv'])}",
        "",
    ]

    sam_whz = analysis["sam_whz"]
    lines += [
        "### SAM sensitivity — Framing B: WHZ arm alone "
        "(manual height/weight only, collapsed to SAM/MAM/Normal)",
        "",
        "Answers: is the app's weight/height estimate itself any good? "
        "This isolates ML quality from the structural blind spot to "
        "oedema/MUAC-only cases.",
        "",
        f"- Paired statuses: n = {analysis['status_n_whz']}",
        f"- Confusion: TP {sam_whz['tp']}, FN {sam_whz['fn']}, "
        f"FP {sam_whz['fp']}, TN {sam_whz['tn']}",
        f"- SAM sensitivity: {_fmt_rate(sam_whz['sensitivity'])}",
        f"- SAM specificity: {_fmt_rate(sam_whz['specificity'])}",
        "",
    ]

    det = analysis["sam_detectability"]
    lines += [
        "**Why these differ:** the app cannot in principle detect SAM "
        "cases triggered only by oedema or only by MUAC, because it never "
        "receives that data. Of the "
        f"{det['total_actual_sam']} gold-standard SAM cases in this batch, "
        f"{det['whz_detectable']} were detectable in principle from the "
        f"WHZ arm and {det['muac_or_oedema_only']} were reachable only via "
        "MUAC or oedema. A gap between Framing A and Framing B sensitivity "
        "reflects this structural blind spot, not necessarily a defect in "
        "the underlying model.",
        "",
    ]

    sm = analysis["sam_mam"]
    lines += [
        "### SAM+MAM combined (Framing A, full gold standard)",
        "",
        f"- Sensitivity: {_fmt_rate(sm['sensitivity'])}",
        f"- Specificity: {_fmt_rate(sm['specificity'])}",
        "",
    ]
    lines += [
        f"- Weighted κ (linear, SAM>MAM>Normal), Framing A: "
        + _fmt_kappa(analysis["kappa"]),
        f"- Weighted κ (linear, SAM>MAM>Normal), Framing B (WHZ arm "
        "alone): " + _fmt_kappa(analysis["kappa_whz"]),
        "",
    ]

    lines += ["## Subgroups", ""]
    for name, block in analysis["subgroups"].items():
        h, w = block["height"], block["weight"]
        s = block["sam"]
        lines += [
            f"### {name}",
            "",
            f"- Height n={h.get('n', 0)}"
            + (f", bias {h['bias']:+.2f} cm, MAE {h['mae']:.2f} cm"
               if h.get("n") else ""),
            f"- Weight n={w.get('n', 0)}"
            + (f", bias {w['bias']:+.2f} kg, MAE {w['mae']:.2f} kg"
               if w.get("n") else ""),
            f"- SAM sensitivity (Framing A, full gold standard): "
            f"{_fmt_rate(s['sensitivity'])}",
            "",
        ]

    lines += [
        "---",
        "",
        "Interpretation notes: Bland–Altman per Lancet 1986; Wilson CIs; "
        "weighted κ because SAM→Normal errors are worse than SAM→MAM. "
        "A small SAM count makes sensitivity CIs wide — enrich with "
        "known-malnourished sites if the CI is too wide to act on.",
    ]
    return "\n".join(lines)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("field_data/reports/batch_results.csv"))
    parser.add_argument("--qc-report", type=Path,
                        default=Path("field_data/reports/qc_report.csv"))
    parser.add_argument("--intake", type=Path,
                        default=Path("field_data/reports/intake_manifest.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("field_data/reports/study_report.md"))
    args = parser.parse_args()

    results = _load(args.results)
    if not results:
        print(f"No results at {args.results} — run batch_assess first.",
              file=sys.stderr)
        sys.exit(1)

    intake_rows = _load(args.intake)
    qc_rows = _load(args.qc_report)
    if not intake_rows:
        print("WARNING: no intake manifest — coverage uses results rows only.")
        intake_rows = [
            {"child_id": (r.get("child_name") or "").strip()} for r in results
        ]

    report = render_report(analyze(results),
                           coverage(intake_rows, qc_rows, results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
