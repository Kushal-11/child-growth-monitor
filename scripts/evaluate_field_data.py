"""Evaluate real field-assessment results without copying child-level data.

Input is the aggregate-row CSV produced by ``scripts/batch_assess.py``.  The
command uses only rows backed by manual field measurements:

* weight MAE: actual_weight_kg vs pred_weight_ml_kg
* classification metrics: actual WHZ from measured height/weight vs ML class
* screening-vs-manual-Poshan agreement: Poshan Setu v1 gold status derived
  from measured height/weight/MUAC vs the legacy ML/WHZ screening verdict.
  This does not imply that the image-only app verdict is a Poshan result.

The JSON and Markdown outputs contain aggregate counts/metrics only—no child
IDs, filenames, dates, notes, or measurements.

Run:

    PYTHONPATH=. .venv/bin/python scripts/evaluate_field_data.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from app.services.poshan_setu_service import classify_poshan_setu


DEFAULT_RESULTS = Path("field_data/reports/batch_results.csv")
DEFAULT_JSON = Path("field_data/reports/field_evaluation.json")
DEFAULT_REPORT = Path("field_data/reports/field_evaluation.md")
SAM_RECALL_FLOOR = 0.80
REQUIRED_RESULT_COLUMNS = {
    "age_months",
    "sex",
    "actual_height_cm",
    "actual_weight_kg",
    "muac_cm",
    "actual_whz_status",
    "pred_weight_ml_kg",
    "ml_wasting_status",
    "effective_height_source",
    "ml_model_version",
    "ml_training_data",
    "ml_non_clinical",
    "ml_runtime_manifest_sha256",
    "pred_status_final",
    "error",
}


class FieldEvaluationError(ValueError):
    """Raised for an unusable field-results input."""


def _number(row: dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _collapse_status(value: Any) -> Optional[str]:
    status = str(value or "").strip()
    if status in ("SAM", "MAM"):
        return status
    if status in ("Normal", "Risk_Overweight", "Overweight", "Obese"):
        return "Normal"
    return None


def _ml_status(value: Any) -> Optional[str]:
    status = str(value or "").strip()
    return status if status in {
        "SAM", "MAM", "Normal", "Risk_Overweight", "Overweight",
    } else None


def manual_poshan_status(row: dict[str, Any]) -> str:
    """Poshan Setu v1 gold status from eligible manual field measurements."""

    sex = str(row.get("sex") or "").strip().upper()
    age = _number(row, "age_months")
    height = _number(row, "actual_height_cm")
    weight = _number(row, "actual_weight_kg")
    muac = _number(row, "muac_cm")
    result = classify_poshan_setu(
        sex=sex,
        age_months=age if age is not None else float("nan"),
        height_cm=height,
        weight_kg=weight,
        height_source="manual" if height is not None else "unavailable",
        weight_source="manual" if weight is not None else "unavailable",
        muac_cm=muac,
        muac_method="manual" if muac is not None else "unavailable",
    )
    return result.final_status


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "value": _ratio(numerator, denominator),
        "numerator": numerator,
        "denominator": denominator,
    }


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise FieldEvaluationError(
            "results CSV contains a header but no assessment rows; "
            "no field metrics can be calculated"
        )

    clean_rows = [
        row for row in rows if not str(row.get("error") or "").strip()
    ]
    prediction_rows = [
        row
        for row in rows
        if _ml_status(row.get("ml_wasting_status")) is not None
        or _number(row, "pred_weight_ml_kg") is not None
    ]
    provenance: set[tuple[str, str, str, str]] = set()
    for row in prediction_rows:
        values = (
            str(row.get("ml_model_version") or "").strip(),
            str(row.get("ml_training_data") or "").strip(),
            str(row.get("ml_non_clinical") or "").strip().lower(),
            str(row.get("ml_runtime_manifest_sha256") or "").strip().lower(),
        )
        if not all(values):
            raise FieldEvaluationError(
                "an ML prediction row is missing model version, training-data, "
                "non-clinical, or runtime-manifest provenance"
            )
        if values[2] not in ("true", "1", "yes"):
            raise FieldEvaluationError(
                "ML prediction provenance must explicitly mark the runtime "
                "as non-clinical"
            )
        if len(values[3]) != 64 or any(
            character not in "0123456789abcdef" for character in values[3]
        ):
            raise FieldEvaluationError(
                "ML runtime manifest provenance must be a SHA-256 hex digest"
            )
        provenance.add(values)
    if len(provenance) > 1:
        raise FieldEvaluationError(
            "results contain mixed ML model versions or runtime provenance; "
            "evaluate each promoted runtime separately"
        )
    model_provenance = None
    if provenance:
        version, training_data, non_clinical, manifest_hash = next(
            iter(provenance)
        )
        model_provenance = {
            "model_version": version,
            "training_data": training_data,
            "non_clinical": non_clinical in ("true", "1", "yes"),
            "runtime_manifest_sha256": manifest_hash,
        }
    weight_errors: list[float] = []
    weight_height_sources: Counter[str] = Counter()
    actual_classes: list[str] = []
    predicted_classes: list[str] = []
    classification_height_sources: Counter[str] = Counter()
    poshan_actual: list[str] = []
    poshan_predicted: list[str] = []
    poshan_indeterminate = 0
    end_to_end_sam_actual = 0
    end_to_end_sam_caught = 0
    end_to_end_sam_missing_or_error = 0

    # Safety-floor denominator is end-to-end: every manually classifiable SAM
    # child counts even if image processing errored or produced no prediction.
    # Otherwise a crash on the hardest SAM cases would improve the reported
    # recall by silently removing them.
    for row in rows:
        actual_poshan = manual_poshan_status(row)
        if actual_poshan == "Indeterminate":
            poshan_indeterminate += 1
            continue
        predicted_poshan = _collapse_status(row.get("pred_status_final"))
        if predicted_poshan is not None:
            poshan_actual.append(actual_poshan)
            poshan_predicted.append(predicted_poshan)
        if actual_poshan == "SAM":
            end_to_end_sam_actual += 1
            if predicted_poshan == "SAM" and not str(
                row.get("error") or ""
            ).strip():
                end_to_end_sam_caught += 1
            elif predicted_poshan is None or str(
                row.get("error") or ""
            ).strip():
                end_to_end_sam_missing_or_error += 1

    for row in clean_rows:
        actual_weight = _number(row, "actual_weight_kg")
        predicted_weight = _number(row, "pred_weight_ml_kg")
        height_source = str(
            row.get("effective_height_source") or "unknown_legacy_result"
        ).strip()
        if actual_weight is not None and predicted_weight is not None:
            weight_errors.append(abs(predicted_weight - actual_weight))
            weight_height_sources[height_source] += 1

        # actual_whz_status is produced only from measured height+weight by
        # batch_assess. Require those raw measurements as an additional guard.
        actual_height = _number(row, "actual_height_cm")
        actual_class = _ml_status(row.get("actual_whz_status"))
        predicted_class = _ml_status(row.get("ml_wasting_status"))
        if (
            actual_height is not None
            and actual_weight is not None
            and actual_class is not None
            and predicted_class is not None
        ):
            actual_classes.append(actual_class)
            predicted_classes.append(predicted_class)
            classification_height_sources[height_source] += 1

    correct = sum(
        actual == predicted
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    sam_actual = sum(status == "SAM" for status in actual_classes)
    sam_caught = sum(
        actual == "SAM" and predicted == "SAM"
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    mam_actual = sum(status == "MAM" for status in actual_classes)
    mam_caught = sum(
        actual == "MAM" and predicted == "MAM"
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    mam_predicted = sum(status == "MAM" for status in predicted_classes)
    mam_true_positive = mam_caught
    poshan_matches = sum(
        actual == predicted
        for actual, predicted in zip(poshan_actual, poshan_predicted)
    )

    conditional_sam_recall = _metric(sam_caught, sam_actual)
    sam_recall = _metric(end_to_end_sam_caught, end_to_end_sam_actual)
    return {
        "source": "real_field_measurements",
        "clinical_validity": "not_established",
        "privacy": "aggregate_only_no_child_level_values",
        "model_provenance": model_provenance,
        "rows": {
            "total": len(rows),
            "clean": len(clean_rows),
            "excluded_with_errors": len(rows) - len(clean_rows),
        },
        "weight_mae_kg": {
            "value": (
                sum(weight_errors) / len(weight_errors) if weight_errors else None
            ),
            "sample_count": len(weight_errors),
            "effective_height_source_counts": dict(sorted(weight_height_sources.items())),
            "manual_height_assisted_count": weight_height_sources["manual_measured"],
        },
        "classification_accuracy": {
            **_metric(correct, len(actual_classes)),
            "effective_height_source_counts": dict(
                sorted(classification_height_sources.items())
            ),
            "manual_height_assisted_count": (
                classification_height_sources["manual_measured"]
            ),
        },
        "sam_recall": {
            **sam_recall,
            "definition": "end_to_end_manual_poshan_sam",
            "missed_due_to_error_or_missing_prediction": (
                end_to_end_sam_missing_or_error
            ),
            "floor": SAM_RECALL_FLOOR,
            "floor_evaluable": sam_recall["value"] is not None,
            "floor_met": (
                sam_recall["value"] >= SAM_RECALL_FLOOR
                if sam_recall["value"] is not None
                else None
            ),
        },
        "conditional_ml_whz_sam_recall": {
            **conditional_sam_recall,
            "definition": "clean_rows_with_manual_whz_and_ml_prediction",
        },
        "mam_recall": _metric(mam_caught, mam_actual),
        "mam_precision": _metric(mam_true_positive, mam_predicted),
        "screening_vs_manual_poshan_agreement": {
            **_metric(poshan_matches, len(poshan_actual)),
            "gold_method": "poshan_setu_v1_manual_height_weight_muac",
            "prediction_method": "legacy_ml_whz_screening_not_poshan",
            "indeterminate_gold_rows": poshan_indeterminate,
        },
    }


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FieldEvaluationError(f"results CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_RESULT_COLUMNS.difference(columns))
        if missing:
            raise FieldEvaluationError(
                "results CSV is missing required columns: " + ", ".join(missing)
            )
        return list(reader)


def _display_metric(metric: dict[str, Any], digits: int = 3) -> str:
    value = metric.get("value")
    if value is None:
        return "not evaluable (no eligible pairs)"
    count = metric.get("sample_count", metric.get("denominator"))
    suffix = f" (n={count})" if count is not None else ""
    return f"{value:.{digits}f}{suffix}"


def render_markdown(result: dict[str, Any]) -> str:
    sam = result["sam_recall"]
    if sam["floor_evaluable"]:
        floor_text = "met" if sam["floor_met"] else "NOT MET"
        floor_line = (
            f"- SAM recall safety floor ({sam['floor']:.2f}): {floor_text}"
        )
    else:
        floor_line = (
            f"- SAM recall safety floor ({sam['floor']:.2f}): not evaluable "
            "(no measured SAM cases)"
        )
    return "\n".join(
        [
            "# Field Evaluation",
            "",
            "Aggregate-only evaluation from real manual field measurements.",
            "Clinical validity has not been established.",
            "",
            f"- Rows: {result['rows']['total']} total, "
            f"{result['rows']['clean']} clean, "
            f"{result['rows']['excluded_with_errors']} excluded with errors",
            f"- Weight MAE: {_display_metric(result['weight_mae_kg'])} kg",
            f"- Classification accuracy: "
            f"{_display_metric(result['classification_accuracy'])}",
            f"- SAM recall: {_display_metric(result['sam_recall'])}",
            f"- Conditional ML-vs-WHZ SAM recall (clean predicted rows): "
            f"{_display_metric(result['conditional_ml_whz_sam_recall'])}",
            f"- Manual Poshan SAM rows missed because processing errored or "
            f"returned no verdict: "
            f"{result['sam_recall']['missed_due_to_error_or_missing_prediction']}",
            f"- MAM recall: {_display_metric(result['mam_recall'])}",
            f"- MAM precision: {_display_metric(result['mam_precision'])}",
            f"- Legacy screening vs manual Poshan Setu v1 agreement: "
            f"{_display_metric(result['screening_vs_manual_poshan_agreement'])}",
            f"- Poshan gold rows indeterminate from missing/ineligible manual "
            f"components: "
            f"{result['screening_vs_manual_poshan_agreement']['indeterminate_gold_rows']}",
            f"- Weight predictions assisted by manual measured height: "
            f"{result['weight_mae_kg']['manual_height_assisted_count']}",
            f"- Classification predictions assisted by manual measured height: "
            f"{result['classification_accuracy']['manual_height_assisted_count']}",
            floor_line,
            "",
            "The legacy screening verdict is not reported as a Poshan result; "
            "this metric only compares that screening signal with manual Poshan gold.",
            "",
            "No child identifiers, filenames, notes, or row-level measurements "
            "are included in this report.",
            "",
        ]
    )


def _write_outputs(result: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    try:
        result = evaluate_rows(load_results(args.results))
    except FieldEvaluationError as exc:
        print(f"FIELD EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1

    _write_outputs(result, args.json_output, args.report_output)
    print(render_markdown(result))
    print(f"JSON written to {args.json_output}")
    print(f"Markdown written to {args.report_output}")
    if not result["sam_recall"]["floor_evaluable"]:
        print(
            "SAM recall safety floor is not evaluable because the field set "
            "contains no eligible manual Poshan SAM cases.",
            file=sys.stderr,
        )
        return 3
    if not result["sam_recall"]["floor_met"]:
        print(
            f"SAM recall is below the {SAM_RECALL_FLOOR:.2f} safety floor.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
