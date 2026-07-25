"""
Assessment orchestrator service.

Ties together measurement (image processing) and nutrition (Z-score) services.
Handles the full flow: image -> measurements -> WHO lookup -> classification.

Height resolution priority:
  1. Manual height_cm input
  2. Image-based measurement/screening estimate
"""
from datetime import date, datetime, time, timezone
import math
from typing import Optional

from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from app.schemas.assessment import (
    AssessmentResponse,
    MeasurementDetail,
    MLPrediction,
    MUACDetail,
    NutritionDetail,
    PoshanDetail,
)
from app.services.age_service import age_months_at, completed_months
from app.services.measurement_service import MeasurementOutput, MeasurementService
from app.services.ml_service import MLService
from app.services.nutrition_service import NutritionService
from app.services.poshan_setu_service import (
    ELIGIBLE_BMI_SOURCES,
    classify_poshan_setu,
    normalize_source,
)
from app.services.who_data_service import WHODataService


class AssessmentService:
    def __init__(self, who_data: WHODataService):
        self.measurement_svc = MeasurementService()
        self.nutrition_svc = NutritionService(who_data)
        self.who_data = who_data
        self.ml_svc = MLService()

    def assess(
        self,
        db: Session,
        image_path: Optional[str],
        child_name: str,
        dob: date,
        sex: str,
        weight_kg: Optional[float] = None,
        height_cm: Optional[float] = None,
        guardian_name: Optional[str] = None,
        location: Optional[str] = None,
        muac_cm: Optional[float] = None,
        side_image: Optional[bytes] = None,
        assessment_date: Optional[date] = None,
    ) -> AssessmentResponse:
        """Run full assessment pipeline and persist results."""

        # 1. Compute age in months
        today = datetime.now(timezone.utc).date()
        assessed_on = assessment_date or today
        if assessed_on > today:
            raise ValueError("assessment_date must not be in the future")
        if assessed_on < dob:
            raise ValueError("assessment_date must not be before date_of_birth")
        age_months = age_months_at(dob, assessed_on)
        self._validate_inputs(
            child_name=child_name,
            sex=sex,
            age_months=age_months,
            weight_kg=weight_kg,
            height_cm=height_cm,
            muac_cm=muac_cm,
        )

        # 2. Process an image only when one was supplied. Authoritative manual
        # height/weight/MUAC can produce a Poshan assessment without MediaPipe.
        if image_path:
            meas: MeasurementOutput = (
                self.measurement_svc.process_image_with_estimation(
                    image_path=image_path,
                    age_months=age_months,
                    sex=sex,
                    who_data=self.who_data,
                )
            )
        else:
            meas = MeasurementOutput()

        # 3. Determine effective height.  Manual measurements always win.
        if height_cm is not None:
            effective_height = height_cm
            height_source = "manual"
        elif self._is_finite_in_range(
            meas.predicted_height_cm, 30.0, 130.0
        ):
            effective_height = meas.predicted_height_cm
            height_source = normalize_source(meas.estimation_method)
        else:
            effective_height = None
            height_source = "unavailable"

        # 3b. Process side-view image for AP depth features (optional)
        side_segments = None
        if side_image is not None and effective_height is not None:
            side_segments = self.measurement_svc.process_side_image(
                side_image, effective_height
            )

        # 4a. Run ML prediction (uses body proportions from pose landmarks)
        ml_pred = None
        if effective_height is not None and meas.body_segments is not None:
            ml_pred = self.ml_svc.predict(
                meas.body_segments, age_months, sex, effective_height, side_segments
            )

        # 4b. Determine effective weight
        # Priority: manual_weight > ML-estimated > WHO-median (slender/stocky adjusted)
        effective_weight = weight_kg
        estimated_weight = None
        weight_source = "manual" if weight_kg is not None else "unavailable"

        if effective_weight is None:
            # Try ML weight estimate first (captures wasting signal)
            if ml_pred is not None and ml_pred.estimated_weight_kg is not None and effective_height is not None:
                ml_weight = ml_pred.estimated_weight_kg
                # Sanity check against WHO physiological bounds.
                # If ML output is outside 45–180% of WHO median, bad input features
                # (e.g. frontal photo uploaded as side view) caused extrapolation —
                # fall through to WHO median instead.
                who_median_ref = self.who_data.get_median_weight_for_height(
                    sex, effective_height, age_months=age_months
                )
                weight_in_bounds = (
                    self._is_finite_in_range(ml_weight, 0.5, 40.0)
                    and self._is_finite_in_range(
                        who_median_ref, 0.5, 40.0
                    )
                    and 0.45 * who_median_ref
                    <= ml_weight
                    <= 1.80 * who_median_ref
                )
                if weight_in_bounds:
                    effective_weight = ml_weight
                    estimated_weight = effective_weight
                    weight_source = "ml_estimated"

            if effective_weight is None and effective_height is not None:
                # Fall back to WHO median with body build adjustment
                estimated_weight = self.who_data.get_median_weight_for_height(
                    sex, effective_height, age_months=age_months
                )
                if estimated_weight is not None:
                    weight_adjustment = getattr(meas, 'weight_adjustment', 1.0)
                    estimated_weight = estimated_weight * weight_adjustment
                    if not self._is_finite_in_range(
                        estimated_weight, 0.5, 40.0
                    ):
                        estimated_weight = None
                effective_weight = estimated_weight
                if effective_weight is not None:
                    weight_source = "who_statistical"

        # 5. Compute Z-scores
        haz_z = None
        whz_z = None
        haz_status = None
        whz_status = None

        reliable_height = height_source in ELIGIBLE_BMI_SOURCES
        reliable_weight = weight_source in ELIGIBLE_BMI_SOURCES

        if effective_height is not None and reliable_height:
            haz_z = self.nutrition_svc.compute_haz(
                sex, completed_months(dob, assessed_on), effective_height
            )
            if haz_z is not None:
                haz_status = self.nutrition_svc.classify_haz(haz_z)

        if (
            effective_height is not None
            and effective_weight is not None
            and reliable_height
            and reliable_weight
        ):
            whz_z = self.nutrition_svc.compute_whz(
                sex, age_months, effective_height, effective_weight
            )
            if whz_z is not None:
                whz_status = self.nutrition_svc.classify_whz(whz_z)

        # 5b. The canonical MUAC field is reserved for a tape/manual value.
        # Landmark/WHZ-derived estimates must not be presented or persisted as
        # though a field worker measured the child's arm.
        muac_method = "manual" if muac_cm is not None else "unavailable"
        effective_muac_cm = muac_cm
        muac_age_in_range = 6.0 <= age_months < 60.0

        # 5c. Authoritative Poshan Setu result.  WHO/ML/MUAC estimates above
        # remain screening outputs and cannot certify a Normal result.
        poshan = classify_poshan_setu(
            sex=sex,
            age_months=age_months,
            weight_kg=effective_weight,
            height_cm=effective_height,
            weight_source=weight_source,
            height_source=height_source,
            muac_cm=effective_muac_cm,
            muac_method=muac_method,
        )

        # 6. Persist to database
        child = self._get_or_create_child(
            db, child_name, dob, sex, guardian_name, location
        )
        visit = Visit(
            child_id=child.id,
            visit_date=datetime.combine(assessed_on, time.min),
            age_months=age_months,
            image_path=image_path,
        )
        db.add(visit)
        db.flush()

        measurement_record = MeasurementResult(
            visit_id=visit.id,
            predicted_height_cm=meas.predicted_height_cm,
            predicted_weight_kg=estimated_weight,
            manual_height_cm=height_cm,
            manual_weight_kg=weight_kg,
            effective_height_cm=effective_height,
            effective_weight_kg=effective_weight,
            height_source=height_source,
            weight_source=weight_source,
            reference_object_detected=str(meas.reference_object_detected).lower(),
            scale_factor=meas.scale_factor,
            haz_zscore=haz_z,
            whz_zscore=whz_z,
            haz_status=haz_status,
            whz_status=whz_status,
            confidence_score=meas.confidence_score,
            body_build=(
                meas.body_build.get("body_build")
                if isinstance(meas.body_build, dict)
                else None
            ),
            side_view_used=False,
            ml_estimated_weight_kg=(
                ml_pred.estimated_weight_kg if ml_pred is not None else None
            ),
            ml_wasting_status=(
                ml_pred.wasting_status if ml_pred is not None else None
            ),
            ml_model_version=(
                getattr(ml_pred, "model_version", None)
                if ml_pred is not None
                else None
            ),
            ml_training_data=(
                getattr(ml_pred, "training_data", None)
                if ml_pred is not None
                else None
            ),
            ml_non_clinical=(
                getattr(ml_pred, "non_clinical", True)
                if ml_pred is not None
                else None
            ),
            sam_probability=(
                ml_pred.sam_probability if ml_pred is not None else None
            ),
            mam_probability=(
                ml_pred.mam_probability if ml_pred is not None else None
            ),
            normal_probability=(
                ml_pred.normal_probability if ml_pred is not None else None
            ),
            risk_probability=(
                ml_pred.risk_probability if ml_pred is not None else None
            ),
            overweight_probability=(
                ml_pred.overweight_probability if ml_pred is not None else None
            ),
            muac_cm=effective_muac_cm,
            muac_status=poshan.muac_status,
            muac_method=muac_method,
            bmi=poshan.bmi,
            bmi_status=poshan.bmi_status,
            poshan_status=poshan.final_status,
            poshan_triggered_by=list(poshan.triggered_by),
            classification_method=poshan.classification_method,
            classification_rationale=poshan.rationale,
        )

        # Extract body build from measurement result.
        body_build_str = None
        if meas.body_build and isinstance(meas.body_build, dict):
            body_build_str = meas.body_build.get("body_build")
        
        # Compute depth in cm for response (if side view was used and measurements are valid)
        chest_depth_cm_out = None
        abd_depth_cm_out   = None
        if side_segments is not None and effective_height is not None and side_segments.total_height_px:
            side_scale = effective_height / side_segments.total_height_px
            # Reference widths for validation (Snyder 1975 mean ratios at ~36 months)
            approx_shoulder = effective_height * 0.211
            approx_hip      = approx_shoulder * 0.88
            if side_segments.chest_depth_px and side_segments.chest_confidence >= 0.5:
                raw = round(side_segments.chest_depth_px * side_scale, 1)
                # Accept only if within true side-view range (15–65% of shoulder width)
                if 0.15 * approx_shoulder < raw < 0.65 * approx_shoulder:
                    chest_depth_cm_out = raw
            if side_segments.abd_depth_px and side_segments.abd_confidence >= 0.5:
                raw = round(side_segments.abd_depth_px * side_scale, 1)
                if 0.15 * approx_hip < raw < 0.65 * approx_hip:
                    abd_depth_cm_out = raw

        # Complete the side-view persistence fields once validation is done.
        measurement_record.side_view_used = (
            chest_depth_cm_out is not None or abd_depth_cm_out is not None
        )
        measurement_record.chest_depth_cm = chest_depth_cm_out
        measurement_record.abd_depth_cm = abd_depth_cm_out
        db.add(measurement_record)
        db.commit()

        # 7. Build response
        summary = self._build_summary(
            child_name,
            age_months,
            effective_height,
            height_source,
            effective_weight,
            weight_source,
            haz_status,
            whz_status,
            effective_muac_cm,
            poshan.muac_status,
            poshan.final_status,
        )

        return AssessmentResponse(
            child_name=child_name,
            sex=sex,
            age_months=age_months,
            measurement=MeasurementDetail(
                predicted_height_cm=meas.predicted_height_cm,
                predicted_weight_kg=estimated_weight,
                manual_height_cm=height_cm,
                manual_weight_kg=weight_kg,
                effective_height_cm=effective_height,
                effective_weight_kg=effective_weight,
                height_source=height_source,
                weight_source=weight_source,
                reference_object_detected=meas.reference_object_detected,
                scale_factor=meas.scale_factor,
                confidence_score=meas.confidence_score,
                annotated_image=meas.annotated_image_filename,
                estimation_method=meas.estimation_method,
                body_build=body_build_str,
                side_view_used=chest_depth_cm_out is not None or abd_depth_cm_out is not None,
                chest_depth_cm=chest_depth_cm_out,
                abd_depth_cm=abd_depth_cm_out,
            ),
            nutrition=NutritionDetail(
                haz_zscore=haz_z,
                whz_zscore=whz_z,
                haz_status=haz_status,
                whz_status=whz_status,
                age_months=age_months,
            ),
            ml_prediction=MLPrediction(
                estimated_weight_kg=ml_pred.estimated_weight_kg if ml_pred else None,
                sam_probability=ml_pred.sam_probability if ml_pred else 0.0,
                mam_probability=ml_pred.mam_probability if ml_pred else 0.0,
                normal_probability=ml_pred.normal_probability if ml_pred else 0.0,
                risk_probability=ml_pred.risk_probability if ml_pred else 0.0,
                overweight_probability=ml_pred.overweight_probability if ml_pred else 0.0,
                wasting_status=ml_pred.wasting_status if ml_pred else None,
                wasting_method=ml_pred.wasting_method if ml_pred else "unavailable",
                model_version=getattr(ml_pred, "model_version", None),
                training_data=getattr(ml_pred, "training_data", None),
                non_clinical=getattr(ml_pred, "non_clinical", True),
            ) if ml_pred else None,
            muac=MUACDetail(
                muac_cm=effective_muac_cm,
                muac_status=poshan.muac_status,
                muac_method=muac_method,
                age_in_range=muac_age_in_range,
            ),
            poshan=PoshanDetail(
                bmi=poshan.bmi,
                bmi_status=poshan.bmi_status,
                muac_status=poshan.muac_status,
                final_status=poshan.final_status,
                triggered_by=list(poshan.triggered_by),
                classification_method=poshan.classification_method,
                rationale=poshan.rationale,
                complete=poshan.complete,
            ),
            summary=summary,
        )

    @staticmethod
    def _compute_age_months(dob: date, today: date) -> float:
        """Compute calendar-aware age in fractional months."""
        return age_months_at(dob, today)

    @staticmethod
    def _validate_inputs(
        *,
        child_name: str,
        sex: str,
        age_months: float,
        weight_kg: Optional[float],
        height_cm: Optional[float],
        muac_cm: Optional[float],
    ) -> None:
        """Reject invalid under-five metadata before running the pipeline."""
        if not child_name or not child_name.strip():
            raise ValueError("child_name must not be empty")
        if sex not in ("M", "F"):
            raise ValueError("sex must be 'M' or 'F'")
        if not math.isfinite(age_months) or not 0.0 <= age_months < 60.0:
            raise ValueError("date_of_birth must identify a child under 60 months")

        for field, value, lower, upper in (
            ("height_cm", height_cm, 30.0, 130.0),
            ("weight_kg", weight_kg, 0.5, 40.0),
            ("muac_cm", muac_cm, 5.0, 25.0),
        ):
            if value is None:
                continue
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"{field} must be finite and between {lower:g} and {upper:g}"
                )

    @staticmethod
    def _is_finite_in_range(
        value: Optional[float],
        lower: float,
        upper: float,
    ) -> bool:
        return (
            value is not None
            and math.isfinite(value)
            and lower <= value <= upper
        )

    @staticmethod
    def _get_or_create_child(
        db: Session,
        name: str,
        dob: date,
        sex: str,
        guardian_name: Optional[str],
        location: Optional[str],
    ) -> Child:
        """Find existing child by name+DOB+sex or create new."""
        child = (
            db.query(Child)
            .filter(
                Child.name == name,
                Child.date_of_birth == dob,
                Child.sex == sex,
            )
            .first()
        )
        if child is None:
            child = Child(
                name=name,
                date_of_birth=dob,
                sex=sex,
                guardian_name=guardian_name,
                location=location,
            )
            db.add(child)
            db.flush()
        return child

    @staticmethod
    def _build_summary(
        name,
        age_months,
        effective_height,
        height_source,
        effective_weight,
        weight_source,
        haz_status,
        whz_status,
        muac_cm,
        muac_status,
        poshan_status,
    ) -> str:
        """Build a human-readable summary string."""
        lines = [f"Assessment for {name} ({age_months:.1f} months old):"]

        if effective_height is not None:
            lines.append(
                f"  Height: {effective_height:.1f} cm ({height_source})"
            )
        else:
            lines.append("  Height: Could not be determined.")

        if effective_weight is not None:
            lines.append(
                f"  Weight: {effective_weight:.1f} kg ({weight_source})"
            )

        if muac_cm is not None:
            lines.append(f"  MUAC: {muac_cm:.1f} cm ({muac_status})")

        lines.append(f"  Poshan Setu: {poshan_status}")

        if haz_status:
            lines.append(f"  WHO stunting screen (HAZ): {haz_status}")
        if whz_status:
            lines.append(f"  WHO wasting screen (WHZ): {whz_status}")

        return "\n".join(lines)
