"""
Assessment orchestrator service.

Ties together measurement (image processing) and nutrition (Z-score) services.
Handles the full flow: image -> measurements -> WHO lookup -> classification.

Height resolution priority:
  1. Image-based (WHO statistical + anthropometric ratios)
  2. Manual height_cm input (fallback when image detection fails)
"""
from datetime import date, datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from app.schemas.assessment import (
    AssessmentResponse,
    CombinedNutritionDetail,
    MeasurementDetail,
    MLPrediction,
    MUACDetail,
    NutritionDetail,
)
from app.services.measurement_service import MeasurementOutput, MeasurementService
from app.services.ml_service import MLService
from app.services.muac_service import MUACService
from app.services.nutrition_service import NutritionService
from app.services.who_data_service import WHODataService
from config import (
    BMI_THRESHOLDS,
    ML_WEIGHT_MAX_MEDIAN_RATIO,
    ML_WEIGHT_MIN_MEDIAN_RATIO,
)


class AssessmentService:
    def __init__(self, who_data: WHODataService):
        self.measurement_svc = MeasurementService()
        self.nutrition_svc = NutritionService(who_data)
        self.who_data = who_data
        self.ml_svc = MLService()

    def assess(
        self,
        db: Session,
        image_path: str,
        child_name: str,
        dob: date,
        sex: str,
        weight_kg: Optional[float] = None,
        height_cm: Optional[float] = None,
        guardian_name: Optional[str] = None,
        location: Optional[str] = None,
        muac_cm: Optional[float] = None,
        side_image: Optional[bytes] = None,
    ) -> AssessmentResponse:
        """Run full assessment pipeline and persist results."""

        # 1. Compute age in months
        today = datetime.utcnow().date()
        age_months = self._compute_age_months(dob, today)

        # 2. Process image for height estimation using hybrid approach
        # Pass age, sex, and WHO data for statistical estimation
        meas: MeasurementOutput = self.measurement_svc.process_image_with_estimation(
            image_path=image_path,
            age_months=age_months,
            sex=sex,
            who_data=self.who_data
        )

        # 3. Determine effective height
        # A physical measurement always overrides an estimate.
        effective_height = height_cm if height_cm is not None else meas.predicted_height_cm
        height_method = "manual" if height_cm is not None else meas.estimation_method

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
        weight_source = "manual" if weight_kg is not None else None

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
                    who_median_ref is None
                    or (
                        ML_WEIGHT_MIN_MEDIAN_RATIO * who_median_ref
                        <= ml_weight
                        <= ML_WEIGHT_MAX_MEDIAN_RATIO * who_median_ref
                    )
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
                    estimated_weight = round(estimated_weight * weight_adjustment, 2)
                effective_weight = estimated_weight
                weight_source = "who_median_estimated"

        # 5. Compute Z-scores
        haz_z = None
        whz_z = None
        haz_status = None
        whz_status = None

        if effective_height is not None:
            haz_z = self.nutrition_svc.compute_haz(
                sex, int(round(age_months)), effective_height
            )
            if haz_z is not None:
                haz_status = self.nutrition_svc.classify_haz(haz_z)
                haz_z = round(haz_z, 2)

        if effective_height is not None and effective_weight is not None:
            whz_z = self.nutrition_svc.compute_whz(
                sex, age_months, effective_height, effective_weight
            )
            if whz_z is not None:
                whz_status = self.nutrition_svc.classify_whz(whz_z)
                whz_z = round(whz_z, 2)

        bmi = None
        bmi_status = None
        if effective_height is not None and effective_weight is not None:
            bmi = round(effective_weight / ((effective_height / 100.0) ** 2), 2)
            bmi_status = self._classify_bmi(bmi, sex)

        # 5b. Estimate MUAC — pathway priority: manual > landmark > WHZ-derived
        # Pull arm length & shoulder width from measurement service (in cm).
        # ml_service.extract_features applies the same pixel→cm scaling using
        # effective_height; do the same here for the landmark MUAC pathway.
        muac_arm_cm = None
        muac_shoulder_cm = None
        if (
            meas.body_segments is not None
            and effective_height is not None
            and meas.body_segments.total_height_px
            and meas.body_segments.total_height_px > 0
        ):
            scale = effective_height / meas.body_segments.total_height_px
            if meas.body_segments.upper_arm_length_px:
                muac_arm_cm = float(meas.body_segments.upper_arm_length_px * scale)
            if meas.body_segments.shoulder_width_px:
                muac_shoulder_cm = float(meas.body_segments.shoulder_width_px * scale)

        muac_result = MUACService.estimate(
            age_months=age_months,
            sex=sex,
            whz=whz_z,
            manual_muac_cm=muac_cm,
            upper_arm_length_cm=muac_arm_cm,
            shoulder_width_cm=muac_shoulder_cm,
            height_cm=effective_height,
        )

        # Requested programme protocol: BMI OR MUAC, worst result wins.
        programme_status = self._combine_bmi_muac(bmi_status, muac_result.muac_status)

        body_build_str = None
        if meas.body_build and isinstance(meas.body_build, dict):
            body_build_str = meas.body_build.get("body_build")

        # 6. Persist to database
        child = self._get_or_create_child(
            db, child_name, dob, sex, guardian_name, location
        )
        visit = Visit(
            child_id=child.id,
            age_months=round(age_months, 1),
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
            reference_object_detected=str(meas.reference_object_detected).lower(),
            scale_factor=meas.scale_factor,
            haz_zscore=haz_z,
            whz_zscore=whz_z,
            haz_status=haz_status,
            whz_status=whz_status,
            bmi=bmi,
            bmi_status=bmi_status,
            combined_status=programme_status.status,
            combined_triggered_by=",".join(programme_status.triggered_by),
            height_method=height_method,
            weight_method=weight_source,
            confidence_score=meas.confidence_score,
            body_build=body_build_str,
            ml_estimated_weight_kg=ml_pred.estimated_weight_kg if ml_pred else None,
            ml_wasting_status=ml_pred.wasting_status if ml_pred else None,
            sam_probability=ml_pred.sam_probability if ml_pred else None,
            mam_probability=ml_pred.mam_probability if ml_pred else None,
            normal_probability=ml_pred.normal_probability if ml_pred else None,
            risk_probability=ml_pred.risk_probability if ml_pred else None,
            overweight_probability=ml_pred.overweight_probability if ml_pred else None,
            muac_cm=muac_result.muac_cm,
            muac_status=muac_result.muac_status,
            muac_method=muac_result.muac_method,
        )
        db.add(measurement_record)
        db.commit()

        # 7. Build response
        summary = self._build_summary(
            child_name,
            age_months,
            effective_height,
            meas.predicted_height_cm,
            height_cm,
            effective_weight,
            haz_status,
            whz_status,
            meas.reference_object_detected,
            muac_result.muac_cm,
            muac_result.muac_status,
            programme_status.status,
        )

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

        return AssessmentResponse(
            child_name=child_name,
            sex=sex,
            age_months=round(age_months, 1),
            measurement=MeasurementDetail(
                predicted_height_cm=meas.predicted_height_cm,
                predicted_weight_kg=estimated_weight,
                manual_height_cm=height_cm,
                manual_weight_kg=weight_kg,
                reference_object_detected=meas.reference_object_detected,
                scale_factor=meas.scale_factor,
                confidence_score=meas.confidence_score,
                annotated_image=meas.annotated_image_filename,
                estimation_method=height_method,
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
                age_months=round(age_months, 1),
                bmi=bmi,
                bmi_status=bmi_status,
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
            ) if ml_pred else None,
            muac=MUACDetail(
                muac_cm=muac_result.muac_cm,
                muac_status=muac_result.muac_status,
                muac_method=muac_result.muac_method,
                age_in_range=muac_result.age_in_range,
            ),
            combined_nutrition=CombinedNutritionDetail(
                status=programme_status.status,
                triggered_by=programme_status.triggered_by,
                rationale=programme_status.rationale,
            ),
            summary=summary,
        )

    @staticmethod
    def _classify_bmi(bmi: float, sex: str) -> str:
        thresholds = BMI_THRESHOLDS[sex.upper()]
        if bmi < thresholds["sam_below"]:
            return "SAM"
        if bmi < thresholds["mam_below"]:
            return "MAM"
        return "Normal"

    @staticmethod
    def _combine_bmi_muac(bmi_status, muac_status):
        muac = "MAM" if muac_status == "At Risk (MAM)" else muac_status
        result = MUACService.combine_with_whz_status(muac, bmi_status)
        result.triggered_by = ["bmi" if item == "whz" else item for item in result.triggered_by]
        result.rationale = f"{result.status} from programme BMI/MUAC worst-indicator rule"
        return result

    @staticmethod
    def _compute_age_months(dob: date, today: date) -> float:
        """Compute age in fractional months."""
        delta = today - dob
        return delta.days / 30.4375

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
        name, age_months, effective_height, predicted_height, manual_height,
        weight, haz_status, whz_status, ref_detected,
        muac_cm=None, muac_status=None, combined_status=None,
    ) -> str:
        """Build a human-readable summary string."""
        lines = [f"Assessment for {name} ({age_months:.1f} months old):"]

        if effective_height is not None:
            if manual_height is not None:
                lines.append(f"  Height: {effective_height:.1f} cm (manual input)")
            elif predicted_height is not None:
                lines.append(f"  Height: {effective_height:.1f} cm (from image)")
        else:
            lines.append("  Height: Could not be determined.")
            if not ref_detected:
                lines.append(
                    "  Note: Image-based height estimation was not successful. "
                    "Provide manual height for classification."
                )

        if weight is not None:
            lines.append(f"  Weight: {weight:.1f} kg")

        if muac_cm is not None:
            lines.append(f"  MUAC: {muac_cm:.1f} cm ({muac_status or 'N/A'})")

        if haz_status:
            lines.append(f"  Stunting (HAZ): {haz_status}")
        if whz_status:
            lines.append(f"  Wasting (WHZ): {whz_status}")
        if combined_status:
            lines.append(f"  Final BMI/MUAC classification: {combined_status}")

        if not haz_status and not whz_status:
            lines.append("  Nutritional status could not be determined.")

        return "\n".join(lines)
