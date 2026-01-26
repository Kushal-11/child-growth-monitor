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
    MeasurementDetail,
    NutritionDetail,
)
from app.services.measurement_service import MeasurementOutput, MeasurementService
from app.services.nutrition_service import NutritionService
from app.services.who_data_service import WHODataService


class AssessmentService:
    def __init__(self, who_data: WHODataService):
        self.measurement_svc = MeasurementService()
        self.nutrition_svc = NutritionService(who_data)
        self.who_data = who_data

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
        # Priority: image-based prediction > manual input
        effective_height = meas.predicted_height_cm
        if effective_height is None and height_cm is not None:
            effective_height = height_cm

        # 4. Determine effective weight
        # Priority: manual weight > WHO median estimate for effective height
        # Apply body build adjustment if available
        effective_weight = weight_kg
        estimated_weight = None
        if effective_weight is None and effective_height is not None:
            estimated_weight = self.who_data.get_median_weight_for_height(
                sex, effective_height, age_months=age_months
            )
            if estimated_weight is not None:
                # Adjust weight based on body build (slender/average/stocky)
                weight_adjustment = getattr(meas, 'weight_adjustment', 1.0)
                estimated_weight = round(estimated_weight * weight_adjustment, 2)
            effective_weight = estimated_weight

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
            confidence_score=meas.confidence_score,
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
        )

        # Extract body build from measurement result
        body_build_str = None
        if meas.body_build and isinstance(meas.body_build, dict):
            body_build_str = meas.body_build.get("body_build")
        
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
                estimation_method=meas.estimation_method,
                body_build=body_build_str,
            ),
            nutrition=NutritionDetail(
                haz_zscore=haz_z,
                whz_zscore=whz_z,
                haz_status=haz_status,
                whz_status=whz_status,
                age_months=round(age_months, 1),
            ),
            summary=summary,
        )

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
        weight, haz_status, whz_status, ref_detected
    ) -> str:
        """Build a human-readable summary string."""
        lines = [f"Assessment for {name} ({age_months:.1f} months old):"]

        if effective_height is not None:
            if predicted_height is not None:
                lines.append(f"  Height: {effective_height:.1f} cm (from image)")
            elif manual_height is not None:
                lines.append(f"  Height: {effective_height:.1f} cm (manual input)")
        else:
            lines.append("  Height: Could not be determined.")
            if not ref_detected:
                lines.append(
                    "  Note: Image-based height estimation was not successful. "
                    "Provide manual height for classification."
                )

        if weight is not None:
            lines.append(f"  Weight: {weight:.1f} kg")

        if haz_status:
            lines.append(f"  Height-for-Age: {haz_status}")
        if whz_status:
            lines.append(f"  Weight-for-Height: {whz_status}")

        if not haz_status and not whz_status:
            lines.append("  Nutritional status could not be determined.")

        return "\n".join(lines)
