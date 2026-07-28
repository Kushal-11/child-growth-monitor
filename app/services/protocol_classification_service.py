"""BMI+MUAC screening protocol, separate from WHO WHZ/HAZ classification."""
from dataclasses import dataclass
from typing import Optional

from config import BMI_MAM_MAX_BY_SEX, BMI_SAM_MAX_BY_SEX


@dataclass(frozen=True)
class ProtocolClassification:
    bmi_value: Optional[float]
    bmi_status: str
    final_status: str
    triggered_indicators: list[str]


class ProtocolClassificationService:
    """Apply the supplied sex-specific BMI boundaries and combine with MUAC."""

    @staticmethod
    def classify_bmi(weight_kg: Optional[float], height_cm: Optional[float], sex: str) -> tuple[Optional[float], str]:
        sex = sex.upper()
        if weight_kg is None or height_cm is None or weight_kg <= 0 or height_cm <= 0:
            return None, "Insufficient data"
        if sex not in BMI_SAM_MAX_BY_SEX:
            return None, "Unknown"
        bmi = weight_kg / ((height_cm / 100.0) ** 2)
        # The named boundary belongs to the next category: 13.0/12.8 is MAM,
        # and 13.7/13.5 is Normal.
        if bmi < BMI_SAM_MAX_BY_SEX[sex]:
            status = "SAM"
        elif bmi < BMI_MAM_MAX_BY_SEX[sex]:
            status = "MAM"
        else:
            status = "Normal"
        return round(bmi, 2), status

    @classmethod
    def classify(cls, weight_kg: Optional[float], height_cm: Optional[float], sex: str, muac_status: Optional[str]) -> ProtocolClassification:
        bmi, bmi_status = cls.classify_bmi(weight_kg, height_cm, sex)
        muac = "MAM" if muac_status == "At Risk (MAM)" else muac_status
        known = [("bmi", bmi_status)] if bmi_status in {"SAM", "MAM", "Normal"} else []
        if muac in {"SAM", "MAM", "Normal"}:
            known.append(("muac", muac))
        for severity in ("SAM", "MAM"):
            triggered = [name for name, status in known if status == severity]
            if triggered:
                return ProtocolClassification(bmi, bmi_status, severity, triggered)
        if known:
            return ProtocolClassification(bmi, bmi_status, "Normal", [])
        missing = "Unknown" if bmi_status == "Unknown" else "Insufficient data"
        return ProtocolClassification(bmi, bmi_status, missing, [])
