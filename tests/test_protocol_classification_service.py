import pytest

from app.services.protocol_classification_service import ProtocolClassificationService


@pytest.mark.parametrize("sex,bmi,expected", [
    ("M", 12.99, "SAM"), ("M", 13.0, "MAM"), ("M", 13.69, "MAM"), ("M", 13.7, "Normal"),
    ("F", 12.79, "SAM"), ("F", 12.8, "MAM"), ("F", 13.49, "MAM"), ("F", 13.5, "Normal"),
])
def test_bmi_boundaries(sex, bmi, expected):
    value, status = ProtocolClassificationService.classify_bmi(bmi, 100, sex)
    assert value == pytest.approx(bmi)
    assert status == expected


def test_missing_measurements_are_not_normal():
    result = ProtocolClassificationService.classify(None, None, "M", None)
    assert result.final_status == "Insufficient data"
    assert result.bmi_status == "Insufficient data"


@pytest.mark.parametrize("bmi_status,muac,final,trigger", [
    (12.0, "Normal", "SAM", ["bmi"]),
    (14.0, "SAM", "SAM", ["muac"]),
    (13.2, "Normal", "MAM", ["bmi"]),
    (14.0, "At Risk (MAM)", "MAM", ["muac"]),
])
def test_severity_ordering_with_contradictory_indicators(bmi_status, muac, final, trigger):
    result = ProtocolClassificationService.classify(bmi_status, 100, "M", muac)
    assert result.final_status == final
    assert result.triggered_indicators == trigger
