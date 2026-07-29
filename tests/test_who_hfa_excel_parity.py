"""Authoritative WHO HFA workbook and checksum contract tests."""

import hashlib
import json

import pytest

from app.services.nutrition_service import NutritionService
from app.services.who_data_service import WHODataError, WHODataService
from config import DATA_DIR, WHO_DATA_FILES, WHO_REFERENCE_MANIFEST_PATH


EXPECTED_LMS = {
    ("F", 0): (1.0, 49.1477, 0.03790),
    ("M", 0): (1.0, 49.8842, 0.03795),
    ("F", 24): (1.0, 85.7153, 0.03764),
    ("M", 24): (1.0, 87.1161, 0.03507),
    ("F", 60): (1.0, 109.4233, 0.04347),
    ("M", 60): (1.0, 109.9638, 0.04214),
}


def test_manifest_hashes_every_authoritative_hfa_workbook():
    manifest = json.loads(WHO_REFERENCE_MANIFEST_PATH.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source_page"].startswith("https://www.who.int/")

    for key in (
        "lhfa_boys_0_2",
        "lhfa_boys_2_5",
        "lhfa_girls_0_2",
        "lhfa_girls_2_5",
    ):
        path = WHO_DATA_FILES[key]
        record = manifest["files"][path.name]
        payload = path.read_bytes()
        assert path.parent == DATA_DIR
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert record["source_url"].startswith("https://cdn.who.int/")


@pytest.mark.parametrize(("sex", "age_months"), EXPECTED_LMS)
def test_python_hfa_lms_matches_official_workbooks(sex, age_months):
    service = WHODataService()
    service.load_all()

    assert service.get_haz_lms(sex, age_months) == pytest.approx(
        EXPECTED_LMS[(sex, age_months)],
        abs=1e-6,
    )


@pytest.mark.parametrize(("sex", "age_months"), EXPECTED_LMS)
def test_haz_uses_lms_formula_at_boundaries(sex, age_months):
    service = WHODataService()
    service.load_all()
    l_value, median, s_value = EXPECTED_LMS[(sex, age_months)]
    measurement_at_minus_two = median * (
        (1 + l_value * s_value * -2.0) ** (1 / l_value)
    )

    assert NutritionService(service).compute_haz(
        sex,
        age_months,
        measurement_at_minus_two,
    ) == pytest.approx(-2.0, abs=1e-9)


def test_checksum_mismatch_fails_closed(tmp_path):
    workbook = tmp_path / "who_lhfa_boys_0_2.xlsx"
    workbook.write_bytes(b"not the official workbook")
    record = {"size_bytes": workbook.stat().st_size, "sha256": "0" * 64}

    with pytest.raises(WHODataError, match="checksum"):
        WHODataService.verify_reference_file(workbook, record)
