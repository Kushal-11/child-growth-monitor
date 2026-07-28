"""Tests for WHO data loading and lookup."""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from config import WHO_DATA_FILES
from app.services.who_data_service import WHODataService
from app.services.nutrition_service import NutritionService


@pytest.fixture(scope="module")
def who_data():
    svc = WHODataService()
    svc.load_all()
    return svc


class TestHAZBoundaries:
    def test_female_0_months(self, who_data):
        b = who_data.get_haz_boundaries("F", 0)
        assert b is not None
        assert b["z_0"] == pytest.approx(49.1, abs=0.1)

    def test_male_24_months(self, who_data):
        b = who_data.get_haz_boundaries("M", 24)
        assert b is not None
        # At 24 months, measure should be 'height'
        assert b["measure"] == "height"

    def test_out_of_range(self, who_data):
        b = who_data.get_haz_boundaries("M", 100)
        assert b is None

    def test_boundaries_are_ordered(self, who_data):
        b = who_data.get_haz_boundaries("M", 12)
        assert b is not None
        assert b["z_minus_3"] < b["z_minus_2"] < b["z_minus_1"] < b["z_0"]
        assert b["z_0"] < b["z_plus_1"] < b["z_plus_2"] < b["z_plus_3"]

    @pytest.mark.parametrize("sex", ["F", "M"])
    @pytest.mark.parametrize("age_months", [0, 24, 60])
    def test_authoritative_lms_age_and_sex_boundaries(self, who_data, sex, age_months):
        lms = who_data.get_haz_lms(sex, age_months)
        assert lms is not None
        L, M, S = lms
        assert M > 0
        assert S > 0
        assert NutritionService(who_data).compute_haz(sex, age_months, M) == pytest.approx(0)

    @pytest.mark.parametrize("sex", ["F", "M"])
    def test_24_month_transition_selects_standing_height_row(
        self, who_data, sex
    ):
        rows = who_data._haz_lms[
            (who_data._haz_lms["sex"] == sex)
            & (who_data._haz_lms["age_months"] == 24)
        ]
        assert len(rows) == 1
        assert rows.iloc[0]["measure"] == "height"

    def test_age_above_authoritative_boundary_is_unavailable(self, who_data):
        assert who_data.get_haz_lms("F", 61) is None


class TestWFHLMS:
    def test_female_wfl_under_2(self, who_data):
        """Test Weight-for-Length lookup for infant (age < 24 months)."""
        result = who_data.get_wfh_lms("F", 65.0, age_months=12)
        assert result is not None
        L, M, S = result
        assert M > 0
        assert S > 0

    def test_male_wfh_over_2(self, who_data):
        """Test Weight-for-Height lookup for child (age >= 24 months)."""
        result = who_data.get_wfh_lms("M", 85.0, age_months=30)
        assert result is not None
        L, M, S = result
        assert M > 0
        assert S > 0

    def test_interpolation(self, who_data):
        """Test that non-exact heights are interpolated."""
        result = who_data.get_wfh_lms("M", 85.3, age_months=30)
        assert result is not None
        L, M, S = result
        assert M > 0

    def test_out_of_range(self, who_data):
        """Height outside the reference range should return None."""
        result = who_data.get_wfh_lms("M", 200.0, age_months=30)
        assert result is None


class TestMedianWeight:
    def test_male_60cm(self, who_data):
        # WHO WFL boys: at length 60 cm, SD0 (median) ≈ 6.0 kg
        # Source: wfl_boys_0-to-2-years_zscores.xlsx (verified)
        # Previous test used wrong value from the now-fixed who_whz_reference.csv
        w = who_data.get_median_weight_for_height("M", 60.0, age_months=6.0)
        assert w is not None
        assert w == pytest.approx(6.0, abs=0.4)

    def test_out_of_range(self, who_data):
        w = who_data.get_median_weight_for_height("F", 200.0)
        assert w is None

    def test_below_table_range_has_no_csv_fallback(self, who_data):
        assert who_data.get_median_weight_for_height("M", 20, age_months=6) is None


def test_clinical_loading_never_reads_csv(monkeypatch):
    def fail_read_csv(*args, **kwargs):
        raise AssertionError("CSV must not participate in clinical WHO calculations")

    monkeypatch.setattr(pd, "read_csv", fail_read_csv)
    service = WHODataService()
    service.load_all()
    assert service.get_haz_lms("M", 24) is not None
    assert service.get_median_weight_for_height("F", 85, 30) is not None


def test_lhfa_workbook_is_text_packaged_for_pull_requests():
    path = WHO_DATA_FILES["lhfa_0_5"]
    assert path.suffix == ".b64"
    assert path.read_bytes().isascii()


class TestMedianHeight:
    """Tests for the new get_median_height_for_age method."""
    
    def test_female_0_months(self, who_data):
        """Newborn female median height should be ~49 cm."""
        h = who_data.get_median_height_for_age("F", 0)
        assert h is not None
        assert h == pytest.approx(49.1, abs=0.5)
    
    def test_male_12_months(self, who_data):
        """12-month male median length should be ~76 cm."""
        h = who_data.get_median_height_for_age("M", 12)
        assert h is not None
        assert 74 < h < 78
    
    def test_male_24_months(self, who_data):
        """24-month male median height should be ~87 cm."""
        h = who_data.get_median_height_for_age("M", 24)
        assert h is not None
        assert 85 < h < 90
    
    def test_female_48_months(self, who_data):
        """48-month female median height should be ~101 cm."""
        h = who_data.get_median_height_for_age("F", 48)
        assert h is not None
        assert 99 < h < 105
    
    def test_out_of_range(self, who_data):
        """Age outside 0-59 months should return None."""
        h = who_data.get_median_height_for_age("M", 100)
        assert h is None


class TestHeightSD:
    """Tests for the get_height_sd_for_age method."""
    
    def test_sd_positive(self, who_data):
        """SD should be a positive number."""
        sd = who_data.get_height_sd_for_age("M", 24)
        assert sd is not None
        assert sd > 0
    
    def test_sd_reasonable_range(self, who_data):
        """SD for children should be roughly 2-4 cm."""
        sd = who_data.get_height_sd_for_age("F", 12)
        assert sd is not None
        assert 1.5 < sd < 4.0


class TestHeightRange:
    """Tests for the get_height_range_for_age method."""
    
    def test_range_3sd(self, who_data):
        """Test ±3 SD range."""
        r = who_data.get_height_range_for_age("M", 24, 3.0)
        assert r is not None
        min_h, max_h = r
        assert min_h < max_h
        # 24-month male should have range roughly 78-96 cm
        assert 75 < min_h < 85
        assert 92 < max_h < 100
    
    def test_range_bounds_median(self, who_data):
        """Median should fall within the range."""
        r = who_data.get_height_range_for_age("F", 12, 3.0)
        median = who_data.get_median_height_for_age("F", 12)
        assert r is not None
        assert median is not None
        assert r[0] < median < r[1]
