"""Validated access to the authoritative WHO Excel LMS reference tables."""
import base64
import binascii
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import WHO_DATA_FILES, WHO_REFERENCE_MANIFEST_PATH


class WHODataError(RuntimeError):
    """Raised when an authoritative WHO workbook cannot be used safely."""


class WHODataService:
    """Load and query WHO LMS workbooks without clinical CSV fallbacks."""

    _LMS_COLUMNS = {"L", "M", "S"}

    def __init__(self):
        self._haz_lms: Optional[pd.DataFrame] = None
        self._wfl_lms: Optional[pd.DataFrame] = None
        self._wfh_lms: Optional[pd.DataFrame] = None
        self._loaded = False

    def load_all(self):
        """Load and validate every authoritative reference at startup."""
        self._haz_lms = self._load_haz_lms()
        self._wfl_lms = self._load_size_lms("wfl", ("wfl_boys_0_2", "wfl_girls_0_2"))
        self._wfh_lms = self._load_size_lms("wfh", ("wfh_boys_2_5", "wfh_girls_2_5"))
        self._loaded = True

    @staticmethod
    def verify_reference_file(path: Path, record: dict) -> None:
        """Fail closed unless a reference file matches its pinned manifest."""
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise WHODataError(
                f"Authoritative WHO workbook unavailable: {path}: {exc}"
            ) from exc

        expected_size = record.get("size_bytes")
        expected_checksum = record.get("sha256")
        if not isinstance(expected_size, int) or not isinstance(
            expected_checksum, str
        ):
            raise WHODataError(
                f"WHO reference manifest entry for {path.name} is malformed"
            )
        if len(payload) != expected_size:
            raise WHODataError(
                f"WHO reference size mismatch for {path.name}: "
                f"expected {expected_size}, got {len(payload)}"
            )
        actual_checksum = hashlib.sha256(payload).hexdigest()
        if actual_checksum != expected_checksum:
            raise WHODataError(
                f"WHO reference checksum mismatch for {path.name}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )

    @staticmethod
    def _load_reference_manifest() -> dict:
        try:
            manifest = json.loads(WHO_REFERENCE_MANIFEST_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise WHODataError(
                "Authoritative WHO reference manifest unavailable or "
                f"unreadable: {WHO_REFERENCE_MANIFEST_PATH}: {exc}"
            ) from exc
        if manifest.get("schema_version") != 1 or not isinstance(
            manifest.get("files"), dict
        ):
            raise WHODataError(
                f"WHO reference manifest is malformed: {WHO_REFERENCE_MANIFEST_PATH}"
            )
        return manifest

    @classmethod
    def _read_excel(cls, path: Path) -> pd.DataFrame:
        try:
            if path.suffix == ".b64":
                # GitHub's PR API rejects binary patches. The authoritative XLSX
                # is therefore stored losslessly as Base64 text and decoded only
                # in memory; no CSV or generated clinical values are involved.
                encoded = b"".join(path.read_bytes().split())
                workbook = BytesIO(base64.b64decode(encoded, validate=True))
                return pd.read_excel(workbook)
            return pd.read_excel(path)
        except (FileNotFoundError, ValueError, OSError, ImportError, binascii.Error) as exc:
            raise WHODataError(f"Authoritative WHO workbook unavailable or unreadable: {path}: {exc}") from exc

    @classmethod
    def _numeric_lms(cls, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        missing = cls._LMS_COLUMNS.difference(df.columns)
        if missing:
            raise WHODataError(f"WHO workbook {path} is missing required columns: {sorted(missing)}")
        for col in cls._LMS_COLUMNS:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("\u2212", "-"), errors="coerce"
            )
        invalid = df[list(cls._LMS_COLUMNS)].isna().any(axis=1) | (df["M"] <= 0) | (df["S"] <= 0)
        if invalid.any():
            raise WHODataError(f"WHO workbook {path} contains malformed L, M, or S values")
        return df

    def _load_haz_lms(self) -> pd.DataFrame:
        manifest = self._load_reference_manifest()
        workbook_specs = (
            ("lhfa_boys_0_2", "M", "length", 0, 23),
            ("lhfa_boys_2_5", "M", "height", 24, 60),
            ("lhfa_girls_0_2", "F", "length", 0, 23),
            ("lhfa_girls_2_5", "F", "height", 24, 60),
        )
        frames = []
        for key, sex, measure, minimum_age, maximum_age in workbook_specs:
            path = WHO_DATA_FILES[key]
            record = manifest["files"].get(path.name)
            if not isinstance(record, dict):
                raise WHODataError(
                    f"WHO reference manifest has no entry for {path.name}"
                )
            self.verify_reference_file(path, record)
            df = self._read_excel(path)
            df = df.rename(columns=lambda column: str(column).strip()).rename(
                columns={"Month": "age_months"}
            )
            required = {"age_months", *self._LMS_COLUMNS}
            missing = required.difference(df.columns)
            if missing:
                raise WHODataError(
                    f"WHO length/height-for-age workbook {path} is missing "
                    f"required columns: {sorted(missing)}"
                )
            df = self._numeric_lms(df, path)
            df["age_months"] = pd.to_numeric(
                df["age_months"], errors="coerce"
            )
            if df["age_months"].isna().any() or (
                df["age_months"] % 1 != 0
            ).any():
                raise WHODataError(
                    f"WHO length/height-for-age workbook {path} contains "
                    "malformed ages"
                )
            df["age_months"] = df["age_months"].astype(int)
            df = df[
                df["age_months"].between(minimum_age, maximum_age)
            ].copy()
            expected_ages = set(range(minimum_age, maximum_age + 1))
            actual_ages = set(df["age_months"])
            if actual_ages != expected_ages or df["age_months"].duplicated().any():
                raise WHODataError(
                    f"WHO length/height-for-age workbook {path} has invalid "
                    f"age coverage: expected {minimum_age}-{maximum_age}"
                )
            df["sex"] = sex
            df["measure"] = measure
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        if df.duplicated(["sex", "age_months"]).any():
            raise WHODataError(
                "WHO length/height-for-age workbooks have duplicate sex/age rows"
            )
        for sex in ("F", "M"):
            ages = set(df.loc[df["sex"] == sex, "age_months"])
            missing_ages = set(range(61)).difference(ages)
            extra_ages = ages.difference(range(61))
            if missing_ages or extra_ages:
                raise WHODataError(
                    "WHO length/height-for-age workbooks lack complete "
                    f"{sex} age coverage: missing={sorted(missing_ages)}, "
                    f"extra={sorted(extra_ages)}"
                )
        return df

    def _load_size_lms(self, label: str, keys: Tuple[str, str]) -> pd.DataFrame:
        frames = []
        for key, sex in zip(keys, ("M", "F")):
            path = WHO_DATA_FILES[key]
            df = self._read_excel(path)
            df = df.rename(columns={df.columns[0]: "index_value"})
            df = self._numeric_lms(df, path)
            df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
            if df["index_value"].isna().any() or df["index_value"].duplicated().any():
                raise WHODataError(f"WHO {label.upper()} workbook {path} has malformed measurement coverage")
            df["sex"] = sex
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_haz_lms(self, sex: str, age_months: int) -> Optional[Tuple[float, float, float]]:
        """Return length/height-for-age LMS values, or ``None`` outside 0-60 months."""
        if self._haz_lms is None:
            return None
        row = self._haz_lms[(self._haz_lms["sex"] == sex) & (self._haz_lms["age_months"] == age_months)]
        if len(row) != 1:
            return None
        value = row.iloc[0]
        return float(value["L"]), float(value["M"]), float(value["S"])

    @staticmethod
    def _measurement_at_z(L: float, M: float, S: float, z: float) -> float:
        return M * np.exp(S * z) if abs(L) < 1e-6 else M * ((1 + L * S * z) ** (1 / L))

    def get_haz_boundaries(self, sex: str, age_months: int) -> Optional[dict]:
        """Compatibility view derived only from the authoritative LMS workbook."""
        lms = self.get_haz_lms(sex, age_months)
        if lms is None:
            return None
        L, M, S = lms
        names = ["z_minus_3", "z_minus_2", "z_minus_1", "z_0", "z_plus_1", "z_plus_2", "z_plus_3"]
        result = {name: self._measurement_at_z(L, M, S, z) for name, z in zip(names, range(-3, 4))}
        result["sex"] = sex
        result["age_months"] = age_months
        result["measure"] = "length" if age_months < 24 else "height"
        return result

    def get_wfh_lms(self, sex: str, height_cm: float, age_months: float) -> Optional[Tuple[float, float, float]]:
        df = self._wfl_lms if age_months < 24 else self._wfh_lms
        if df is None:
            return None
        subset = df[df["sex"] == sex].sort_values("index_value")
        if subset.empty or height_cm < subset["index_value"].min() or height_cm > subset["index_value"].max():
            return None
        exact = subset[np.isclose(subset["index_value"], height_cm, atol=0.05)]
        if not exact.empty:
            row = exact.iloc[0]
            return float(row["L"]), float(row["M"]), float(row["S"])
        below, above = subset[subset["index_value"] < height_cm].iloc[-1], subset[subset["index_value"] > height_cm].iloc[0]
        fraction = (height_cm - below["index_value"]) / (above["index_value"] - below["index_value"])
        return tuple(float(below[c] + fraction * (above[c] - below[c])) for c in ("L", "M", "S"))

    def get_median_weight_for_height(self, sex: str, height_cm: float, age_months: float = 36.0) -> Optional[float]:
        """Return Excel LMS median, explicitly ``None`` when it is unavailable."""
        lms = self.get_wfh_lms(sex, height_cm, age_months)
        return round(lms[1], 2) if lms is not None else None

    def get_median_height_for_age(self, sex: str, age_months: int) -> Optional[float]:
        lms = self.get_haz_lms(sex, age_months)
        return lms[1] if lms else None

    def get_height_sd_for_age(self, sex: str, age_months: int) -> Optional[float]:
        lms = self.get_haz_lms(sex, age_months)
        if not lms:
            return None
        L, M, S = lms
        return M - self._measurement_at_z(L, M, S, -1)

    def get_height_range_for_age(self, sex: str, age_months: int, num_sd: float = 3.0) -> Optional[Tuple[float, float]]:
        lms = self.get_haz_lms(sex, age_months)
        if not lms:
            return None
        L, M, S = lms
        return self._measurement_at_z(L, M, S, -num_sd), self._measurement_at_z(L, M, S, num_sd)
