"""One-shot migration: add mobile-computed columns to measurement_results table.

Adds: body_build, side_view_used, chest_depth_cm, abd_depth_cm,
ml_estimated_weight_kg, ml_wasting_status, sam_probability, mam_probability,
normal_probability, risk_probability, overweight_probability,
muac_cm, muac_status, muac_method.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/migrate_extend_measurements.py
"""
import sqlite3
from pathlib import Path

from config import DATABASE_URL

NEW_COLUMNS = [
    ("body_build", "TEXT"),
    ("side_view_used", "INTEGER DEFAULT 0"),
    ("chest_depth_cm", "REAL"),
    ("abd_depth_cm", "REAL"),
    ("ml_estimated_weight_kg", "REAL"),
    ("ml_wasting_status", "TEXT"),
    ("sam_probability", "REAL"),
    ("mam_probability", "REAL"),
    ("normal_probability", "REAL"),
    ("risk_probability", "REAL"),
    ("overweight_probability", "REAL"),
    ("muac_cm", "REAL"),
    ("muac_status", "TEXT"),
    ("muac_method", "TEXT"),
]


def main() -> None:
    if not DATABASE_URL.startswith("sqlite:///"):
        raise SystemExit(f"Only sqlite databases supported: {DATABASE_URL}")
    db_path = Path(DATABASE_URL.replace("sqlite:///", "", 1))
    if not db_path.exists():
        print(f"Database does not exist yet at {db_path}; nothing to migrate.")
        return

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(measurement_results)")
        existing = {row[1] for row in cur.fetchall()}
        added = []
        for col_name, col_type in NEW_COLUMNS:
            if col_name in existing:
                continue
            cur.execute(
                f"ALTER TABLE measurement_results ADD COLUMN {col_name} {col_type}"
            )
            added.append(col_name)
        conn.commit()
        if added:
            print(f"Added columns: {', '.join(added)}")
        else:
            print("All columns already present; nothing to do.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
