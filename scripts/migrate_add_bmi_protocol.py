"""Add BMI+MUAC protocol fields to an existing SQLite database."""
import sqlite3
from pathlib import Path
from config import DATABASE_URL

NEW_COLUMNS = [("bmi_value", "REAL"), ("bmi_status", "TEXT"),
               ("protocol_status", "TEXT"), ("triggered_indicators", "TEXT"),
               ("measurement_methods", "TEXT")]

def main() -> None:
    path = Path(DATABASE_URL.removeprefix("sqlite:///"))
    if not path.exists():
        return
    with sqlite3.connect(path) as connection:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(measurement_results)")}
        for name, kind in NEW_COLUMNS:
            if name not in existing:
                connection.execute(f"ALTER TABLE measurement_results ADD COLUMN {name} {kind}")

if __name__ == "__main__":
    main()
