"""One-shot migration: add side_image_path and back_image_path to visits.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/migrate_add_visit_image_paths.py
"""
import sqlite3
from pathlib import Path

from config import DATABASE_URL

NEW_COLUMNS = [
    ("side_image_path", "TEXT"),
    ("back_image_path", "TEXT"),
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
        cur.execute("PRAGMA table_info(visits)")
        existing = {row[1] for row in cur.fetchall()}
        added = []
        for col_name, col_type in NEW_COLUMNS:
            if col_name in existing:
                continue
            cur.execute(
                f"ALTER TABLE visits ADD COLUMN {col_name} {col_type}"
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
