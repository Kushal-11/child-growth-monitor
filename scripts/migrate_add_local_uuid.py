"""One-shot migration: add local_uuid TEXT UNIQUE column to visits table.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/migrate_add_local_uuid.py
"""
import sqlite3
from pathlib import Path

from config import DATABASE_URL


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
        cols = {row[1] for row in cur.fetchall()}
        if "local_uuid" in cols:
            print("local_uuid already exists; nothing to do.")
            return
        cur.execute("ALTER TABLE visits ADD COLUMN local_uuid TEXT")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_visits_local_uuid "
            "ON visits(local_uuid) WHERE local_uuid IS NOT NULL"
        )
        conn.commit()
        print("Added local_uuid column and unique index.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
