"""Idempotent ALTER TABLE migration tests."""
from sqlalchemy import create_engine, inspect, text

from app.models import database as dbmod


def _legacy_engine():
    """Create an engine with the OLD children/visits schema (no new columns)."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE children (id INTEGER PRIMARY KEY, name VARCHAR, "
            "date_of_birth DATE, sex VARCHAR, guardian_name VARCHAR, location VARCHAR, "
            "created_at DATETIME, updated_at DATETIME)"
        ))
        conn.execute(text(
            "CREATE TABLE visits (id INTEGER PRIMARY KEY, child_id INTEGER, "
            "visit_date DATETIME, age_months FLOAT, image_path VARCHAR, "
            "side_image_path VARCHAR, back_image_path VARCHAR, notes TEXT, local_uuid VARCHAR)"
        ))
    return engine


def test_migration_adds_missing_columns():
    engine = _legacy_engine()
    dbmod.run_migrations(engine)
    insp = inspect(engine)
    child_cols = {c["name"] for c in insp.get_columns("children")}
    visit_cols = {c["name"] for c in insp.get_columns("visits")}
    assert {"user_id", "photo_path", "is_archived"} <= child_cols
    assert {"user_id", "entry_method"} <= visit_cols


def test_migration_is_idempotent():
    engine = _legacy_engine()
    dbmod.run_migrations(engine)
    dbmod.run_migrations(engine)  # second run must not raise
    insp = inspect(engine)
    assert "user_id" in {c["name"] for c in insp.get_columns("children")}


def test_migration_noop_when_table_absent():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    dbmod.run_migrations(engine)  # no tables yet — must not raise
