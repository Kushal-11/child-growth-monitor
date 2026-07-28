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


def test_migration_adds_poshan_setu_columns_to_existing_measurements():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE measurement_results (id INTEGER PRIMARY KEY)")
        )
    dbmod.run_migrations(engine)
    columns = {
        column["name"]
        for column in inspect(engine).get_columns("measurement_results")
    }
    assert {
        "bmi",
        "bmi_status",
        "poshan_status",
        "poshan_triggered_by",
        "classification_method",
        "classification_rationale",
        "poshan_complete",
    } <= columns


def test_migration_matches_create_all_for_new_columns_and_indexes():
    from sqlalchemy import create_engine, inspect
    from app.models.database import Base, run_migrations
    # ensure all models are registered
    import app.models  # noqa: F401

    # Fresh DB built from models
    fresh = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=fresh)
    fresh_insp = inspect(fresh)

    # Legacy DB upgraded via migration
    legacy = _legacy_engine()
    run_migrations(legacy)
    legacy_insp = inspect(legacy)

    for table in ("children", "visits"):
        fresh_cols = {c["name"] for c in fresh_insp.get_columns(table)}
        legacy_cols = {c["name"] for c in legacy_insp.get_columns(table)}
        # every column the fresh schema has on these tables must exist after migration
        # (legacy may differ in unrelated original columns, so check the NEW ones explicitly)
        for new_col in ("user_id",):
            assert new_col in fresh_cols and new_col in legacy_cols
        # index parity for user_id
        fresh_idx = {i["name"] for i in fresh_insp.get_indexes(table)}
        legacy_idx = {i["name"] for i in legacy_insp.get_indexes(table)}
        # the user_id index from create_all must also be present after migration
        user_id_indexes = {n for n in fresh_idx if "user_id" in n}
        assert user_id_indexes, f"expected a user_id index on {table} in fresh schema"
        assert user_id_indexes <= legacy_idx, (
            f"migration missing index(es) {user_id_indexes - legacy_idx} on {table}")
