"""Non-destructive guided-capture SQLite migration tests."""

from sqlalchemy import create_engine, inspect, text

from app.models.database import run_migrations


def _current_schema_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE children ("
                "id INTEGER PRIMARY KEY, name VARCHAR, date_of_birth DATE, "
                "sex VARCHAR, user_id INTEGER)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE visits ("
                "id INTEGER PRIMARY KEY, child_id INTEGER, visit_date DATETIME, "
                "age_months FLOAT, local_uuid VARCHAR, user_id INTEGER, "
                "entry_method VARCHAR)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE measurement_results ("
                "id INTEGER PRIMARY KEY, visit_id INTEGER, "
                "manual_height_cm FLOAT, manual_weight_kg FLOAT, "
                "muac_method VARCHAR)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO users (id, username) VALUES (7, 'asha')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO children "
                "(id, name, date_of_birth, sex, user_id) "
                "VALUES (11, 'Child 011', '2024-01-01', 'F', 7)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO visits "
                "(id, child_id, visit_date, age_months, local_uuid, user_id, "
                "entry_method) VALUES "
                "(21, 11, '2026-07-29', 30, 'measured-visit', 7, 'manual'), "
                "(22, 11, '2026-07-29', 30, 'estimated-visit', 7, 'assessment'), "
                "(23, 11, '2026-07-29', 30, 'empty-visit', 7, 'assessment')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO measurement_results "
                "(id, visit_id, manual_height_cm, manual_weight_kg, muac_method) "
                "VALUES "
                "(31, 21, 88, 12, 'manual'), "
                "(32, 22, NULL, NULL, 'estimated_from_whz')"
            )
        )
    return engine


def test_migration_adds_guided_columns_tables_and_indexes_without_data_loss():
    engine = _current_schema_engine()
    run_migrations(engine)
    inspector = inspect(engine)

    visit_columns = {column["name"] for column in inspector.get_columns("visits")}
    measurement_columns = {
        column["name"]
        for column in inspector.get_columns("measurement_results")
    }
    assert {
        "capture_state",
        "capture_started_at",
        "capture_completed_at",
        "device_metadata_json",
        "consent_version",
        "consent_timestamp",
        "consent_operator_identifier",
        "media_deleted_at",
    } <= visit_columns
    assert {
        "measurement_mode",
        "oedema",
        "measured_at",
        "editor_user_id",
        "measured_notes",
        "who_acute_status",
        "who_acute_triggered_by",
        "who_acute_rationale",
    } <= measurement_columns
    assert {
        "capture_assets",
        "camera_results",
        "measured_detail_revisions",
    } <= set(inspector.get_table_names())

    index_names = {
        index["name"]
        for table in (
            "visits",
            "capture_assets",
            "camera_results",
            "measured_detail_revisions",
        )
        for index in inspector.get_indexes(table)
    }
    assert {
        "ix_visits_owner_local_uuid",
        "ix_capture_assets_visit_role",
        "ix_camera_results_visit_version",
        "ix_measured_revisions_visit_revision",
    } <= index_names

    with engine.connect() as connection:
        child_name = connection.scalar(
            text("SELECT name FROM children WHERE id = 11")
        )
        states = dict(
            connection.execute(
                text("SELECT id, capture_state FROM visits ORDER BY id")
            ).all()
        )
        measured_height = connection.scalar(
            text(
                "SELECT manual_height_cm FROM measurement_results "
                "WHERE visit_id = 21"
            )
        )
    assert child_name == "Child 011"
    assert measured_height == 88
    assert states == {
        21: "measured_report",
        22: "estimated_report",
        23: "incomplete_capture",
    }


def test_guided_capture_migration_is_idempotent():
    engine = _current_schema_engine()
    run_migrations(engine)
    run_migrations(engine)

    assert {
        "capture_assets",
        "camera_results",
        "measured_detail_revisions",
    } <= set(inspect(engine).get_table_names())
