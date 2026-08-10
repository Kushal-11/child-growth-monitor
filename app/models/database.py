"""Database engine and session configuration."""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Create all tables."""
    from app.models.user import User  # noqa: F401
    from app.models.child import Child  # noqa: F401
    from app.models.visit import Visit  # noqa: F401
    from app.models.measurement import MeasurementResult  # noqa: F401
    from app.models.capture_asset import CaptureAsset  # noqa: F401
    from app.models.camera_result import CameraResult  # noqa: F401
    from app.models.measured_detail_revision import MeasuredDetailRevision  # noqa: F401

    Base.metadata.create_all(bind=engine)


# Columns added after the original schema.
# (table, column, DDL fragment: type + optional constraints/default)
_MIGRATIONS = [
    ("children", "user_id", "INTEGER"),
    ("children", "photo_path", "VARCHAR(500)"),
    ("children", "is_archived", "BOOLEAN NOT NULL DEFAULT 0"),
    ("visits", "user_id", "INTEGER"),
    ("visits", "entry_method", "VARCHAR(20) NOT NULL DEFAULT 'assessment'"),
    ("visits", "capture_state", "VARCHAR(30)"),
    ("visits", "capture_started_at", "DATETIME"),
    ("visits", "capture_completed_at", "DATETIME"),
    ("visits", "device_metadata_json", "JSON"),
    ("visits", "consent_version", "VARCHAR(50)"),
    ("visits", "consent_timestamp", "DATETIME"),
    ("visits", "consent_operator_identifier", "VARCHAR(100)"),
    ("visits", "media_deleted_at", "DATETIME"),
    ("camera_results", "estimated_muac_cm", "FLOAT"),
    ("camera_results", "muac_source", "VARCHAR(100)"),
    ("camera_results", "height_range_lower_cm", "FLOAT"),
    ("camera_results", "height_range_upper_cm", "FLOAT"),
    ("camera_results", "weight_range_lower_kg", "FLOAT"),
    ("camera_results", "weight_range_upper_kg", "FLOAT"),
    ("camera_results", "muac_range_lower_cm", "FLOAT"),
    ("camera_results", "muac_range_upper_cm", "FLOAT"),
    ("measurement_results", "effective_height_cm", "FLOAT"),
    ("measurement_results", "effective_weight_kg", "FLOAT"),
    ("measurement_results", "height_method", "VARCHAR(50)"),
    ("measurement_results", "weight_method", "VARCHAR(50)"),
    ("measurement_results", "estimation_method", "VARCHAR(50)"),
    ("measurement_results", "bmi", "FLOAT"),
    ("measurement_results", "bmi_status", "VARCHAR(50)"),
    ("measurement_results", "height_confidence", "FLOAT"),
    ("measurement_results", "weight_confidence", "FLOAT"),
    ("measurement_results", "classification_confidence", "FLOAT"),
    ("measurement_results", "ml_wasting_method", "VARCHAR(50)"),
    ("measurement_results", "muac_age_in_range", "BOOLEAN"),
    ("measurement_results", "combined_status", "VARCHAR(30)"),
    ("measurement_results", "combined_triggered_by", "VARCHAR(100)"),
    ("measurement_results", "combined_rationale", "VARCHAR(255)"),
    ("measurement_results", "combined_method", "VARCHAR(50)"),
    ("measurement_results", "combined_confidence_score", "FLOAT"),
    ("measurement_results", "combined_protocol_version", "VARCHAR(50)"),
    ("measurement_results", "muac_confidence", "FLOAT"),
    ("measurement_results", "muac_uncertainty_lower_cm", "FLOAT"),
    ("measurement_results", "muac_uncertainty_upper_cm", "FLOAT"),
    ("measurement_results", "muac_model_version", "VARCHAR(100)"),
    ("measurement_results", "muac_calibration_version", "VARCHAR(100)"),
    ("measurement_results", "muac_is_direct_measurement", "BOOLEAN"),
    ("measurement_results", "muac_requires_confirmation", "BOOLEAN"),
    ("measurement_results", "muac_referral_guidance", "TEXT"),
    ("measurement_results", "poshan_status", "VARCHAR(30)"),
    ("measurement_results", "poshan_triggered_by", "VARCHAR(100)"),
    ("measurement_results", "classification_method", "VARCHAR(50)"),
    ("measurement_results", "classification_rationale", "TEXT"),
    ("measurement_results", "poshan_complete", "BOOLEAN"),
    ("measurement_results", "measurement_mode", "VARCHAR(30)"),
    ("measurement_results", "oedema", "VARCHAR(20)"),
    ("measurement_results", "measured_at", "DATETIME"),
    ("measurement_results", "editor_user_id", "INTEGER"),
    ("measurement_results", "measured_notes", "TEXT"),
    ("measurement_results", "who_acute_status", "VARCHAR(30)"),
    ("measurement_results", "who_acute_triggered_by", "TEXT"),
    ("measurement_results", "who_acute_rationale", "TEXT"),
]

# Indexes mirroring the index=True model columns, for the migration path.
_INDEXES = [
    ("ix_children_user_id", "children", "user_id", False),
    ("ix_visits_user_id", "visits", "user_id", False),
    (
        "ix_visits_owner_local_uuid",
        "visits",
        "user_id, local_uuid",
        True,
    ),
    (
        "ix_capture_assets_visit_role",
        "capture_assets",
        "visit_id, role",
        False,
    ),
    (
        "ix_camera_results_visit_version",
        "camera_results",
        "visit_id, version",
        False,
    ),
    (
        "ix_measured_revisions_visit_revision",
        "measured_detail_revisions",
        "visit_id, revision_number",
        False,
    ),
]


def run_migrations(target_engine=None):
    """Idempotently add columns missing from existing tables (SQLite, no Alembic)."""
    from app.models.camera_result import CameraResult
    from app.models.capture_asset import CaptureAsset
    from app.models.measured_detail_revision import MeasuredDetailRevision

    eng = target_engine or engine
    insp = inspect(eng)
    existing_tables = set(insp.get_table_names())
    with eng.begin() as conn:
        for table, column, ddl in _MIGRATIONS:
            if table not in existing_tables:
                continue
            cols = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            if column in cols:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))

        if "visits" in existing_tables:
            # Legacy manual reports are measured; image/assessment results stay
            # estimated. Visits without a result remain incomplete.
            has_measurements = "measurement_results" in existing_tables
            if has_measurements:
                conn.execute(
                    text(
                        "UPDATE visits SET capture_state = CASE "
                        "WHEN EXISTS ("
                        "SELECT 1 FROM measurement_results mr "
                        "WHERE mr.visit_id = visits.id AND ("
                        "mr.manual_height_cm IS NOT NULL OR "
                        "mr.manual_weight_kg IS NOT NULL OR "
                        "LOWER(COALESCE(mr.muac_method, '')) IN ('manual', 'tape')"
                        ")) THEN 'measured_report' "
                        "WHEN EXISTS ("
                        "SELECT 1 FROM measurement_results mr "
                        "WHERE mr.visit_id = visits.id"
                        ") THEN 'estimated_report' "
                        "ELSE 'incomplete_capture' END "
                        "WHERE capture_state IS NULL"
                    )
                )
            else:
                conn.execute(
                    text(
                        "UPDATE visits SET capture_state = 'incomplete_capture' "
                        "WHERE capture_state IS NULL"
                    )
                )

    if "visits" in existing_tables:
        Base.metadata.create_all(
            bind=eng,
            tables=[
                CaptureAsset.__table__,
                CameraResult.__table__,
                MeasuredDetailRevision.__table__,
            ],
        )

    existing_tables = set(inspect(eng).get_table_names())
    with eng.begin() as conn:
        for index_name, table, column, unique in _INDEXES:
            if table not in existing_tables:
                continue
            uniqueness = "UNIQUE " if unique else ""
            conn.execute(text(
                f"CREATE {uniqueness}INDEX IF NOT EXISTS "
                f"{index_name} ON {table} ({column})"
            ))
