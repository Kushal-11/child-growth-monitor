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

    Base.metadata.create_all(bind=engine)


# Columns added after the original schema. (table, column, DDL type with default)
_MIGRATIONS = [
    ("children", "user_id", "INTEGER"),
    ("children", "photo_path", "VARCHAR(500)"),
    ("children", "is_archived", "BOOLEAN NOT NULL DEFAULT 0"),
    ("visits", "user_id", "INTEGER"),
    ("visits", "entry_method", "VARCHAR(20) NOT NULL DEFAULT 'assessment'"),
]


def run_migrations(target_engine=None):
    """Idempotently add columns missing from existing tables (SQLite, no Alembic)."""
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
