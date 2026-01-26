"""Database engine and session configuration."""
from sqlalchemy import create_engine
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
    finally:
        db.close()


def init_db():
    """Create all tables."""
    from app.models.child import Child  # noqa: F401
    from app.models.visit import Visit  # noqa: F401
    from app.models.measurement import MeasurementResult  # noqa: F401

    Base.metadata.create_all(bind=engine)
