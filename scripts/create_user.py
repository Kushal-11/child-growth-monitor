"""CLI: create a health-worker or admin account.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/create_user.py \
      --username admin --full-name "Site Admin" --role admin
(prompts for password)
"""
import argparse
import getpass

from sqlalchemy.orm import Session

from app.models.database import SessionLocal, init_db, run_migrations
from app.models.user import User
from app.services import auth_service


def create_user(db: Session, *, username: str, full_name: str, password: str, role: str) -> User:
    if db.query(User).filter(User.username == username).first() is not None:
        raise ValueError(f"User '{username}' already exists")
    user = User(
        username=username,
        full_name=full_name,
        hashed_password=auth_service.hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CGM user account")
    parser.add_argument("--username", required=True)
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--role", default="worker", choices=["worker", "admin"])
    args = parser.parse_args()

    init_db()
    run_migrations()
    password = getpass.getpass("Password: ")
    if not password:
        raise SystemExit("Password cannot be empty")

    db = SessionLocal()
    try:
        user = create_user(
            db, username=args.username, full_name=args.full_name,
            password=password, role=args.role,
        )
        print(f"Created {user.role} '{user.username}' (id={user.id})")
    except ValueError as exc:
        raise SystemExit(str(exc))
    finally:
        db.close()


if __name__ == "__main__":
    main()
