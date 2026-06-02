"""Admin web UI: session-cookie login + user management."""
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from app.services import auth_service

router = APIRouter(prefix="/admin", tags=["Admin"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _current_admin(request: Request, db: Session):
    user_id = request.session.get("admin_user_id")
    if not user_id:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active or user.role != "admin":
        return None
    return user


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(request, "admin_login.html", {"error": None})


@router.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if (user is None or not user.is_active or user.role != "admin"
            or not auth_service.verify_password(password, user.hashed_password)):
        return templates.TemplateResponse(
            request, "admin_login.html",
            {"error": "Invalid admin credentials"}, status_code=200,
        )
    request.session["admin_user_id"] = user.id
    return RedirectResponse("/admin/users", status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.pop("admin_user_id", None)
    return RedirectResponse("/admin/login", status_code=303)


@router.get("/users", response_class=HTMLResponse)
def list_users(request: Request, db: Session = Depends(get_db)):
    admin = _current_admin(request, db)
    if admin is None:
        return RedirectResponse("/admin/login", status_code=303)
    users = db.query(User).order_by(User.username).all()
    return templates.TemplateResponse(
        request, "admin_users.html", {"users": users, "admin": admin, "error": None},
    )


@router.post("/users/create")
def create_user_web(
    request: Request,
    username: str = Form(...),
    full_name: str = Form(...),
    password: str = Form(...),
    role: str = Form("worker"),
    db: Session = Depends(get_db),
):
    admin = _current_admin(request, db)
    if admin is None:
        return RedirectResponse("/admin/login", status_code=303)
    error = None
    if db.query(User).filter(User.username == username).first() is not None:
        error = f"Username '{username}' already exists"
    elif role not in ("worker", "admin"):
        error = "Invalid role"
    else:
        db.add(User(
            username=username, full_name=full_name, role=role,
            hashed_password=auth_service.hash_password(password),
        ))
        db.commit()
    users = db.query(User).order_by(User.username).all()
    return templates.TemplateResponse(
        request, "admin_users.html", {"users": users, "admin": admin, "error": error},
    )


@router.post("/users/{user_id}/toggle")
def toggle_user(request: Request, user_id: int, db: Session = Depends(get_db)):
    admin = _current_admin(request, db)
    if admin is None:
        return RedirectResponse("/admin/login", status_code=303)
    user = db.query(User).filter(User.id == user_id).first()
    if user is not None and user.id != admin.id:
        user.is_active = not user.is_active
        db.commit()
    return RedirectResponse("/admin/users", status_code=303)
