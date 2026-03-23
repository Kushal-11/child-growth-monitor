"""
Web UI route handlers using FastAPI + Jinja2 templates.

Pages:
  GET  /              - Landing page with upload form
  POST /assess        - Form submission handler
  GET  /children      - Child listing page
  GET  /children/{id} - Child detail/history page
"""
import shutil
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.database import get_db
from app.services.assessment_service import AssessmentService
from config import UPLOAD_DIR


def parse_date_input(date_str: str) -> date:
    """Parse date from yyyy-mm-dd format (HTML5 date input)."""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected yyyy-mm-dd")

router = APIRouter(tags=["Web UI"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def get_assessment_service() -> AssessmentService:
    """Placeholder; overridden at app startup in main.py."""
    raise NotImplementedError


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.post("/assess", response_class=HTMLResponse)
async def web_assess(
    request: Request,
    image: UploadFile = File(...),
    image_side: Optional[UploadFile] = File(None),
    image_back: Optional[UploadFile] = File(None),
    child_name: str = Form(...),
    date_of_birth: str = Form(...),
    sex: str = Form(...),
    weight_kg: float = Form(None),
    height_cm: float = Form(None),
    muac_cm: float = Form(None),
    guardian_name: str = Form(None),
    location: str = Form(None),
    db: Session = Depends(get_db),
    svc: AssessmentService = Depends(get_assessment_service),
):
    """Handle form submission, run assessment, show results."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Read side image bytes (optional)
    side_image_bytes = None
    if image_side is not None:
        side_image_bytes = await image_side.read()
        if not side_image_bytes:
            side_image_bytes = None

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "Invalid date format. Please use the date picker."},
        )

    try:
        result = svc.assess(
            db=db,
            image_path=str(file_path),
            child_name=child_name,
            dob=dob,
            sex=sex,
            weight_kg=weight_kg,
            height_cm=height_cm,
            muac_cm=muac_cm,
            guardian_name=guardian_name,
            location=location,
            side_image=side_image_bytes,
        )
        error = None
    except Exception as e:
        result = None
        error = str(e)

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": result,
            "error": error,
            "image_filename": filename,
        },
    )


@router.get("/children", response_class=HTMLResponse)
async def children_list(request: Request, db: Session = Depends(get_db)):
    children = db.query(Child).order_by(Child.name).all()
    return templates.TemplateResponse(
        request, "children.html", {"children": children}
    )


@router.get("/children/{child_id}", response_class=HTMLResponse)
async def child_detail(
    request: Request, child_id: int, db: Session = Depends(get_db)
):
    child = db.query(Child).filter(Child.id == child_id).first()
    return templates.TemplateResponse(
        request, "child_detail.html", {"child": child}
    )
