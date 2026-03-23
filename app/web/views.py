"""
Web UI route handlers using FastAPI + Jinja2 templates.

Pages:
  GET  /              - Landing page with upload form
  POST /assess        - Form submission handler
  GET  /children      - Child listing page
  GET  /children/{id} - Child detail/history page
"""
import json
import shutil
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.database import get_db
from app.services.assessment_service import AssessmentService
from app.web.translations import i18n_context
from config import UPLOAD_DIR


def parse_date_input(date_str: str) -> date:
    """Parse date from yyyy-mm-dd format (HTML5 date input)."""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected yyyy-mm-dd")

router = APIRouter(tags=["Web UI"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["tojson"] = lambda obj: Markup(json.dumps(obj))


def _template_context(request: Request, **extra) -> dict:
    ctx = i18n_context(request)
    ctx.update(extra)
    return ctx


def _growth_chart_points(child: Optional[Child]) -> List[dict]:
    """Visit series for Chart.js (oldest first); empty if insufficient data."""
    if child is None or not child.visits:
        return []
    visits = sorted(
        child.visits,
        key=lambda v: v.visit_date or datetime.min,
    )
    out: List[dict] = []
    for v in visits:
        m = v.measurement
        if not m:
            continue
        h = m.predicted_height_cm or m.manual_height_cm
        w = m.manual_weight_kg or m.predicted_weight_kg
        if h is None and w is None:
            continue
        label = v.visit_date.strftime("%Y-%m-%d") if v.visit_date else ""
        out.append(
            {
                "label": label,
                "height": float(h) if h is not None else None,
                "weight": float(w) if w is not None else None,
            }
        )
    return out


def get_assessment_service() -> AssessmentService:
    """Placeholder; overridden at app startup in main.py."""
    raise NotImplementedError


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _template_context(request))


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
        ctx = _template_context(request)
        ctx["error"] = ctx["t"]["err_invalid_date"]
        return templates.TemplateResponse(request, "index.html", ctx)

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
        _template_context(
            request,
            result=result,
            error=error,
            image_filename=filename,
        ),
    )


@router.get("/children", response_class=HTMLResponse)
async def children_list(request: Request, db: Session = Depends(get_db)):
    children = db.query(Child).order_by(Child.name).all()
    return templates.TemplateResponse(
        request,
        "children.html",
        _template_context(request, children=children),
    )


@router.get("/children/{child_id}", response_class=HTMLResponse)
async def child_detail(
    request: Request, child_id: int, db: Session = Depends(get_db)
):
    child = db.query(Child).filter(Child.id == child_id).first()
    chart_points = _growth_chart_points(child)
    return templates.TemplateResponse(
        request,
        "child_detail.html",
        _template_context(
            request,
            child=child,
            growth_chart_json=json.dumps(chart_points),
            show_growth_chart=len(chart_points) >= 2,
        ),
    )
