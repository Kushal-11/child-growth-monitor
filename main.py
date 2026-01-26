"""
Child Growth Monitor - Application Entry Point

Starts the FastAPI application server with:
  - API routes at /api/v1/...
  - Web UI at /
  - Static file serving for uploaded images and CSS/JS
  - WHO data preloading at startup
  - Database initialization
"""
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.models.database import init_db
from app.services.assessment_service import AssessmentService
from app.services.who_data_service import WHODataService
from app.web.views import router as web_router
from config import UPLOAD_DIR


def create_app() -> FastAPI:
    app = FastAPI(
        title="Child Growth Monitor",
        description="WHO standard-based child growth assessment using computer vision",
        version="1.0.0",
    )

    # Initialize database tables
    init_db()

    # Load WHO reference data at startup
    who_data = WHODataService()
    who_data.load_all()

    # Create shared assessment service instance
    assessment_svc = AssessmentService(who_data)

    # Register routers
    app.include_router(api_router)
    app.include_router(web_router)

    # Wire up dependency injection via FastAPI's override mechanism
    from app.api.routes import get_assessment_service as api_dep
    from app.web.views import get_assessment_service as web_dep

    app.dependency_overrides[api_dep] = lambda: assessment_svc
    app.dependency_overrides[web_dep] = lambda: assessment_svc

    # Serve uploaded images
    UPLOAD_DIR.mkdir(exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

    # Serve static assets (CSS, JS)
    static_dir = Path(__file__).parent / "app" / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    print("=" * 60)
    print("  Child Growth Monitor")
    print("  Open http://localhost:8000 in your browser")
    print("  API docs at http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
