from datetime import datetime, timezone
from types import SimpleNamespace

from app.web.views import _growth_chart_points


def _visit(day, **measurement):
    defaults = {
        "effective_height_cm": None,
        "effective_weight_kg": None,
        "height_method": "unavailable",
        "weight_method": "unavailable",
        "predicted_height_cm": 99.0,
        "predicted_weight_kg": 19.0,
    }
    defaults.update(measurement)
    return SimpleNamespace(
        visit_date=datetime(2026, 1, day, tzinfo=timezone.utc),
        measurement=SimpleNamespace(**defaults),
    )


def test_growth_chart_excludes_camera_and_who_estimates():
    child = SimpleNamespace(
        visits=[
            _visit(1),
            _visit(
                2,
                effective_height_cm=88.0,
                effective_weight_kg=12.0,
                height_method="manual",
                weight_method="manual",
            ),
        ]
    )
    assert _growth_chart_points(child) == [
        {"label": "2026-01-02", "height": 88.0, "weight": 12.0}
    ]
