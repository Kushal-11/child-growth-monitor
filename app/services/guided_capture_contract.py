"""Canonical wire values and pure guided-capture state transitions."""

from enum import Enum


class CaptureState(str, Enum):
    DRAFT_CAPTURE = "draft_capture"
    INCOMPLETE_CAPTURE = "incomplete_capture"
    PROCESSING = "processing"
    ESTIMATED_REPORT = "estimated_report"
    PROCESSING_FAILED = "processing_failed"
    MEASURED_REPORT = "measured_report"


class CaptureAssetRole(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    ARM_FRONT = "arm_front"
    ARM_SIDE = "arm_side"


class MeasurementMode(str, Enum):
    STANDING_HEIGHT = "standing_height"
    RECUMBENT_LENGTH = "recumbent_length"


class OedemaStatus(str, Enum):
    YES = "yes"
    NO = "no"
    NOT_CHECKED = "not_checked"


REQUIRED_CAPTURE_ROLES = (
    CaptureAssetRole.FRONT,
    CaptureAssetRole.SIDE,
)

ALLOWED_CAPTURE_TRANSITIONS = {
    CaptureState.DRAFT_CAPTURE: (
        CaptureState.INCOMPLETE_CAPTURE,
        CaptureState.PROCESSING,
    ),
    CaptureState.INCOMPLETE_CAPTURE: (CaptureState.DRAFT_CAPTURE,),
    CaptureState.PROCESSING: (
        CaptureState.ESTIMATED_REPORT,
        CaptureState.PROCESSING_FAILED,
    ),
    CaptureState.ESTIMATED_REPORT: (CaptureState.MEASURED_REPORT,),
    CaptureState.PROCESSING_FAILED: (CaptureState.PROCESSING,),
    CaptureState.MEASURED_REPORT: (CaptureState.MEASURED_REPORT,),
}


def can_transition_capture_state(
    current: CaptureState,
    target: CaptureState,
) -> bool:
    """Return whether the canonical state machine allows this transition."""
    current_state = CaptureState(current)
    target_state = CaptureState(target)
    return target_state in ALLOWED_CAPTURE_TRANSITIONS[current_state]


def require_capture_transition(
    current: CaptureState,
    target: CaptureState,
) -> None:
    """Raise when a caller attempts a non-canonical state transition."""
    current_state = CaptureState(current)
    target_state = CaptureState(target)
    if not can_transition_capture_state(current_state, target_state):
        raise ValueError(
            "Invalid capture-state transition: "
            f"{current_state.value} -> {target_state.value}"
        )
