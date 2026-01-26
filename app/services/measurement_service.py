"""
Measurement service using OpenCV and MediaPipe.

Pipeline:
  1. Load image via OpenCV
  2. Run MediaPipe PoseLandmarker to detect body landmarks
  3. Draw pose landmarks on annotated copy of the image
  4. Compute child height using hybrid approach:
     - Primary: WHO statistical estimation (age/sex median)
     - Supplementary: Anthropometric ratio-based estimation
     - Fallback: Reference object detection (if available)

Height estimation uses age, sex, and body segment measurements
to provide accurate height estimates without requiring reference objects.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import (
    PARLEG_LENGTH_CM,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_PRESENCE_CONFIDENCE,
    POSE_MODEL_PATH,
    UPLOAD_DIR,
    get_anthropometric_ratios,
    HEIGHT_RANGES_BY_AGE,
    SEGMENT_AGREEMENT_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmark = mp.tasks.vision.PoseLandmark
VisionRunningMode = mp.tasks.vision.RunningMode

# Pose skeleton connections for drawing
POSE_CONNECTIONS = [
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
    (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX),
]


@dataclass
class BodySegments:
    """Body segment measurements in pixels."""

    head_height_px: Optional[float] = None  # Top of head to chin
    torso_length_px: Optional[float] = None  # Shoulder midpoint to hip midpoint
    leg_length_px: Optional[float] = None  # Hip midpoint to heel
    shoulder_width_px: Optional[float] = None  # Left to right shoulder
    total_height_px: Optional[float] = None  # Top of head to heel
    
    # Pixel coordinates for key points
    head_top_y: Optional[float] = None
    chin_y: Optional[float] = None
    shoulder_midpoint_y: Optional[float] = None
    hip_midpoint_y: Optional[float] = None
    heel_y: Optional[float] = None
    
    # Segment visibility/confidence
    head_confidence: float = 0.0
    torso_confidence: float = 0.0
    leg_confidence: float = 0.0


@dataclass
class MeasurementOutput:
    """Result from the measurement pipeline."""

    predicted_height_cm: Optional[float] = None
    reference_object_detected: bool = False
    scale_factor: Optional[float] = None  # cm per pixel
    confidence_score: Optional[float] = None
    height_pixels: Optional[float] = None
    annotated_image_filename: Optional[str] = None
    
    # New fields for hybrid approach
    body_segments: Optional[BodySegments] = None
    estimation_method: str = "none"  # "anthropometric", "who_statistical", "reference_object", "manual"
    height_estimates: Optional[dict] = None  # Multiple estimates from different methods
    body_build: Optional[dict] = None  # Body build estimation (slender/average/stocky)
    weight_adjustment: float = 1.0  # Multiplier for weight based on body build


class MeasurementService:
    def __init__(self):
        self._landmarker_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=POSE_MIN_PRESENCE_CONFIDENCE,
        )

    def _measure_body_segments(
        self, landmarks, image_shape: Tuple[int, int]
    ) -> BodySegments:
        """
        Extract body segment measurements from pose landmarks.
        
        Segments measured:
        - Head: Top of head (estimated) to chin/mouth
        - Torso: Shoulder midpoint to hip midpoint
        - Legs: Hip midpoint to heel (lowest foot point)
        - Shoulder width: Left to right shoulder
        
        Returns BodySegments dataclass with all measurements in pixels.
        """
        h, w = image_shape
        segments = BodySegments()
        
        # Helper to get landmark pixel coords if visible
        def get_lm(idx, min_visibility=0.3):
            lm = landmarks[idx]
            if lm.visibility >= min_visibility:
                return (lm.x * w, lm.y * h, lm.visibility)
            return None
        
        # === HEAD MEASUREMENTS ===
        nose = get_lm(PoseLandmark.NOSE)
        left_eye = get_lm(PoseLandmark.LEFT_EYE_INNER)
        right_eye = get_lm(PoseLandmark.RIGHT_EYE_INNER)
        left_ear = get_lm(PoseLandmark.LEFT_EAR)
        right_ear = get_lm(PoseLandmark.RIGHT_EAR)
        mouth_left = get_lm(PoseLandmark.MOUTH_LEFT)
        mouth_right = get_lm(PoseLandmark.MOUTH_RIGHT)
        
        if nose and (left_eye or right_eye):
            # Estimate top of head using nose-to-eye distance
            eye_y = None
            if left_eye and right_eye:
                eye_y = (left_eye[1] + right_eye[1]) / 2
            elif left_eye:
                eye_y = left_eye[1]
            else:
                eye_y = right_eye[1]
            
            nose_y = nose[1]
            nose_to_eye = nose_y - eye_y
            
            # Top of head is approximately 2.5x nose-to-eye distance above nose
            # This accounts for forehead + hair
            head_top_y = nose_y - (nose_to_eye * 2.5)
            segments.head_top_y = head_top_y
            
            # Chin estimation: use mouth or estimate from nose
            if mouth_left and mouth_right:
                mouth_y = (mouth_left[1] + mouth_right[1]) / 2
                # Chin is roughly 0.5x nose-to-eye below mouth
                chin_y = mouth_y + (nose_to_eye * 0.5)
            else:
                # Estimate chin from nose (chin is ~1.5x nose-to-eye below nose)
                chin_y = nose_y + (nose_to_eye * 1.5)
            
            segments.chin_y = chin_y
            segments.head_height_px = chin_y - head_top_y
            
            # Head confidence based on visible landmarks
            visible_head_lms = sum(1 for lm in [nose, left_eye, right_eye, left_ear, right_ear] if lm)
            segments.head_confidence = visible_head_lms / 5.0
        
        # === SHOULDER MEASUREMENTS ===
        left_shoulder = get_lm(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_lm(PoseLandmark.RIGHT_SHOULDER)
        
        shoulder_midpoint_y = None
        if left_shoulder and right_shoulder:
            shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
            segments.shoulder_midpoint_y = shoulder_midpoint_y
            
            # Shoulder width (horizontal distance)
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            segments.shoulder_width_px = shoulder_width
        elif left_shoulder:
            shoulder_midpoint_y = left_shoulder[1]
            segments.shoulder_midpoint_y = shoulder_midpoint_y
        elif right_shoulder:
            shoulder_midpoint_y = right_shoulder[1]
            segments.shoulder_midpoint_y = shoulder_midpoint_y
        
        # === HIP MEASUREMENTS ===
        left_hip = get_lm(PoseLandmark.LEFT_HIP)
        right_hip = get_lm(PoseLandmark.RIGHT_HIP)
        
        hip_midpoint_y = None
        if left_hip and right_hip:
            hip_midpoint_y = (left_hip[1] + right_hip[1]) / 2
            segments.hip_midpoint_y = hip_midpoint_y
        elif left_hip:
            hip_midpoint_y = left_hip[1]
            segments.hip_midpoint_y = hip_midpoint_y
        elif right_hip:
            hip_midpoint_y = right_hip[1]
            segments.hip_midpoint_y = hip_midpoint_y
        
        # === TORSO LENGTH ===
        if shoulder_midpoint_y is not None and hip_midpoint_y is not None:
            segments.torso_length_px = abs(hip_midpoint_y - shoulder_midpoint_y)
            
            # Torso confidence
            visible_torso_lms = sum(1 for lm in [left_shoulder, right_shoulder, left_hip, right_hip] if lm)
            segments.torso_confidence = visible_torso_lms / 4.0
        
        # === LEG MEASUREMENTS ===
        foot_landmarks = [
            get_lm(PoseLandmark.LEFT_HEEL),
            get_lm(PoseLandmark.RIGHT_HEEL),
            get_lm(PoseLandmark.LEFT_ANKLE),
            get_lm(PoseLandmark.RIGHT_ANKLE),
            get_lm(PoseLandmark.LEFT_FOOT_INDEX),
            get_lm(PoseLandmark.RIGHT_FOOT_INDEX),
        ]
        visible_feet = [f for f in foot_landmarks if f is not None]
        
        if visible_feet and hip_midpoint_y is not None:
            # Use the lowest foot point as heel
            heel_y = max(f[1] for f in visible_feet)
            segments.heel_y = heel_y
            segments.leg_length_px = abs(heel_y - hip_midpoint_y)
            
            # Leg confidence
            segments.leg_confidence = len(visible_feet) / 6.0
        
        # === TOTAL HEIGHT ===
        if segments.head_top_y is not None and segments.heel_y is not None:
            segments.total_height_px = abs(segments.heel_y - segments.head_top_y)
        
        return segments

    def _estimate_height_from_anthropometric_ratios(
        self, segments: BodySegments, age_months: float
    ) -> dict:
        """
        Estimate height in cm using anthropometric ratios.
        
        Uses age-specific body segment ratios to convert pixel measurements
        to real-world height estimates. Multiple estimates are generated
        from different segments and combined.
        
        Args:
            segments: BodySegments with pixel measurements
            age_months: Child's age in months
            
        Returns:
            dict with:
                - estimates: list of individual estimates from each segment
                - combined_height_cm: weighted average of estimates
                - confidence: confidence score (0-1)
                - method: "anthropometric_ratios"
        """
        ratios = get_anthropometric_ratios(age_months)
        estimates = []
        weights = []
        
        # Method 1: Estimate from head height
        if segments.head_height_px and segments.head_height_px > 0:
            # If head is X% of total height, then total = head_px / head_ratio
            # But we need a reference to convert px to cm
            # We can use the relationship between segments
            pass  # Will combine with other segments below
        
        # Method 2: Use segment ratios to estimate from multiple segments
        # Key insight: If we have multiple segments, we can cross-validate
        # by checking if their ratios match expected ratios
        
        # Calculate the "unit" size based on different segments
        # unit = total_height / 1.0 (i.e., total height)
        
        unit_from_head = None
        unit_from_torso = None
        unit_from_legs = None
        
        if segments.head_height_px and segments.head_height_px > 0:
            # head_px = head_ratio * total_height_px
            # total_height_px = head_px / head_ratio
            unit_from_head = segments.head_height_px / ratios["head_ratio"]
            estimates.append({
                "source": "head",
                "height_px": unit_from_head,
                "confidence": segments.head_confidence
            })
            weights.append(segments.head_confidence)
        
        if segments.torso_length_px and segments.torso_length_px > 0:
            unit_from_torso = segments.torso_length_px / ratios["torso_ratio"]
            estimates.append({
                "source": "torso",
                "height_px": unit_from_torso,
                "confidence": segments.torso_confidence
            })
            weights.append(segments.torso_confidence)
        
        if segments.leg_length_px and segments.leg_length_px > 0:
            unit_from_legs = segments.leg_length_px / ratios["leg_ratio"]
            estimates.append({
                "source": "legs",
                "height_px": unit_from_legs,
                "confidence": segments.leg_confidence
            })
            weights.append(segments.leg_confidence)
        
        if not estimates:
            return {
                "estimates": [],
                "combined_height_cm": None,
                "combined_height_px": None,
                "confidence": 0.0,
                "method": "anthropometric_ratios"
            }
        
        # Compute weighted average of height estimates (in pixels)
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_height_px = sum(
                e["height_px"] * w for e, w in zip(estimates, weights)
            ) / total_weight
        else:
            weighted_height_px = np.mean([e["height_px"] for e in estimates])
        
        # Check agreement between estimates
        if len(estimates) > 1:
            height_values = [e["height_px"] for e in estimates]
            max_diff = (max(height_values) - min(height_values)) / np.mean(height_values)
            agreement_score = max(0, 1 - max_diff / SEGMENT_AGREEMENT_THRESHOLD)
        else:
            agreement_score = 0.5  # Single estimate, medium confidence
        
        # Overall confidence
        avg_segment_confidence = np.mean(weights) if weights else 0
        overall_confidence = avg_segment_confidence * agreement_score
        
        return {
            "estimates": estimates,
            "combined_height_px": weighted_height_px,
            "combined_height_cm": None,  # Will be set by WHO statistical method
            "confidence": overall_confidence,
            "method": "anthropometric_ratios"
        }

    def _estimate_height_from_who_statistics(
        self,
        segments: BodySegments,
        age_months: float,
        sex: str,
        who_data
    ) -> dict:
        """
        Estimate height using WHO growth chart statistics.
        
        Uses the WHO median height for age as a statistical baseline.
        Combines with segment ratios to scale the estimate.
        
        Key insight: If the child's body segment ratios match typical proportions,
        and pose detection is accurate, the pixel ratio between segments should
        match the expected ratio. We can then use WHO median as the baseline.
        
        Args:
            segments: BodySegments with pixel measurements
            age_months: Child's age in months
            sex: 'M' or 'F'
            who_data: WHODataService instance
            
        Returns:
            dict with height_cm estimate, confidence, and method info
        """
        # Get WHO median height for this age/sex
        age_int = int(round(age_months))
        if age_int > 59:
            age_int = 59
        if age_int < 0:
            age_int = 0
            
        median_height = who_data.get_median_height_for_age(sex, age_int)
        if median_height is None:
            return {
                "height_cm": None,
                "confidence": 0.0,
                "method": "who_statistical",
                "detail": "WHO data not available for this age"
            }
        
        # Get height range for validation
        height_range = who_data.get_height_range_for_age(sex, age_int, 3.0)
        
        # Strategy: Use the WHO median as our primary estimate
        # Adjust based on segment proportions if they suggest deviation from median
        
        # Get expected ratios for this age
        ratios = get_anthropometric_ratios(age_months)
        
        # Calculate expected segment sizes in cm (if child were at median height)
        expected_head_cm = median_height * ratios["head_ratio"]
        expected_torso_cm = median_height * ratios["torso_ratio"]
        expected_leg_cm = median_height * ratios["leg_ratio"]
        
        # If we have reliable total height in pixels, and good segment proportions,
        # we can estimate confidence in the median being accurate
        confidence = 0.5  # Base confidence from having age/sex data
        
        # Check if segment proportions match expected proportions
        if segments.total_height_px and segments.total_height_px > 0:
            proportion_checks = []
            
            if segments.head_height_px:
                actual_head_ratio = segments.head_height_px / segments.total_height_px
                expected_head_ratio = ratios["head_ratio"]
                head_ratio_error = abs(actual_head_ratio - expected_head_ratio) / expected_head_ratio
                proportion_checks.append(1 - min(head_ratio_error, 1))
            
            if segments.torso_length_px:
                actual_torso_ratio = segments.torso_length_px / segments.total_height_px
                expected_torso_ratio = ratios["torso_ratio"]
                torso_ratio_error = abs(actual_torso_ratio - expected_torso_ratio) / expected_torso_ratio
                proportion_checks.append(1 - min(torso_ratio_error, 1))
            
            if segments.leg_length_px:
                actual_leg_ratio = segments.leg_length_px / segments.total_height_px
                expected_leg_ratio = ratios["leg_ratio"]
                leg_ratio_error = abs(actual_leg_ratio - expected_leg_ratio) / expected_leg_ratio
                proportion_checks.append(1 - min(leg_ratio_error, 1))
            
            if proportion_checks:
                # Higher confidence if proportions match expected
                proportion_confidence = np.mean(proportion_checks)
                confidence = 0.5 + (0.5 * proportion_confidence)
        
        # The WHO median is our best statistical estimate for this age/sex
        # Without absolute scale reference, we can't do better than the population median
        height_cm = median_height
        
        # Provide range info for user
        detail = f"Based on WHO median for {age_int}mo {sex}. "
        if height_range:
            detail += f"Normal range: {height_range[0]:.1f}-{height_range[1]:.1f} cm"
        
        return {
            "height_cm": height_cm,
            "median_height_cm": median_height,
            "height_range": height_range,
            "confidence": confidence,
            "method": "who_statistical",
            "detail": detail
        }

    def _validate_height_estimate(
        self,
        height_cm: float,
        age_months: float,
        sex: str,
        who_data
    ) -> dict:
        """
        Validate a height estimate against WHO growth standards.
        
        Checks if the estimated height falls within expected ranges for
        the child's age and sex. Returns validation status and confidence.
        
        Args:
            height_cm: Estimated height in centimeters
            age_months: Child's age in months
            sex: 'M' or 'F'
            who_data: WHODataService instance
            
        Returns:
            dict with:
                - is_valid: Boolean indicating if height is within ±3 SD
                - is_plausible: Boolean indicating if height is within ±5 SD
                - z_score_approx: Approximate Z-score (deviation from median)
                - confidence: Confidence score (higher if closer to median)
                - warnings: List of validation warnings
        """
        age_int = int(round(age_months))
        if age_int > 59:
            age_int = 59
        if age_int < 0:
            age_int = 0
        
        warnings = []
        
        # Get WHO reference data
        median_height = who_data.get_median_height_for_age(sex, age_int)
        height_sd = who_data.get_height_sd_for_age(sex, age_int)
        height_range = who_data.get_height_range_for_age(sex, age_int, 3.0)
        
        if median_height is None or height_sd is None:
            return {
                "is_valid": False,
                "is_plausible": False,
                "z_score_approx": None,
                "confidence": 0.0,
                "warnings": ["WHO reference data not available for this age"]
            }
        
        # Calculate approximate Z-score
        z_score = (height_cm - median_height) / height_sd
        
        # Validation checks
        is_valid = abs(z_score) <= 3.0  # Within ±3 SD
        is_plausible = abs(z_score) <= 5.0  # Within ±5 SD (extreme but possible)
        
        if not is_plausible:
            warnings.append(f"Height {height_cm:.1f}cm is extremely unlikely for {age_int}mo {sex}")
        elif not is_valid:
            warnings.append(f"Height {height_cm:.1f}cm is outside normal range (z={z_score:.1f})")
        
        # Sanity check using age-based ranges
        for (age_min, age_max), (height_min, height_max) in HEIGHT_RANGES_BY_AGE.items():
            if age_min <= age_months < age_max:
                if height_cm < height_min * 0.8:  # 20% below minimum
                    warnings.append(f"Height unusually low for age {age_int}mo")
                    is_plausible = False
                elif height_cm > height_max * 1.2:  # 20% above maximum
                    warnings.append(f"Height unusually high for age {age_int}mo")
                    is_plausible = False
                break
        
        # Confidence scoring based on Z-score
        # Higher confidence closer to median, drops off towards ±3 SD
        if abs(z_score) <= 1:
            confidence = 0.9
        elif abs(z_score) <= 2:
            confidence = 0.7
        elif abs(z_score) <= 3:
            confidence = 0.5
        else:
            confidence = max(0.1, 0.5 - (abs(z_score) - 3) * 0.1)
        
        return {
            "is_valid": is_valid,
            "is_plausible": is_plausible,
            "z_score_approx": round(z_score, 2),
            "median_height_cm": median_height,
            "height_range_3sd": height_range,
            "confidence": confidence,
            "warnings": warnings
        }

    def _estimate_body_build(
        self, segments: BodySegments, age_months: float
    ) -> dict:
        """
        Estimate body build from shoulder width and height proportions.
        
        Uses shoulder width relative to total height to classify body build.
        This can be used to adjust weight estimates (wider build = likely heavier).
        
        Body build categories:
        - slender: shoulder_ratio < 0.20
        - average: 0.20 <= shoulder_ratio < 0.25
        - stocky: shoulder_ratio >= 0.25
        
        Args:
            segments: BodySegments with pixel measurements
            age_months: Child's age in months
            
        Returns:
            dict with body_build category, shoulder_to_height_ratio, and weight_adjustment
        """
        if not segments.shoulder_width_px or not segments.total_height_px:
            return {
                "body_build": "unknown",
                "shoulder_to_height_ratio": None,
                "weight_adjustment": 1.0,  # No adjustment
                "confidence": 0.0
            }
        
        if segments.total_height_px <= 0:
            return {
                "body_build": "unknown",
                "shoulder_to_height_ratio": None,
                "weight_adjustment": 1.0,
                "confidence": 0.0
            }
        
        # Calculate shoulder width to height ratio
        shoulder_ratio = segments.shoulder_width_px / segments.total_height_px
        
        # Expected ratios vary by age (younger children have proportionally smaller shoulders)
        # Typical shoulder/height ratio: 0.20-0.25 for children
        if age_months < 24:
            expected_ratio = 0.19
        elif age_months < 48:
            expected_ratio = 0.21
        else:
            expected_ratio = 0.23
        
        # Classify body build
        ratio_deviation = shoulder_ratio - expected_ratio
        
        if ratio_deviation < -0.03:
            body_build = "slender"
            weight_adjustment = 0.95  # -5% weight
        elif ratio_deviation > 0.03:
            body_build = "stocky"
            weight_adjustment = 1.05  # +5% weight
        else:
            body_build = "average"
            weight_adjustment = 1.0
        
        # Confidence based on segment visibility
        confidence = min(segments.torso_confidence, 0.8)
        
        return {
            "body_build": body_build,
            "shoulder_to_height_ratio": round(shoulder_ratio, 3),
            "expected_ratio": expected_ratio,
            "weight_adjustment": weight_adjustment,
            "confidence": confidence
        }

    def process_image(self, image_path: str) -> MeasurementOutput:
        """
        Basic image processing pipeline (legacy compatibility).
        
        For full hybrid estimation with age/sex, use process_image_with_estimation().
        """
        return self.process_image_with_estimation(image_path, None, None, None)

    def process_image_with_estimation(
        self,
        image_path: str,
        age_months: Optional[float],
        sex: Optional[str],
        who_data=None
    ) -> MeasurementOutput:
        """
        Full measurement pipeline with hybrid height estimation.
        
        Uses multiple methods to estimate height:
        1. WHO statistical estimation using age/sex (primary)
        2. Anthropometric ratios based on body segments (supplementary)
        3. Reference object detection (fallback, if available)
        4. Validation against WHO growth standards
        
        Args:
            image_path: Path to the image file
            age_months: Child's age in months (optional but recommended)
            sex: 'M' or 'F' (optional but recommended)
            who_data: WHODataService instance (optional)
            
        Returns:
            MeasurementOutput with height estimate and confidence
        """
        image = cv2.imread(image_path)
        if image is None:
            return MeasurementOutput(confidence_score=0.0)

        # Step 1: Detect body pose and get landmarks
        pose_result = self._detect_pose(image_path, image.shape[:2])
        head_y = pose_result["head_y"]
        heel_y = pose_result["heel_y"]
        confidence = pose_result["confidence"]
        landmarks_px = pose_result["landmarks_px"]
        raw_landmarks = pose_result.get("raw_landmarks")
        posture_valid = pose_result.get("posture_valid", True)
        posture_issues = pose_result.get("posture_issues", [])

        # Step 2: Detect reference object (optional, kept for backwards compatibility)
        scale_factor, ref_detected, parleg_box = self._detect_parlegi(image)

        # Step 3: Measure body segments if landmarks available
        body_segments = None
        if raw_landmarks is not None:
            body_segments = self._measure_body_segments(raw_landmarks, image.shape[:2])

        # Step 4: Draw annotations on image
        annotated_filename = self._draw_annotations(
            image, image_path, landmarks_px, head_y, heel_y, parleg_box
        )

        # Initialize result
        result = MeasurementOutput(
            reference_object_detected=ref_detected,
            scale_factor=scale_factor,
            confidence_score=confidence,
            annotated_image_filename=annotated_filename,
            body_segments=body_segments,
        )

        # Step 5: Compute height in pixels
        if head_y is not None and heel_y is not None:
            height_pixels = abs(heel_y - head_y)
            result.height_pixels = height_pixels

        # Step 6: Hybrid height estimation
        height_estimates = {}
        final_height_cm = None
        estimation_method = "none"

        # Method A: WHO Statistical (primary method when age/sex available)
        if age_months is not None and sex is not None and who_data is not None:
            who_estimate = self._estimate_height_from_who_statistics(
                body_segments if body_segments else BodySegments(),
                age_months, sex, who_data
            )
            height_estimates["who_statistical"] = who_estimate
            
            if who_estimate["height_cm"] is not None:
                final_height_cm = who_estimate["height_cm"]
                estimation_method = "who_statistical"
                result.confidence_score = who_estimate["confidence"]

        # Method B: Anthropometric ratios (supplementary)
        if body_segments and age_months is not None:
            anthro_estimate = self._estimate_height_from_anthropometric_ratios(
                body_segments, age_months
            )
            height_estimates["anthropometric"] = anthro_estimate
            
            # If WHO method unavailable, use anthropometric segment ratios
            # Note: This only gives relative proportions, not absolute height
            # without a reference object or WHO baseline

        # Method C: Reference object detection - kept as fallback option
        if ref_detected and result.height_pixels is not None and scale_factor is not None:
            ref_height = round(result.height_pixels * scale_factor, 1)
            height_estimates["reference_object"] = {
                "height_cm": ref_height,
                "confidence": 0.3,  # Low confidence due to unreliability
                "method": "reference_object"
            }
            
            # Only use reference object if WHO method unavailable
            if final_height_cm is None:
                final_height_cm = ref_height
                estimation_method = "reference_object"
                result.confidence_score = 0.3

        # Step 7: Validate the final estimate
        if final_height_cm is not None and age_months is not None and sex is not None and who_data is not None:
            validation = self._validate_height_estimate(
                final_height_cm, age_months, sex, who_data
            )
            height_estimates["validation"] = validation
            
            # Adjust confidence based on validation
            if not validation["is_plausible"]:
                result.confidence_score *= 0.5
            elif not validation["is_valid"]:
                result.confidence_score *= 0.7

        # Step 8: Estimate body build for weight adjustment
        if body_segments and age_months is not None:
            body_build = self._estimate_body_build(body_segments, age_months)
            result.body_build = body_build
            result.weight_adjustment = body_build.get("weight_adjustment", 1.0)
            height_estimates["body_build"] = body_build

        # Set final values
        result.predicted_height_cm = final_height_cm
        result.estimation_method = estimation_method
        result.height_estimates = height_estimates

        # Reduce confidence if posture issues detected
        if not posture_valid:
            result.confidence_score *= 0.8

        return result

    def _detect_parlegi(
        self, image: np.ndarray
    ) -> Tuple[Optional[float], bool, Optional[np.ndarray]]:
        """Detect reference object (yellow rectangular packet) and compute scale factor (cm/pixel).

        This is a fallback method for scale calibration. The primary height estimation
        uses WHO statistical methods and anthropometric ratios, which don't require reference objects.

        Returns (scale_factor_cm_per_pixel, was_detected, box_points_or_None).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Broader HSV ranges to catch yellow reference objects under varying lighting
        yellow_ranges = [
            (np.array([15, 60, 60]), np.array([40, 255, 255])),   # standard yellow
            (np.array([10, 50, 100]), np.array([30, 255, 255])),  # warm/orange-yellow
            (np.array([20, 80, 80]), np.array([45, 255, 255])),   # bright yellow
        ]

        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in yellow_ranges:
            combined_mask |= cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_rect = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue

            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)

            # Reference object aspect ratio: 12.7 / 5.5 ~ 2.31
            # Broader tolerance: 1.5 to 3.5 to account for perspective distortion
            if 1.5 <= aspect_ratio <= 3.5 and area > best_area:
                best_rect = rect
                best_area = area

        if best_rect is None:
            return None, False, None

        w, h = best_rect[1]
        longer_edge_pixels = max(w, h)
        scale_factor = PARLEG_LENGTH_CM / longer_edge_pixels
        box_points = cv2.boxPoints(best_rect).astype(int)
        return scale_factor, True, box_points

    def _detect_pose(
        self, image_path: str, image_shape: Tuple[int, int]
    ) -> dict:
        """
        Detect body pose with enhanced accuracy.
        
        Improvements:
        - Better head top estimation using multiple face landmarks
        - Posture validation (checks for straight standing pose)
        - Returns raw landmarks for segment measurement
        - More robust confidence scoring
        
        Returns dict with:
            head_y, heel_y: Y coordinates in pixels
            confidence: Overall pose confidence (0-1)
            landmarks_px: List of (x, y, visibility) tuples
            raw_landmarks: Original MediaPipe landmarks object
            posture_valid: Boolean indicating if posture is suitable for measurement
            posture_issues: List of detected posture problems
        """
        h, w = image_shape
        empty = {
            "head_y": None,
            "heel_y": None,
            "confidence": 0.0,
            "landmarks_px": [],
            "raw_landmarks": None,
            "posture_valid": False,
            "posture_issues": ["No pose detected"],
        }

        mp_image = mp.Image.create_from_file(image_path)

        with PoseLandmarker.create_from_options(self._landmarker_options) as landmarker:
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return empty

            landmarks = result.pose_landmarks[0]

            # Convert all landmarks to pixel coordinates for annotation
            landmarks_px = [
                (int(lm.x * w), int(lm.y * h), lm.visibility)
                for lm in landmarks
            ]

            # Helper to get landmark if visible
            def get_lm_y(idx, min_vis=0.3):
                lm = landmarks[idx]
                return lm.y * h if lm.visibility >= min_vis else None

            def get_lm_xy(idx, min_vis=0.3):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h) if lm.visibility >= min_vis else None

            # === IMPROVED HEAD ESTIMATION ===
            # Use multiple face landmarks for better estimation
            nose = landmarks[PoseLandmark.NOSE]
            left_eye_inner = landmarks[PoseLandmark.LEFT_EYE_INNER]
            right_eye_inner = landmarks[PoseLandmark.RIGHT_EYE_INNER]
            left_eye = landmarks[PoseLandmark.LEFT_EYE]
            right_eye = landmarks[PoseLandmark.RIGHT_EYE]
            left_ear = landmarks[PoseLandmark.LEFT_EAR]
            right_ear = landmarks[PoseLandmark.RIGHT_EAR]

            # Need at least nose and one eye for head estimation
            if nose.visibility < 0.3:
                return {**empty, "landmarks_px": landmarks_px, "raw_landmarks": landmarks}

            # Calculate eye level (average of all visible eye landmarks)
            eye_ys = []
            for eye_lm in [left_eye_inner, right_eye_inner, left_eye, right_eye]:
                if eye_lm.visibility >= 0.3:
                    eye_ys.append(eye_lm.y * h)
            
            if not eye_ys:
                return {**empty, "landmarks_px": landmarks_px, "raw_landmarks": landmarks}

            eye_y_avg = np.mean(eye_ys)
            nose_y = nose.y * h
            nose_to_eye = nose_y - eye_y_avg

            # Calculate ear level for head width estimation (helps validate head top)
            ear_ys = []
            for ear_lm in [left_ear, right_ear]:
                if ear_lm.visibility >= 0.3:
                    ear_ys.append(ear_lm.y * h)
            
            # Top of head estimation with validation
            # Method 1: 2.5x nose-to-eye distance above nose
            head_top_method1 = nose_y - (nose_to_eye * 2.5)
            
            # Method 2: If ears visible, use them for validation
            # Ears should be roughly at eye level; head top is about 1 head-height above eyes
            if ear_ys:
                ear_y_avg = np.mean(ear_ys)
                # Head width ≈ eye-to-ear distance * 2
                # Head height ≈ head width * 1.2 for children
                head_top_method2 = ear_y_avg - (nose_to_eye * 3.0)
                
                # Use average of both methods for robustness
                top_of_head_y = (head_top_method1 + head_top_method2) / 2
            else:
                top_of_head_y = head_top_method1

            # === FEET DETECTION ===
            foot_landmark_ids = [
                PoseLandmark.LEFT_HEEL,
                PoseLandmark.RIGHT_HEEL,
                PoseLandmark.LEFT_ANKLE,
                PoseLandmark.RIGHT_ANKLE,
                PoseLandmark.LEFT_FOOT_INDEX,
                PoseLandmark.RIGHT_FOOT_INDEX,
            ]
            foot_ys = [
                landmarks[lm].y * h
                for lm in foot_landmark_ids
                if landmarks[lm].visibility > 0.3
            ]

            if not foot_ys:
                return {**empty, "landmarks_px": landmarks_px, "raw_landmarks": landmarks}

            heel_y = max(foot_ys)

            # === POSTURE VALIDATION ===
            posture_issues = []
            
            # Check 1: Shoulder alignment (should be roughly horizontal)
            left_shoulder = get_lm_xy(PoseLandmark.LEFT_SHOULDER)
            right_shoulder = get_lm_xy(PoseLandmark.RIGHT_SHOULDER)
            if left_shoulder and right_shoulder:
                shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) / max(1, abs(left_shoulder[0] - right_shoulder[0]))
                if shoulder_tilt > 0.3:  # >30% tilt
                    posture_issues.append("Shoulders tilted")
            
            # Check 2: Hip alignment
            left_hip = get_lm_xy(PoseLandmark.LEFT_HIP)
            right_hip = get_lm_xy(PoseLandmark.RIGHT_HIP)
            if left_hip and right_hip:
                hip_tilt = abs(left_hip[1] - right_hip[1]) / max(1, abs(left_hip[0] - right_hip[0]))
                if hip_tilt > 0.3:
                    posture_issues.append("Hips tilted")
            
            # Check 3: Standing straight (shoulders roughly above hips)
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
                hip_mid_x = (left_hip[0] + right_hip[0]) / 2
                body_lean = abs(shoulder_mid_x - hip_mid_x) / w
                if body_lean > 0.1:  # >10% of image width
                    posture_issues.append("Body leaning")
            
            # Check 4: Both feet visible and on similar level
            left_heel_y = get_lm_y(PoseLandmark.LEFT_HEEL)
            right_heel_y = get_lm_y(PoseLandmark.RIGHT_HEEL)
            if left_heel_y and right_heel_y:
                heel_diff = abs(left_heel_y - right_heel_y) / h
                if heel_diff > 0.05:  # >5% of image height
                    posture_issues.append("Feet at different levels")
            
            posture_valid = len(posture_issues) == 0

            # === CONFIDENCE SCORING ===
            # Include more landmarks in confidence calculation
            key_lm_ids = [
                PoseLandmark.NOSE,
                PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
                PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
                PoseLandmark.LEFT_HEEL, PoseLandmark.RIGHT_HEEL,
            ]
            visibilities = [landmarks[lm].visibility for lm in key_lm_ids]
            avg_confidence = float(np.mean(visibilities))
            
            # Reduce confidence if posture is not valid
            if not posture_valid:
                avg_confidence *= 0.8

            return {
                "head_y": top_of_head_y,
                "heel_y": heel_y,
                "confidence": round(avg_confidence, 3),
                "landmarks_px": landmarks_px,
                "raw_landmarks": landmarks,
                "posture_valid": posture_valid,
                "posture_issues": posture_issues if posture_issues else [],
            }

    def _draw_annotations(
        self,
        image: np.ndarray,
        image_path: str,
        landmarks_px: List[Tuple[int, int, float]],
        head_y: Optional[float],
        heel_y: Optional[float],
        parleg_box: Optional[np.ndarray],
    ) -> Optional[str]:
        """Draw pose landmarks and measurement lines on the image. Save annotated copy."""
        if not landmarks_px:
            return None

        annotated = image.copy()
        h, w = image.shape[:2]

        # Draw skeleton connections
        for start_lm, end_lm in POSE_CONNECTIONS:
            sx, sy, s_vis = landmarks_px[start_lm]
            ex, ey, e_vis = landmarks_px[end_lm]
            if s_vis > 0.3 and e_vis > 0.3:
                cv2.line(annotated, (sx, sy), (ex, ey), (0, 255, 128), 2)

        # Draw landmark points
        for x, y, vis in landmarks_px:
            if vis > 0.3:
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
                cv2.circle(annotated, (x, y), 5, (255, 255, 255), 1)

        # Draw head-to-heel measurement line
        if head_y is not None and heel_y is not None:
            mid_x = w // 3  # draw line in left third of image
            head_pt = (mid_x, int(head_y))
            heel_pt = (mid_x, int(heel_y))

            # Vertical height line
            cv2.line(annotated, head_pt, heel_pt, (255, 0, 255), 2)
            # Top cap
            cv2.line(annotated, (mid_x - 15, int(head_y)), (mid_x + 15, int(head_y)), (255, 0, 255), 2)
            # Bottom cap
            cv2.line(annotated, (mid_x - 15, int(heel_y)), (mid_x + 15, int(heel_y)), (255, 0, 255), 2)
            # Top of head marker
            cv2.circle(annotated, (mid_x, int(head_y)), 6, (255, 0, 255), -1)

            # Label with pixel height
            height_px = abs(heel_y - head_y)
            cv2.putText(
                annotated,
                f"{height_px:.0f}px",
                (mid_x + 20, int((head_y + heel_y) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

        # Draw reference object bounding box (if detected)
        if parleg_box is not None:
            cv2.drawContours(annotated, [parleg_box], 0, (0, 255, 0), 3)
            cx = int(np.mean(parleg_box[:, 0]))
            cy = int(np.mean(parleg_box[:, 1])) - 15
            cv2.putText(
                annotated,
                "Reference",
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Save annotated image
        orig_name = Path(image_path).stem
        annotated_filename = f"{orig_name}_annotated.jpg"
        UPLOAD_DIR.mkdir(exist_ok=True)
        annotated_path = UPLOAD_DIR / annotated_filename
        cv2.imwrite(str(annotated_path), annotated)

        return annotated_filename
