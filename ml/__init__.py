"""
ML package for camera-based wasting (SAM/MAM) detection.

Overview
--------
Three complementary models are trained on WHO-derived synthetic data:

  1. weight_estimator  — regression MLP predicting child weight (kg) from body proportions
  2. wasting_classifier — 5-class MLP predicting SAM/MAM/Normal/Risk/Overweight

Training labels are generated entirely from the verified WHO Excel LMS files
(wfl/wfh boys and girls zscores).

Body width features (shoulder_width, hip_width) are REAL camera measurements
extracted from MediaPipe pose landmarks. The synthetic training data uses a
physics-based body proportion model (Snyder et al. 1975, NASA/SAE) to simulate
what those widths would look like for children of different wasting status.
This relationship is NOT from WHO standards — it is a physical approximation.
Accuracy will improve significantly with real labeled data.

WHO MUAC thresholds (<11.5 cm = SAM, 11.5–12.5 cm = MAM) are not yet
integrated — download WHO MUAC-for-age tables to enable that pathway.
"""
