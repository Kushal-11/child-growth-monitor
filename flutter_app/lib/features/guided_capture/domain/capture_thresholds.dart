/// Version persisted with every guided-capture asset and camera result.
const String captureThresholdVersion = 'guided_capture_quality_v1';

/// Live pose thresholds.
const double captureMinLandmarkLikelihood = 0.50;
const double captureEdgeMarginFraction = 0.02;
const double captureMinBodyCoverageFraction = 0.50;
const double captureTargetBodyCoverageFraction = 0.75;
const double captureMaxCenterOffsetFraction = 0.20;
const double captureFrontMinWidthToBodyFraction = 0.10;
const double captureFrontTargetWidthToBodyFraction = 0.18;
const double captureSideMaxWidthToBodyFraction = 0.065;
const double captureSideTargetWidthToBodyFraction = 0.02;
const double captureMaxTiltDegrees = 12;

/// Post-capture still-image thresholds.
const double captureMinMeanLuminance = 0.15;
const double captureMaxMeanLuminance = 0.90;
const double captureTargetContrast = 0.20;
const double captureMinContrast = 0.05;
const double captureTargetSharpness = 0.25;
const double captureMinSharpness = 0.03;
const double captureMinBurstOverallScore = 0.55;
const double captureLiveScoreWeight = 0.55;
const double captureStillScoreWeight = 0.45;

/// Burst-controller defaults.
const int captureStableFrameCount = 8;
const int captureBurstFrameCount = 3;
const int captureRetainedFrameCount = 2;
