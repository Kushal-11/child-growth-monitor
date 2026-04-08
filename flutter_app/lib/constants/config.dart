/// Anthropometric segment ratios by age (Snyder et al. 1975)
class AnthropometricRatios {
  final double headRatio;
  final double torsoRatio;
  final double legRatio;

  const AnthropometricRatios({
    required this.headRatio,
    required this.torsoRatio,
    required this.legRatio,
  });
}

const _ratios0to12 = AnthropometricRatios(headRatio: 0.28, torsoRatio: 0.32, legRatio: 0.40);
const _ratios12to24 = AnthropometricRatios(headRatio: 0.25, torsoRatio: 0.32, legRatio: 0.43);
const _ratios24to48 = AnthropometricRatios(headRatio: 0.22, torsoRatio: 0.30, legRatio: 0.48);
const _ratios48to60 = AnthropometricRatios(headRatio: 0.20, torsoRatio: 0.30, legRatio: 0.50);

AnthropometricRatios getAnthropometricRatios(double ageMonths) {
  if (ageMonths < 12) return _ratios0to12;
  if (ageMonths < 24) return _ratios12to24;
  if (ageMonths < 48) return _ratios24to48;
  return _ratios48to60;
}

/// Height validation: flag if > 3 SD from WHO median
const double heightValidationSd = 3.0;

/// Max 15% difference between segment-based estimates
const double segmentAgreementThreshold = 0.15;

/// Minimum pose confidence to use measurement
const double minConfidenceThreshold = 0.5;

/// ML weight must be 45-180% of WHO median
const double mlWeightLowerBound = 0.45;
const double mlWeightUpperBound = 1.80;

/// Days per month for age calculation
const double daysPerMonth = 30.4375;

/// Expected shoulder-to-height ratios by age (for body build classification)
double expectedShoulderRatio(double ageMonths) {
  if (ageMonths < 24) return 0.200;
  if (ageMonths < 48) return 0.210;
  return 0.218;
}

/// Body build deviation thresholds
const double bodyBuildThresholdDisplay = 0.03;
const double bodyBuildThresholdMl = 0.02;

// --- Classification functions ---

String classifyHaz(double z) {
  if (z < -3) return 'Severely Stunted';
  if (z < -2) return 'Stunted';
  if (z < 2) return 'Normal';
  return 'Tall';
}

String classifyWhz(double z) {
  if (z < -3) return 'Severe Acute Malnutrition (SAM)';
  if (z < -2) return 'Moderate Acute Malnutrition (MAM)';
  if (z < 1) return 'Normal';
  if (z < 2) return 'Possible Risk of Overweight';
  if (z < 3) return 'Overweight';
  return 'Obese';
}

String? classifyMuac(double muacCm, bool ageInRange) {
  if (!ageInRange) return null;
  if (muacCm < 11.5) return 'SAM';
  if (muacCm < 12.5) return 'At Risk (MAM)';
  return 'Normal';
}

// --- MUAC WHO medians (age_months, median_cm) ---

const List<(int, double)> muacBoys = [
  (3, 12.5), (6, 14.0), (9, 14.8), (12, 15.2), (18, 15.5), (24, 15.7),
  (30, 15.8), (36, 15.9), (42, 16.0), (48, 16.1), (54, 16.1), (60, 16.2),
];

const List<(int, double)> muacGirls = [
  (3, 12.3), (6, 13.8), (9, 14.6), (12, 14.9), (18, 15.2), (24, 15.4),
  (30, 15.5), (36, 15.6), (42, 15.7), (48, 15.7), (54, 15.8), (60, 15.8),
];

/// Wasting classifier labels (alphabetical, matching training order)
const List<String> wastingLabels = ['MAM', 'Normal', 'Overweight', 'Risk_Overweight', 'SAM'];

/// 14-feature names in exact order
const List<String> featureNames = [
  'age_months', 'sex_binary', 'height_cm', 'shoulder_width_cm',
  'hip_width_cm', 'torso_length_cm', 'upper_arm_length_cm',
  'shoulder_height_ratio', 'hip_height_ratio', 'body_build_score',
  'chest_depth_cm', 'abd_depth_cm', 'chest_depth_ratio', 'abd_depth_ratio',
];
