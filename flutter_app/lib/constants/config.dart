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

const _ratios0to12 = AnthropometricRatios(
  headRatio: 0.28,
  torsoRatio: 0.32,
  legRatio: 0.40,
);
const _ratios12to24 = AnthropometricRatios(
  headRatio: 0.25,
  torsoRatio: 0.32,
  legRatio: 0.43,
);
const _ratios24to48 = AnthropometricRatios(
  headRatio: 0.22,
  torsoRatio: 0.30,
  legRatio: 0.48,
);
const _ratios48to60 = AnthropometricRatios(
  headRatio: 0.20,
  torsoRatio: 0.30,
  legRatio: 0.50,
);

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

/// Conservative under-five entry plausibility gates.
const double minPlausibleHeightCm = 30.0;
const double maxPlausibleHeightCm = 130.0;
const double minPlausibleWeightKg = 0.5;
const double maxPlausibleWeightKg = 40.0;
const double minPlausibleMuacCm = 5.0;
const double maxPlausibleMuacCm = 25.0;
const double maxUnderFiveAgeMonths = 60.0;

/// ML weight must be 45-180% of WHO median
const double mlWeightLowerBound = 0.45;
const double mlWeightUpperBound = 1.80;

/// Contactless weight ranges combine native geometry spread with repeated
/// on-device inference. This floor prevents the UI from showing false
/// sub-kilogram precision before a paired real-child calibration set exists.
const double contactlessWeightRangeMinimumHalfWidthKg = 0.8;
const double contactlessGeometryPerturbationBase = 0.03;
const double contactlessGeometryPerturbationQualityPenalty = 0.08;

/// Days per month for age calculation
const double daysPerMonth = 30.4375;

/// Poshan Setu v1 programme thresholds.
const Map<String, (double, double)> poshanBmiThresholds = {
  'M': (13.0, 13.7),
  'F': (12.8, 13.5),
};
const double poshanMuacSamMaxCm = 11.5;
const double poshanMuacNormalMinCm = 12.5;
const double poshanMuacMinAgeMonths = 6.0;
const double poshanMuacMaxAgeMonths = 60.0;

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
  if (z < -3) return 'SAM';
  if (z < -2) return 'MAM';
  if (z < 1) return 'NORMAL';
  if (z < 2) return 'RISK_OVERWEIGHT';
  if (z < 3) return 'OVERWEIGHT';
  return 'OBESE';
}

String? classifyMuac(double muacCm, bool ageInRange) {
  if (!ageInRange) return null;
  if (muacCm < 11.5) return 'SAM';
  if (muacCm < 12.5) return 'MAM';
  return 'NORMAL';
}

/// Combine WHZ and independently measured MUAC into a single nutrition
/// verdict via the WHO 2009/2013 CMAM **OR-rule**.
///
/// Ported from the backend `MUACService.combine_with_whz_status`. WHZ-derived
/// MUAC is the same evidence transformed and must not be counted twice. ML
/// predictions remain decision support and do not define clinical status.
///
/// SAFETY-CRITICAL: never collapse a SAM/MAM signal to 'Normal'. A tape-
/// measured SAM child (MUAC < 11.5) with a borderline-normal WHZ must still
/// read SAM. Inputs may be null when a measurement could not be computed.
String combineNutritionStatus({
  required String? whzStatus,
  required String? muacStatus,
  required String? muacMethod,
  required bool isDirectMeasurement,
}) {
  final muacCanTrigger = isDirectMeasurement && muacMethod == 'manual';
  final best = [
    _nutritionStatusRank(_canonicalWhz(whzStatus)),
    _nutritionStatusRank(muacCanTrigger ? _canonicalMuac(muacStatus) : null),
  ].reduce((a, b) => a > b ? a : b);
  return _rankToNutritionStatus(best);
}

/// Severity ranks: higher = more clinically urgent. 0 = no signal.
int _nutritionStatusRank(String? canonical) {
  switch (canonical) {
    case 'SAM':
      return 6;
    case 'MAM':
      return 5;
    case 'OBESE':
      return 4;
    case 'OVERWEIGHT':
    case 'Overweight':
      return 3;
    case 'RISK_OVERWEIGHT':
    case 'Risk_Overweight':
      return 2;
    case 'NORMAL':
    case 'Normal':
      return 1;
    default:
      return 0;
  }
}

String _rankToNutritionStatus(int rank) {
  switch (rank) {
    case 6:
      return 'SAM';
    case 5:
      return 'MAM';
    case 4:
      return 'OBESE';
    case 3:
      return 'OVERWEIGHT';
    case 2:
      return 'RISK_OVERWEIGHT';
    case 1:
      return 'NORMAL';
    default:
      return 'UNKNOWN';
  }
}

/// WHZ arrives as the long [classifyWhz] label; normalise to a canonical code.
String? _canonicalWhz(String? s) {
  if (s == null) return null;
  if (s.contains('SAM')) return 'SAM';
  if (s.contains('MAM')) return 'MAM';
  if (s.contains('RISK') || s.contains('Risk')) return 'RISK_OVERWEIGHT';
  if (s == 'OBESE' || s == 'Obese') return 'OBESE';
  if (s == 'OVERWEIGHT' || s == 'Overweight') return 'OVERWEIGHT';
  return 'NORMAL';
}

String wastingStatusLabel(String status) {
  switch (status) {
    case 'SAM':
      return 'Severe Acute Malnutrition (SAM)';
    case 'MAM':
      return 'Moderate Acute Malnutrition (MAM)';
    case 'NORMAL':
    case 'Normal':
      return 'Normal';
    case 'RISK_OVERWEIGHT':
    case 'Risk_Overweight':
      return 'Possible Risk of Overweight';
    case 'OVERWEIGHT':
    case 'Overweight':
      return 'Overweight';
    case 'OBESE':
    case 'Obese':
      return 'Obese';
    default:
      return 'Unknown';
  }
}

/// MUAC arrives as [classifyMuac] codes: 'SAM' | 'At Risk (MAM)' | 'Normal'.
String? _canonicalMuac(String? s) {
  if (s == null) return null;
  if (s == 'SAM') return 'SAM';
  if (s.contains('MAM')) return 'MAM';
  return 'NORMAL';
}

// --- MUAC WHO medians (age_months, median_cm) ---

const List<(int, double)> muacBoys = [
  (3, 12.5),
  (6, 14.0),
  (9, 14.8),
  (12, 15.2),
  (18, 15.5),
  (24, 15.7),
  (30, 15.8),
  (36, 15.9),
  (42, 16.0),
  (48, 16.1),
  (54, 16.1),
  (60, 16.2),
];

const List<(int, double)> muacGirls = [
  (3, 12.3),
  (6, 13.8),
  (9, 14.6),
  (12, 14.9),
  (18, 15.2),
  (24, 15.4),
  (30, 15.5),
  (36, 15.6),
  (42, 15.7),
  (48, 15.7),
  (54, 15.8),
  (60, 15.8),
];

/// Wasting classifier labels (alphabetical, matching training order)
const List<String> wastingLabels = [
  'MAM',
  'Normal',
  'Overweight',
  'Risk_Overweight',
  'SAM',
];

/// Body-build adjustment multipliers for WHO median weight.
/// Slender children weigh ~5% less than median, stocky ~5% more.
double bodyBuildWeightAdjustment(String build) {
  switch (build) {
    case 'slender':
      return 0.95;
    case 'stocky':
      return 1.05;
    default:
      return 1.0;
  }
}

/// 14-feature names in exact order
const List<String> featureNames = [
  'age_months',
  'sex_binary',
  'height_cm',
  'shoulder_width_cm',
  'hip_width_cm',
  'torso_length_cm',
  'upper_arm_length_cm',
  'shoulder_height_ratio',
  'hip_height_ratio',
  'body_build_score',
  'chest_depth_cm',
  'abd_depth_cm',
  'chest_depth_ratio',
  'abd_depth_ratio',
];
