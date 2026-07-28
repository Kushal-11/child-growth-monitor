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

String classifyProgrammeBmi(double bmi, String sex) {
  final samBelow = sex.toUpperCase() == 'M' ? 13.0 : 12.8;
  final mamBelow = sex.toUpperCase() == 'M' ? 13.7 : 13.5;
  if (bmi < samBelow) return 'SAM';
  if (bmi < mamBelow) return 'MAM';
  return 'Normal';
}

String combineProgrammeBmiMuac(String? bmiStatus, String? muacStatus) {
  return combineNutritionStatus(
    whzStatus: bmiStatus,
    muacStatus: muacStatus,
  );
}

/// Combine WHZ, MUAC, and the ML wasting classifier into a single nutrition
/// verdict via the WHO 2009/2013 CMAM **OR-rule**: a child is SAM/MAM if ANY
/// available signal says so — the most-severe signal wins.
///
/// Ported from the backend `MUACService.combine_with_whz_status`, extended to
/// also weigh the ML prediction (matching the reference web result banner).
/// Returns one of: 'SAM' | 'MAM' | 'Overweight' | 'Risk_Overweight' |
/// 'Normal' | 'Unknown' ('Unknown' only when no signal is available).
///
/// SAFETY-CRITICAL: never collapse a SAM/MAM signal to 'Normal'. A tape-
/// measured SAM child (MUAC < 11.5) with a borderline-normal WHZ must still
/// read SAM. Inputs may be null when a measurement could not be computed.
String combineNutritionStatus({
  required String? whzStatus,
  required String? muacStatus,
  String? mlStatus,
}) {
  final best = [
    _nutritionStatusRank(_canonicalWhz(whzStatus)),
    _nutritionStatusRank(_canonicalMuac(muacStatus)),
    _nutritionStatusRank(_canonicalMl(mlStatus)),
  ].reduce((a, b) => a > b ? a : b);
  return _rankToNutritionStatus(best);
}

/// Severity ranks: higher = more clinically urgent. 0 = no signal.
int _nutritionStatusRank(String? canonical) {
  switch (canonical) {
    case 'SAM':
      return 5;
    case 'MAM':
      return 4;
    case 'Overweight':
      return 3;
    case 'Risk_Overweight':
      return 2;
    case 'Normal':
      return 1;
    default:
      return 0;
  }
}

String _rankToNutritionStatus(int rank) {
  switch (rank) {
    case 5:
      return 'SAM';
    case 4:
      return 'MAM';
    case 3:
      return 'Overweight';
    case 2:
      return 'Risk_Overweight';
    case 1:
      return 'Normal';
    default:
      return 'Unknown';
  }
}

/// WHZ arrives as the long [classifyWhz] label; normalise to a canonical code.
String? _canonicalWhz(String? s) {
  if (s == null) return null;
  if (s.contains('SAM')) return 'SAM';
  if (s.contains('MAM')) return 'MAM';
  if (s.contains('Risk')) return 'Risk_Overweight';
  if (s == 'Overweight' || s == 'Obese') return 'Overweight';
  return 'Normal';
}

/// MUAC arrives as [classifyMuac] codes: 'SAM' | 'At Risk (MAM)' | 'Normal'.
String? _canonicalMuac(String? s) {
  if (s == null) return null;
  if (s == 'SAM') return 'SAM';
  if (s.contains('MAM')) return 'MAM';
  return 'Normal';
}

/// ML wasting status uses the training labels (see [wastingLabels]). Anything
/// else (e.g. the 'who_fallback' sentinel) carries no nutrition signal.
String? _canonicalMl(String? s) {
  switch (s) {
    case 'SAM':
    case 'MAM':
    case 'Overweight':
    case 'Risk_Overweight':
    case 'Normal':
      return s;
    default:
      return null;
  }
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
  'age_months', 'sex_binary', 'height_cm', 'shoulder_width_cm',
  'hip_width_cm', 'torso_length_cm', 'upper_arm_length_cm',
  'shoulder_height_ratio', 'hip_height_ratio', 'body_build_score',
  'chest_depth_cm', 'abd_depth_cm', 'chest_depth_ratio', 'abd_depth_ratio',
];
