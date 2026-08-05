/// One persisted assessment/report projected onto the provenance-rich clinical
/// CSV used for field identification, prediction review, and MUAC follow-up.
class ClinicalCsvRecord {
  const ClinicalCsvRecord({
    required this.childId,
    required this.childName,
    required this.area,
    required this.sex,
    required this.dateOfBirth,
    required this.measurementDate,
    required this.actualHeightCm,
    required this.calculatedHeightCm,
    required this.calculatedHeightMethod,
    required this.actualWeightKg,
    required this.calculatedWeightKg,
    required this.calculatedWeightMethod,
    required this.muacCm,
    required this.muacStatus,
    required this.muacMethod,
    required this.muacAgeInRange,
    required this.muacIsDirectMeasurement,
    required this.calculatedMuacCm,
    required this.calculatedMuacMethod,
    required this.muacConfidence,
    required this.muacUncertaintyLowerCm,
    required this.muacUncertaintyUpperCm,
    required this.muacModelVersion,
    required this.muacCalibrationVersion,
    required this.muacRequiresConfirmation,
    required this.muacReferralGuidance,
    required this.hazZscore,
    required this.whzZscore,
    required this.fieldCategory,
    required this.predictedFieldCategory,
    required this.stuntingPrediction,
    required this.wastingPrediction,
    required this.notes,
  });

  static const headers = <String>[
    'child_id',
    'child_name',
    'area',
    'sex',
    'date_of_birth',
    'measurement_date',
    'actual_height_cm',
    'calculated_height_cm',
    'calculated_height_method',
    'actual_weight_kg',
    'calculated_weight_kg',
    'calculated_weight_method',
    'muac_cm',
    'muac_status',
    'muac_method',
    'muac_age_in_range',
    'muac_is_direct_measurement',
    'calculated_muac_cm',
    'calculated_muac_method',
    'calculated_muac_confidence',
    'calculated_muac_uncertainty_lower_cm',
    'calculated_muac_uncertainty_upper_cm',
    'calculated_muac_model_version',
    'calculated_muac_calibration_version',
    'calculated_muac_requires_confirmation',
    'calculated_muac_referral_guidance',
    'haz_zscore',
    'whz_zscore',
    'field_category',
    'predicted_field_category',
    'stunting_prediction',
    'wasting_prediction',
    'notes',
  ];

  final int childId;
  final String childName;
  final String? area;
  final String sex;
  final String dateOfBirth;
  final String measurementDate;
  final double? actualHeightCm;
  final double? calculatedHeightCm;
  final String? calculatedHeightMethod;
  final double? actualWeightKg;
  final double? calculatedWeightKg;
  final String? calculatedWeightMethod;
  final double? muacCm;
  final String? muacStatus;
  final String? muacMethod;
  final bool? muacAgeInRange;
  final bool? muacIsDirectMeasurement;
  final double? calculatedMuacCm;
  final String? calculatedMuacMethod;
  final double? muacConfidence;
  final double? muacUncertaintyLowerCm;
  final double? muacUncertaintyUpperCm;
  final String? muacModelVersion;
  final String? muacCalibrationVersion;
  final bool? muacRequiresConfirmation;
  final String? muacReferralGuidance;
  final double? hazZscore;
  final double? whzZscore;
  final String? fieldCategory;
  final String? predictedFieldCategory;
  final String? stuntingPrediction;
  final String? wastingPrediction;
  final String? notes;

  List<Object?> toCsvRow() => <Object?>[
        childId,
        childName,
        area,
        sex,
        dateOfBirth,
        measurementDate,
        _formatNumber(actualHeightCm),
        _formatNumber(calculatedHeightCm),
        calculatedHeightMethod,
        _formatNumber(actualWeightKg),
        _formatNumber(calculatedWeightKg),
        calculatedWeightMethod,
        _formatNumber(muacCm),
        muacStatus,
        muacMethod,
        _formatBool(muacAgeInRange),
        _formatBool(muacIsDirectMeasurement),
        _formatNumber(calculatedMuacCm),
        calculatedMuacMethod,
        _formatNumber(muacConfidence),
        _formatNumber(muacUncertaintyLowerCm),
        _formatNumber(muacUncertaintyUpperCm),
        muacModelVersion,
        muacCalibrationVersion,
        _formatBool(muacRequiresConfirmation),
        muacReferralGuidance,
        _formatNumber(hazZscore),
        _formatNumber(whzZscore),
        fieldCategory,
        predictedFieldCategory,
        stuntingPrediction,
        wastingPrediction,
        notes,
      ];

  static String? _formatNumber(double? value) {
    if (value == null) return null;
    final fixed = value.toStringAsFixed(4);
    return fixed.replaceFirst(RegExp(r'\.?0+$'), '');
  }

  static String? _formatBool(bool? value) => value?.toString();
}
