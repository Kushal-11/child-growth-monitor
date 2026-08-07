/// One completed visit projected onto the typed, provenance-rich clinical CSV.
///
/// Measured and estimated anthropometry are intentionally independent. A WHO
/// score in an `actual_*` column is calculated only from measured values; a
/// score in a `calculated_*` column is calculated only from estimated values.
class ClinicalCsvRecord {
  const ClinicalCsvRecord({
    required this.exportSchemaVersion,
    required this.childName,
    required this.childId,
    required this.guardianName,
    required this.area,
    required this.sex,
    required this.dateOfBirth,
    required this.measurementDate,
    required this.ageDays,
    required this.recordedAgeMonths,
    required this.whoAgeMonths,
    required this.visitUuid,
    required this.entryMethod,
    required this.captureState,
    required this.consentVersion,
    required this.consentTimestamp,
    required this.consentOperatorIdentifier,
    required this.measurementMode,
    required this.whoExpectedMeasurementMode,
    required this.positionAdjustmentCm,
    required this.oedema,
    required this.oedemaGrade,
    required this.measuredAt,
    required this.measurementUpdateReason,
    required this.actualHeightCm,
    required this.actualHeightMethod,
    required this.actualWhoAdjustedHeightCm,
    required this.calculatedHeightCm,
    required this.calculatedHeightMethod,
    required this.calculatedHeightConfidence,
    required this.calculatedWhoAdjustedHeightCm,
    required this.heightErrorCm,
    required this.actualWeightKg,
    required this.actualWeightMethod,
    required this.calculatedWeightKg,
    required this.calculatedWeightMethod,
    required this.calculatedWeightConfidence,
    required this.weightErrorKg,
    required this.actualMuacCm,
    required this.actualMuacStatus,
    required this.actualMuacMethod,
    required this.actualMuacIsDirectMeasurement,
    required this.muacAgeInRange,
    required this.calculatedMuacCm,
    required this.calculatedMuacStatus,
    required this.calculatedMuacMethod,
    required this.calculatedMuacConfidence,
    required this.calculatedMuacUncertaintyLowerCm,
    required this.calculatedMuacUncertaintyUpperCm,
    required this.calculatedMuacModelVersion,
    required this.calculatedMuacCalibrationVersion,
    required this.calculatedMuacRequiresConfirmation,
    required this.calculatedMuacReferralGuidance,
    required this.muacErrorCm,
    required this.actualBmi,
    required this.actualHazZscore,
    required this.actualStuntingClassification,
    required this.actualHazQualityFlag,
    required this.actualWhzZscore,
    required this.actualWastingClassification,
    required this.actualWhzQualityFlag,
    required this.actualWazZscore,
    required this.actualUnderweightClassification,
    required this.actualWazQualityFlag,
    required this.actualBazZscore,
    required this.actualBmiForAgeClassification,
    required this.actualBazQualityFlag,
    required this.actualWhoCalculationNotes,
    required this.calculatedBmi,
    required this.calculatedHazZscore,
    required this.calculatedStuntingPrediction,
    required this.calculatedHazQualityFlag,
    required this.calculatedWhzZscore,
    required this.calculatedWastingPrediction,
    required this.calculatedWhzQualityFlag,
    required this.calculatedWazZscore,
    required this.calculatedUnderweightPrediction,
    required this.calculatedWazQualityFlag,
    required this.calculatedBazZscore,
    required this.calculatedBmiForAgePrediction,
    required this.calculatedBazQualityFlag,
    required this.calculatedWhoCalculationNotes,
    required this.whoStandardVersion,
    required this.actualAcuteNutritionClassification,
    required this.actualAcuteTriggeredBy,
    required this.actualAcuteMethod,
    required this.actualAcuteCalculationNotes,
    required this.calculatedAcuteNutritionPrediction,
    required this.calculatedAcuteTriggeredBy,
    required this.calculatedAcuteMethod,
    required this.calculatedAcuteScreeningOnly,
    required this.poshanSetuBmiStatus,
    required this.poshanSetuMuacStatus,
    required this.poshanSetuFinalStatus,
    required this.poshanSetuTriggeredBy,
    required this.poshanSetuComplete,
    required this.poshanSetuVersion,
    required this.storedOverallNutritionPrediction,
    required this.storedOverallPredictionMethod,
    required this.storedOverallPredictionConfidence,
    required this.storedOverallPredictionRationale,
    required this.previousMeasurementDate,
    required this.daysSincePreviousMeasurement,
    required this.actualHeightChangeCm,
    required this.actualWeightChangeKg,
    required this.actualMuacChangeCm,
    required this.bodyBuild,
    required this.estimationMethod,
    required this.sideViewUsed,
    required this.samProbability,
    required this.mamProbability,
    required this.normalProbability,
    required this.riskOverweightProbability,
    required this.overweightProbability,
    required this.visitNotes,
    required this.measurementNotes,
    required this.provenanceNotes,
  });

  static const headers = <String>[
    'export_schema_version',
    'child_name',
    'child_id',
    'guardian_name',
    'area',
    'sex',
    'date_of_birth',
    'measurement_date',
    'age_days',
    'entry_method',
    'measurement_mode',
    'position_adjustment_cm',
    'oedema',
    'who_standard_version',

    // Measured/actual evidence and outcomes.
    'actual_height_cm',
    'actual_height_method',
    'actual_weight_kg',
    'actual_weight_method',
    'actual_muac_cm',
    'actual_muac_status',
    'actual_muac_method',
    'actual_bmi',
    'actual_haz_zscore',
    'actual_stunting_classification',
    'actual_haz_quality_flag',
    'actual_whz_zscore',
    'actual_wasting_classification',
    'actual_whz_quality_flag',
    'actual_waz_zscore',
    'actual_underweight_classification',
    'actual_waz_quality_flag',
    'actual_baz_zscore',
    'actual_bmi_for_age_classification',
    'actual_baz_quality_flag',
    'actual_acute_malnutrition_classification',
    'actual_acute_triggered_by',
    'actual_acute_calculation_notes',

    // Calculated/estimated evidence and predictions.
    'calculated_height_cm',
    'calculated_height_method',
    'calculated_height_confidence',
    'calculated_weight_kg',
    'calculated_weight_method',
    'calculated_weight_confidence',
    'calculated_muac_cm',
    'calculated_muac_status',
    'calculated_muac_method',
    'calculated_muac_confidence',
    'calculated_muac_requires_confirmation',
    'calculated_bmi',
    'calculated_haz_zscore',
    'calculated_stunting_prediction',
    'calculated_haz_quality_flag',
    'calculated_whz_zscore',
    'calculated_wasting_prediction',
    'calculated_whz_quality_flag',
    'calculated_waz_zscore',
    'calculated_underweight_prediction',
    'calculated_waz_quality_flag',
    'calculated_baz_zscore',
    'calculated_bmi_for_age_prediction',
    'calculated_baz_quality_flag',
    'calculated_acute_malnutrition_prediction',
    'calculated_acute_triggered_by',
    'calculated_acute_screening_only',

    // Poshan Setu result and human-readable notes.
    'poshan_setu_bmi_status',
    'poshan_setu_muac_status',
    'poshan_setu_final_status',
    'poshan_setu_triggered_by',
    'poshan_setu_complete',
    'poshan_setu_version',
    'visit_notes',
    'measurement_notes',
    'provenance_notes',
    // Keep the technical visit key out of the main reading flow.
    'visit_uuid',
  ];

  final String exportSchemaVersion;
  final String childName;
  final int childId;
  final String? guardianName;
  final String? area;
  final String sex;
  final String dateOfBirth;
  final String measurementDate;
  final int? ageDays;
  final double recordedAgeMonths;
  final double? whoAgeMonths;
  final String visitUuid;
  final String entryMethod;
  final String? captureState;
  final String? consentVersion;
  final String? consentTimestamp;
  final String? consentOperatorIdentifier;
  final String? measurementMode;
  final String? whoExpectedMeasurementMode;
  final double? positionAdjustmentCm;
  final String? oedema;
  final String? oedemaGrade;
  final String? measuredAt;
  final String? measurementUpdateReason;
  final double? actualHeightCm;
  final String? actualHeightMethod;
  final double? actualWhoAdjustedHeightCm;
  final double? calculatedHeightCm;
  final String? calculatedHeightMethod;
  final double? calculatedHeightConfidence;
  final double? calculatedWhoAdjustedHeightCm;
  final double? heightErrorCm;
  final double? actualWeightKg;
  final String? actualWeightMethod;
  final double? calculatedWeightKg;
  final String? calculatedWeightMethod;
  final double? calculatedWeightConfidence;
  final double? weightErrorKg;
  final double? actualMuacCm;
  final String? actualMuacStatus;
  final String? actualMuacMethod;
  final bool? actualMuacIsDirectMeasurement;
  final bool? muacAgeInRange;
  final double? calculatedMuacCm;
  final String? calculatedMuacStatus;
  final String? calculatedMuacMethod;
  final double? calculatedMuacConfidence;
  final double? calculatedMuacUncertaintyLowerCm;
  final double? calculatedMuacUncertaintyUpperCm;
  final String? calculatedMuacModelVersion;
  final String? calculatedMuacCalibrationVersion;
  final bool? calculatedMuacRequiresConfirmation;
  final String? calculatedMuacReferralGuidance;
  final double? muacErrorCm;
  final double? actualBmi;
  final double? actualHazZscore;
  final String? actualStuntingClassification;
  final String actualHazQualityFlag;
  final double? actualWhzZscore;
  final String? actualWastingClassification;
  final String actualWhzQualityFlag;
  final double? actualWazZscore;
  final String? actualUnderweightClassification;
  final String actualWazQualityFlag;
  final double? actualBazZscore;
  final String? actualBmiForAgeClassification;
  final String actualBazQualityFlag;
  final String? actualWhoCalculationNotes;
  final double? calculatedBmi;
  final double? calculatedHazZscore;
  final String? calculatedStuntingPrediction;
  final String calculatedHazQualityFlag;
  final double? calculatedWhzZscore;
  final String? calculatedWastingPrediction;
  final String calculatedWhzQualityFlag;
  final double? calculatedWazZscore;
  final String? calculatedUnderweightPrediction;
  final String calculatedWazQualityFlag;
  final double? calculatedBazZscore;
  final String? calculatedBmiForAgePrediction;
  final String calculatedBazQualityFlag;
  final String? calculatedWhoCalculationNotes;
  final String whoStandardVersion;
  final String actualAcuteNutritionClassification;
  final String? actualAcuteTriggeredBy;
  final String actualAcuteMethod;
  final String? actualAcuteCalculationNotes;
  final String calculatedAcuteNutritionPrediction;
  final String? calculatedAcuteTriggeredBy;
  final String calculatedAcuteMethod;
  final bool calculatedAcuteScreeningOnly;
  final String poshanSetuBmiStatus;
  final String poshanSetuMuacStatus;
  final String poshanSetuFinalStatus;
  final String? poshanSetuTriggeredBy;
  final bool poshanSetuComplete;
  final String poshanSetuVersion;
  final String? storedOverallNutritionPrediction;
  final String? storedOverallPredictionMethod;
  final double? storedOverallPredictionConfidence;
  final String? storedOverallPredictionRationale;
  final String? previousMeasurementDate;
  final int? daysSincePreviousMeasurement;
  final double? actualHeightChangeCm;
  final double? actualWeightChangeKg;
  final double? actualMuacChangeCm;
  final String? bodyBuild;
  final String? estimationMethod;
  final bool? sideViewUsed;
  final double? samProbability;
  final double? mamProbability;
  final double? normalProbability;
  final double? riskOverweightProbability;
  final double? overweightProbability;
  final String? visitNotes;
  final String? measurementNotes;
  final String? provenanceNotes;

  List<Object?> toCsvRow() => <Object?>[
        exportSchemaVersion,
        childName,
        childId,
        guardianName,
        area,
        sex,
        dateOfBirth,
        measurementDate,
        ageDays,
        entryMethod,
        measurementMode,
        _formatNumber(positionAdjustmentCm),
        oedema,
        whoStandardVersion,
        _formatNumber(actualHeightCm),
        actualHeightMethod,
        _formatNumber(actualWeightKg),
        actualWeightMethod,
        _formatNumber(actualMuacCm),
        actualMuacStatus,
        actualMuacMethod,
        _formatNumber(actualBmi),
        _formatNumber(actualHazZscore),
        actualStuntingClassification,
        actualHazQualityFlag,
        _formatNumber(actualWhzZscore),
        actualWastingClassification,
        actualWhzQualityFlag,
        _formatNumber(actualWazZscore),
        actualUnderweightClassification,
        actualWazQualityFlag,
        _formatNumber(actualBazZscore),
        actualBmiForAgeClassification,
        actualBazQualityFlag,
        actualAcuteNutritionClassification,
        actualAcuteTriggeredBy,
        actualAcuteCalculationNotes,
        _formatNumber(calculatedHeightCm),
        calculatedHeightMethod,
        _formatNumber(calculatedHeightConfidence),
        _formatNumber(calculatedWeightKg),
        calculatedWeightMethod,
        _formatNumber(calculatedWeightConfidence),
        _formatNumber(calculatedMuacCm),
        calculatedMuacStatus,
        calculatedMuacMethod,
        _formatNumber(calculatedMuacConfidence),
        _formatBool(calculatedMuacRequiresConfirmation),
        _formatNumber(calculatedBmi),
        _formatNumber(calculatedHazZscore),
        calculatedStuntingPrediction,
        calculatedHazQualityFlag,
        _formatNumber(calculatedWhzZscore),
        calculatedWastingPrediction,
        calculatedWhzQualityFlag,
        _formatNumber(calculatedWazZscore),
        calculatedUnderweightPrediction,
        calculatedWazQualityFlag,
        _formatNumber(calculatedBazZscore),
        calculatedBmiForAgePrediction,
        calculatedBazQualityFlag,
        calculatedAcuteNutritionPrediction,
        calculatedAcuteTriggeredBy,
        _formatBool(calculatedAcuteScreeningOnly),
        poshanSetuBmiStatus,
        poshanSetuMuacStatus,
        poshanSetuFinalStatus,
        poshanSetuTriggeredBy,
        _formatBool(poshanSetuComplete),
        poshanSetuVersion,
        visitNotes,
        measurementNotes,
        provenanceNotes,
        visitUuid,
      ];

  static String? _formatNumber(num? value) {
    if (value == null) return null;
    final fixed = value.toDouble().toStringAsFixed(4);
    return fixed.replaceFirst(RegExp(r'\.?0+$'), '');
  }

  static String? _formatBool(bool? value) => value?.toString();
}
