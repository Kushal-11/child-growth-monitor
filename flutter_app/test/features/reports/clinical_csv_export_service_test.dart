import 'dart:io';

import 'package:child_growth_monitor_app/features/reports/domain/clinical_csv_record.dart';
import 'package:child_growth_monitor_app/features/reports/repositories/clinical_csv_export_repository.dart';
import 'package:child_growth_monitor_app/features/reports/services/clinical_csv_export_service.dart';
import 'package:csv/csv.dart';
import 'package:flutter_test/flutter_test.dart';

class _FakeRepository implements ClinicalCsvExportRepository {
  _FakeRepository(this.records);

  final List<ClinicalCsvRecord> records;
  int? requestedOwner;

  @override
  Future<List<ClinicalCsvRecord>> loadSavedRecords({
    required int ownerUserId,
  }) async {
    requestedOwner = ownerUserId;
    return records;
  }
}

const _expectedClinicalHeaders = <String>[
  'export_schema_version',
  'child_name',
  'child_id',
  'guardian_name',
  'area',
  'sex',
  'date_of_birth',
  'measurement_date',
  'age_days',
  'recorded_age_months',
  'who_age_months',
  'entry_method',
  'capture_state',
  'consent_version',
  'consent_timestamp',
  'consent_operator_identifier',
  'measurement_mode',
  'who_expected_measurement_mode',
  'position_adjustment_cm',
  'oedema',
  'oedema_grade',
  'measured_at',
  'measurement_update_reason',
  'who_standard_version',
  'actual_height_cm',
  'actual_height_method',
  'actual_who_adjusted_height_cm',
  'actual_weight_kg',
  'actual_weight_method',
  'actual_muac_cm',
  'actual_muac_status',
  'actual_muac_method',
  'actual_muac_is_direct_measurement',
  'muac_age_in_range',
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
  'actual_who_calculation_notes',
  'actual_acute_malnutrition_classification',
  'actual_acute_triggered_by',
  'actual_acute_method',
  'actual_acute_calculation_notes',
  'calculated_height_cm',
  'calculated_height_method',
  'calculated_height_confidence',
  'calculated_who_adjusted_height_cm',
  'calculated_height_availability',
  'height_error_cm',
  'calculated_weight_kg',
  'calculated_weight_method',
  'calculated_weight_confidence',
  'calculated_weight_availability',
  'weight_error_kg',
  'calculated_muac_cm',
  'calculated_muac_status',
  'calculated_muac_method',
  'calculated_muac_confidence',
  'calculated_muac_uncertainty_lower_cm',
  'calculated_muac_uncertainty_upper_cm',
  'calculated_muac_model_version',
  'calculated_muac_calibration_version',
  'calculated_muac_requires_confirmation',
  'calculated_muac_referral_guidance',
  'calculated_muac_availability',
  'muac_error_cm',
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
  'calculated_who_calculation_notes',
  'calculated_acute_malnutrition_prediction',
  'calculated_acute_triggered_by',
  'calculated_acute_method',
  'calculated_acute_screening_only',
  'arcore_scan_available',
  'arcore_method',
  'arcore_depth_height_cm',
  'arcore_height_uncertainty_cm',
  'arcore_height_range_lower_cm',
  'arcore_height_range_upper_cm',
  'arcore_geometry_ml_weight_kg',
  'arcore_weight_range_lower_kg',
  'arcore_weight_range_upper_kg',
  'arcore_arm_muac_cm',
  'arcore_muac_uncertainty_cm',
  'arcore_muac_range_lower_cm',
  'arcore_muac_range_upper_cm',
  'arcore_quality_score',
  'arcore_geometry_quality_score',
  'arcore_pose_quality_score',
  'arcore_accepted_keyframes',
  'arcore_depth_confidence',
  'arcore_coverage_degrees',
  'arcore_floor_stability_cm',
  'arcore_shoulder_width_cm',
  'arcore_hip_width_cm',
  'arcore_torso_length_cm',
  'arcore_upper_arm_length_cm',
  'arcore_chest_depth_cm',
  'arcore_abdomen_depth_cm',
  'ml_estimated_weight_kg',
  'ml_weight_accepted_for_calculation',
  'ml_wasting_prediction',
  'ml_wasting_method',
  'sam_probability',
  'mam_probability',
  'normal_probability',
  'risk_overweight_probability',
  'overweight_probability',
  'stored_overall_nutrition_prediction',
  'stored_overall_prediction_method',
  'stored_overall_prediction_confidence',
  'stored_overall_prediction_rationale',
  'body_build',
  'estimation_method',
  'side_view_used',
  'previous_measurement_date',
  'days_since_previous_measurement',
  'actual_height_change_cm',
  'actual_weight_change_kg',
  'actual_muac_change_cm',
  'poshan_setu_bmi_status',
  'poshan_setu_muac_status',
  'poshan_setu_final_status',
  'poshan_setu_triggered_by',
  'poshan_setu_complete',
  'poshan_setu_version',
  'visit_notes',
  'measurement_notes',
  'provenance_notes',
  'visit_uuid',
];

void main() {
  test('writes the complete typed header and RFC-safe rows', () async {
    final repository = _FakeRepository([
      const ClinicalCsvRecord(
        exportSchemaVersion: 'clinical_csv_v5_arcore_recovery',
        childName: 'Child "A", One',
        childId: 1,
        guardianName: 'Guardian One',
        area: 'Clinic, East',
        sex: 'F',
        dateOfBirth: '2022-09-02',
        measurementDate: '2026-06-12',
        ageDays: 1379,
        recordedAgeMonths: 45.3,
        whoAgeMonths: 45.31,
        visitUuid: '10000000-0000-0000-0000-000000000001',
        entryMethod: 'guided_capture',
        captureState: 'measured_report',
        consentVersion: 'photo-consent-v1',
        consentTimestamp: '2026-06-12T10:00:00.000',
        consentOperatorIdentifier: 'field.worker',
        measurementMode: 'standing_height',
        whoExpectedMeasurementMode: 'standing_height',
        positionAdjustmentCm: 0,
        oedema: 'No',
        oedemaGrade: null,
        measuredAt: '2026-06-12T10:30:00.000',
        measurementUpdateReason: 'Tape measurement added',
        actualHeightCm: 91,
        actualHeightMethod: 'manual',
        actualWhoAdjustedHeightCm: 91,
        calculatedHeightCm: 100.9,
        calculatedHeightMethod: 'reference_object',
        calculatedHeightConfidence: 0.91,
        calculatedWhoAdjustedHeightCm: 100.9,
        calculatedHeightAvailability: 'available',
        heightErrorCm: 9.9,
        actualWeightKg: 10.7,
        actualWeightMethod: 'manual',
        calculatedWeightKg: 16.16,
        calculatedWeightMethod: 'ml_estimated',
        calculatedWeightConfidence: 0.82,
        calculatedWeightAvailability: 'available',
        weightErrorKg: 5.46,
        actualMuacCm: 13.5,
        actualMuacStatus: 'MAM',
        actualMuacMethod: 'tape',
        actualMuacIsDirectMeasurement: true,
        muacAgeInRange: true,
        calculatedMuacCm: 16,
        calculatedMuacStatus: 'Normal',
        calculatedMuacMethod: 'landmark_estimated',
        calculatedMuacConfidence: 0.98765,
        calculatedMuacUncertaintyLowerCm: 12.8,
        calculatedMuacUncertaintyUpperCm: 14.2,
        calculatedMuacModelVersion: 'landmark-ratio-v1',
        calculatedMuacCalibrationVersion: 'direct-tape',
        calculatedMuacRequiresConfirmation: false,
        calculatedMuacReferralGuidance:
            'Recheck with tape, then refer if confirmed',
        calculatedMuacAvailability: 'available',
        muacErrorCm: 2.5,
        actualBmi: 12.92,
        actualHazZscore: -1.23456,
        actualStuntingClassification: 'Not Stunted',
        actualHazQualityFlag: 'OK',
        actualWhzZscore: -2.45,
        actualWastingClassification: 'Moderately Wasted',
        actualWhzQualityFlag: 'OK',
        actualWazZscore: -2.15,
        actualUnderweightClassification: 'Underweight',
        actualWazQualityFlag: 'OK',
        actualBazZscore: -2.3,
        actualBmiForAgeClassification: 'Low BMI-for-Age',
        actualBazQualityFlag: 'OK',
        actualWhoCalculationNotes: null,
        calculatedBmi: 15.87,
        calculatedHazZscore: 0.12,
        calculatedStuntingPrediction: 'Not Stunted',
        calculatedHazQualityFlag: 'OK',
        calculatedWhzZscore: 0.35,
        calculatedWastingPrediction: 'Normal',
        calculatedWhzQualityFlag: 'OK',
        calculatedWazZscore: 0.25,
        calculatedUnderweightPrediction: 'Not Underweight',
        calculatedWazQualityFlag: 'OK',
        calculatedBazZscore: 0.42,
        calculatedBmiForAgePrediction: 'Normal',
        calculatedBazQualityFlag: 'OK',
        calculatedWhoCalculationNotes: null,
        whoStandardVersion: 'WHO Child Growth Standards 2006',
        actualAcuteNutritionClassification: 'MAM',
        actualAcuteTriggeredBy: '["whz"]',
        actualAcuteMethod: 'who_measured_whz_muac_oedema_v1',
        actualAcuteCalculationNotes: null,
        calculatedAcuteNutritionPrediction: 'No Acute Malnutrition',
        calculatedAcuteTriggeredBy: null,
        calculatedAcuteMethod: 'who_calculated_whz_muac_screening_v1',
        calculatedAcuteScreeningOnly: true,
        arcoreScanAvailable: true,
        arcoreMethod: 'arcore_contactless_anthropometry_v3',
        arcoreDepthHeightCm: 100.9,
        arcoreHeightUncertaintyCm: 0.7,
        arcoreHeightRangeLowerCm: 100.2,
        arcoreHeightRangeUpperCm: 101.6,
        arcoreGeometryMlWeightKg: 16.16,
        arcoreWeightRangeLowerKg: 15.5,
        arcoreWeightRangeUpperKg: 16.8,
        arcoreArmMuacCm: 16,
        arcoreMuacUncertaintyCm: 0.4,
        arcoreMuacRangeLowerCm: 15.6,
        arcoreMuacRangeUpperCm: 16.4,
        arcoreQualityScore: 0.91,
        arcoreGeometryQualityScore: 0.88,
        arcorePoseQualityScore: 0.9,
        arcoreAcceptedKeyframes: 20,
        arcoreDepthConfidence: 0.82,
        arcoreCoverageDegrees: 91,
        arcoreFloorStabilityCm: 1.2,
        arcoreShoulderWidthCm: 24,
        arcoreHipWidthCm: 22,
        arcoreTorsoLengthCm: 31,
        arcoreUpperArmLengthCm: 17,
        arcoreChestDepthCm: 14,
        arcoreAbdomenDepthCm: 13,
        poshanSetuBmiStatus: 'MAM',
        poshanSetuMuacStatus: 'Normal',
        poshanSetuFinalStatus: 'MAM',
        poshanSetuTriggeredBy: '["bmi"]',
        poshanSetuComplete: true,
        poshanSetuVersion: 'poshan_setu_v1',
        storedOverallNutritionPrediction: 'MAM',
        storedOverallPredictionMethod: 'poshan_setu_v1',
        storedOverallPredictionConfidence: 0.89,
        storedOverallPredictionRationale: 'MAM flagged by WHZ',
        previousMeasurementDate: null,
        daysSincePreviousMeasurement: null,
        actualHeightChangeCm: null,
        actualWeightChangeKg: null,
        actualMuacChangeCm: null,
        bodyBuild: 'average',
        estimationMethod: 'reference_object',
        sideViewUsed: true,
        mlEstimatedWeightKg: 16.16,
        mlWeightAcceptedForCalculation: true,
        mlWastingPrediction: 'MAM',
        mlWastingMethod: 'ml_classifier',
        samProbability: 0.1,
        mamProbability: 0.7,
        normalProbability: 0.15,
        riskOverweightProbability: 0.03,
        overweightProbability: 0.02,
        visitNotes: 'Operator said "recheck"',
        measurementNotes: 'Tape confirmation required',
        provenanceNotes: 'camera_non_clinical=true',
      ),
    ]);
    final service = ClinicalCsvExportService(
      repository,
      now: () => DateTime(2026, 8, 3, 14, 5, 6),
    );
    final directory = await Directory.systemTemp.createTemp('cgm-csv-export-');
    addTearDown(() => directory.delete(recursive: true));

    final export = await service.exportAll(
      ownerUserId: 7,
      outputDirectory: directory,
    );

    expect(repository.requestedOwner, 7);
    expect(export.fileName, 'clinical_predictions_20260803_140506.csv');
    expect(export.recordCount, 1);
    final contents = await File(export.path).readAsString();
    final rows = const CsvToListConverter(
      eol: '\n',
      shouldParseNumbers: false,
    ).convert(contents);
    expect(rows, hasLength(2));
    expect(rows.first, ClinicalCsvRecord.headers);
    expect(ClinicalCsvRecord.headers, _expectedClinicalHeaders);
    expect(ClinicalCsvRecord.headers, hasLength(150));
    expect(ClinicalCsvRecord.headers.last, 'visit_uuid');
    expect(ClinicalCsvRecord.headers, isNot(contains('field_category')));
    expect(ClinicalCsvRecord.headers, isNot(contains('haz_zscore')));
    expect(ClinicalCsvRecord.headers, isNot(contains('whz_zscore')));
    expect(rows.last, hasLength(ClinicalCsvRecord.headers.length));
    Object? value(String header) =>
        rows.last[ClinicalCsvRecord.headers.indexOf(header)];
    expect(value('child_name'), 'Child "A", One');
    expect(value('child_id'), '1');
    expect(value('guardian_name'), 'Guardian One');
    expect(value('age_days'), '1379');
    expect(value('actual_height_cm'), '91');
    expect(value('calculated_height_cm'), '100.9');
    expect(value('actual_weight_kg'), '10.7');
    expect(value('calculated_weight_kg'), '16.16');
    expect(value('actual_muac_cm'), '13.5');
    expect(value('calculated_muac_cm'), '16');
    expect(value('calculated_muac_confidence'), '0.9877');
    expect(value('height_error_cm'), '9.9');
    expect(value('weight_error_kg'), '5.46');
    expect(value('muac_error_cm'), '2.5');
    expect(value('arcore_scan_available'), 'true');
    expect(value('arcore_depth_height_cm'), '100.9');
    expect(value('arcore_geometry_ml_weight_kg'), '16.16');
    expect(value('arcore_arm_muac_cm'), '16');
    expect(value('ml_estimated_weight_kg'), '16.16');
    expect(value('ml_wasting_prediction'), 'MAM');
    expect(value('sam_probability'), '0.1');
    expect(value('mam_probability'), '0.7');
    expect(value('stored_overall_nutrition_prediction'), 'MAM');
    expect(value('actual_stunting_classification'), 'Not Stunted');
    expect(value('actual_wasting_classification'), 'Moderately Wasted');
    expect(value('actual_underweight_classification'), 'Underweight');
    expect(value('calculated_stunting_prediction'), 'Not Stunted');
    expect(value('calculated_wasting_prediction'), 'Normal');
    expect(value('visit_notes'), contains('"recheck"'));
    expect(value('visit_uuid'), '10000000-0000-0000-0000-000000000001');
  });
}
