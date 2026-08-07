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
  'entry_method',
  'measurement_mode',
  'position_adjustment_cm',
  'oedema',
  'who_standard_version',
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
  test(
    'writes the complete typed header and RFC-safe rows',
    () async {
      final repository = _FakeRepository([
        const ClinicalCsvRecord(
          exportSchemaVersion: 'clinical_csv_v3',
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
          heightErrorCm: 9.9,
          actualWeightKg: 10.7,
          actualWeightMethod: 'manual',
          calculatedWeightKg: 16.16,
          calculatedWeightMethod: 'ml_estimated',
          calculatedWeightConfidence: 0.82,
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
      final directory = await Directory.systemTemp.createTemp(
        'cgm-csv-export-',
      );
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
      expect(ClinicalCsvRecord.headers, hasLength(74));
      expect(ClinicalCsvRecord.headers.last, 'visit_uuid');
      expect(ClinicalCsvRecord.headers, isNot(contains('field_category')));
      expect(ClinicalCsvRecord.headers, isNot(contains('haz_zscore')));
      expect(ClinicalCsvRecord.headers, isNot(contains('whz_zscore')));
      expect(
        ClinicalCsvRecord.headers,
        isNot(contains('stored_overall_nutrition_prediction')),
      );
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
      expect(value('actual_stunting_classification'), 'Not Stunted');
      expect(value('actual_wasting_classification'), 'Moderately Wasted');
      expect(value('actual_underweight_classification'), 'Underweight');
      expect(value('calculated_stunting_prediction'), 'Not Stunted');
      expect(value('calculated_wasting_prediction'), 'Normal');
      expect(value('visit_notes'), contains('"recheck"'));
      expect(value('visit_uuid'), '10000000-0000-0000-0000-000000000001');
    },
  );
}
