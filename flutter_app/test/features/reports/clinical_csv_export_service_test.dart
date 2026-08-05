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

void main() {
  test(
    'writes the expanded clinical predictions header and RFC-safe rows',
    () async {
      final repository = _FakeRepository([
        const ClinicalCsvRecord(
          childId: 1,
          childName: 'Child "A", One',
          area: 'Clinic, East',
          sex: 'F',
          dateOfBirth: '2022-09-02',
          measurementDate: '2026-06-12',
          actualHeightCm: 91,
          calculatedHeightCm: 100.9,
          calculatedHeightMethod: 'reference_object',
          actualWeightKg: 10.7,
          calculatedWeightKg: 16.16,
          calculatedWeightMethod: 'ml_estimated',
          muacCm: 13.5,
          muacStatus: 'MAM',
          muacMethod: 'manual',
          muacAgeInRange: true,
          muacIsDirectMeasurement: true,
          calculatedMuacCm: 16,
          calculatedMuacMethod: 'landmark_estimated',
          muacConfidence: 0.98765,
          muacUncertaintyLowerCm: 12.8,
          muacUncertaintyUpperCm: 14.2,
          muacModelVersion: 'landmark-ratio-v1',
          muacCalibrationVersion: 'direct-tape',
          muacRequiresConfirmation: false,
          muacReferralGuidance: 'Recheck with tape, then refer if confirmed',
          hazZscore: -1.23456,
          whzZscore: -2.45,
          fieldCategory: null,
          predictedFieldCategory: 'Normal',
          stuntingPrediction: 'Normal',
          wastingPrediction: 'Normal',
          notes: 'Tape confirmation required; operator said "recheck"',
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
      expect(rows.last, hasLength(ClinicalCsvRecord.headers.length));
      Object? value(String header) =>
          rows.last[ClinicalCsvRecord.headers.indexOf(header)];
      expect(value('child_id'), '1');
      expect(value('child_name'), 'Child "A", One');
      expect(value('area'), 'Clinic, East');
      expect(value('actual_height_cm'), '91');
      expect(value('calculated_height_cm'), '100.9');
      expect(value('calculated_height_method'), 'reference_object');
      expect(value('calculated_weight_method'), 'ml_estimated');
      expect(value('calculated_muac_method'), 'landmark_estimated');
      expect(value('muac_age_in_range'), 'true');
      expect(value('calculated_muac_confidence'), '0.9877');
      expect(value('muac_is_direct_measurement'), 'true');
      expect(value('calculated_muac_requires_confirmation'), 'false');
      expect(value('haz_zscore'), '-1.2346');
      expect(value('whz_zscore'), '-2.45');
      expect(value('field_category'), '');
      expect(value('notes'), contains('"recheck"'));
    },
  );
}
