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
  test('writes the exact clinical predictions header and RFC-safe rows',
      () async {
    final repository = _FakeRepository([
      const ClinicalCsvRecord(
        childId: 1,
        area: 'Clinic, East',
        sex: 'F',
        dateOfBirth: '2022-09-02',
        measurementDate: '2026-06-12',
        actualHeightCm: 91,
        calculatedHeightCm: 100.9,
        actualWeightKg: 10.7,
        calculatedWeightKg: 16.16,
        muacCm: 13.5,
        calculatedMuacCm: 16,
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
    expect(rows.last, hasLength(ClinicalCsvRecord.headers.length));
    expect(rows.last[0], '1');
    expect(rows.last[1], 'Clinic, East');
    expect(rows.last[5], '91');
    expect(rows.last[6], '100.9');
    expect(rows.last[11], '');
    expect(rows.last[15], contains('"recheck"'));
  });
}
