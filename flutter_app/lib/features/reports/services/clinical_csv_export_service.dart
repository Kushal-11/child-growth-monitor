import 'dart:io';

import 'package:csv/csv.dart';
import 'package:intl/intl.dart';
import 'package:path/path.dart' as p;

import '../domain/clinical_csv_record.dart';
import '../repositories/clinical_csv_export_repository.dart';

class ClinicalCsvExportFile {
  const ClinicalCsvExportFile({
    required this.path,
    required this.fileName,
    required this.recordCount,
  });

  final String path;
  final String fileName;
  final int recordCount;
}

class ClinicalCsvExportService {
  ClinicalCsvExportService(
    this._repository, {
    DateTime Function()? now,
  }) : _now = now ?? DateTime.now;

  final ClinicalCsvExportRepository _repository;
  final DateTime Function() _now;

  Future<ClinicalCsvExportFile> exportAll({
    required int ownerUserId,
    required Directory outputDirectory,
  }) async {
    final records = await _repository.loadSavedRecords(
      ownerUserId: ownerUserId,
    );
    await outputDirectory.create(recursive: true);
    final timestamp = DateFormat('yyyyMMdd_HHmmss').format(_now());
    final fileName = 'clinical_predictions_$timestamp.csv';
    final file = File(p.join(outputDirectory.path, fileName));
    await file.writeAsString(buildCsv(records), flush: true);
    return ClinicalCsvExportFile(
      path: file.path,
      fileName: fileName,
      recordCount: records.length,
    );
  }

  String buildCsv(List<ClinicalCsvRecord> records) {
    final rows = <List<Object?>>[
      ClinicalCsvRecord.headers,
      ...records.map((record) => record.toCsvRow()),
    ];
    return const ListToCsvConverter(eol: '\n', convertNullTo: '').convert(rows);
  }
}
