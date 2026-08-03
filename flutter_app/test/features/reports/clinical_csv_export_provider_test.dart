import 'dart:io';
import 'dart:ui';

import 'package:child_growth_monitor_app/features/reports/domain/clinical_csv_record.dart';
import 'package:child_growth_monitor_app/features/reports/providers/clinical_csv_export_provider.dart';
import 'package:child_growth_monitor_app/features/reports/repositories/clinical_csv_export_repository.dart';
import 'package:child_growth_monitor_app/features/reports/services/clinical_csv_export_service.dart';
import 'package:flutter_test/flutter_test.dart';

class _EmptyRepository implements ClinicalCsvExportRepository {
  @override
  Future<List<ClinicalCsvRecord>> loadSavedRecords({
    required int ownerUserId,
  }) async =>
      [];
}

class _RecordingShareGateway implements ClinicalCsvShareGateway {
  ClinicalCsvExportFile? shared;

  @override
  Future<void> share(
    ClinicalCsvExportFile export, {
    Rect? sharePositionOrigin,
  }) async {
    shared = export;
  }
}

void main() {
  test('coordinator keeps a local export and sends it to the share gateway',
      () async {
    final documents =
        await Directory.systemTemp.createTemp('cgm-csv-coordinator-');
    addTearDown(() => documents.delete(recursive: true));
    final shareGateway = _RecordingShareGateway();
    final coordinator = ClinicalCsvExportCoordinator(
      service: ClinicalCsvExportService(
        _EmptyRepository(),
        now: () => DateTime(2026, 8, 3, 9),
      ),
      shareGateway: shareGateway,
      documentsDirectory: () async => documents,
    );

    final export = await coordinator.exportAndShare(ownerUserId: 7);

    expect(export.recordCount, 0);
    expect(shareGateway.shared?.path, export.path);
    expect(export.path, contains('${Platform.pathSeparator}exports'));
    expect(await File(export.path).exists(), isTrue);
  });
}
