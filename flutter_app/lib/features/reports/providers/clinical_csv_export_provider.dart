import 'dart:io';
import 'dart:ui';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../../providers/database_provider.dart';
import '../repositories/clinical_csv_export_repository.dart';
import '../services/clinical_csv_export_service.dart';

abstract interface class ClinicalCsvExportGateway {
  Future<ClinicalCsvExportFile> exportAndShare({
    required int ownerUserId,
    Rect? sharePositionOrigin,
  });
}

abstract interface class ClinicalCsvShareGateway {
  Future<void> share(
    ClinicalCsvExportFile export, {
    Rect? sharePositionOrigin,
  });
}

class SharePlusClinicalCsvGateway implements ClinicalCsvShareGateway {
  const SharePlusClinicalCsvGateway();

  @override
  Future<void> share(
    ClinicalCsvExportFile export, {
    Rect? sharePositionOrigin,
  }) async {
    await SharePlus.instance.share(
      ShareParams(
        title: 'Export clinical prediction records',
        subject: 'Clinical prediction records',
        text: '${export.recordCount} saved assessment/report records',
        files: [XFile(export.path, mimeType: 'text/csv')],
        fileNameOverrides: [export.fileName],
        sharePositionOrigin: sharePositionOrigin,
      ),
    );
  }
}

class ClinicalCsvExportCoordinator implements ClinicalCsvExportGateway {
  ClinicalCsvExportCoordinator({
    required ClinicalCsvExportService service,
    required ClinicalCsvShareGateway shareGateway,
    Future<Directory> Function()? documentsDirectory,
  })  : _service = service,
        _shareGateway = shareGateway,
        _documentsDirectory =
            documentsDirectory ?? getApplicationDocumentsDirectory;

  final ClinicalCsvExportService _service;
  final ClinicalCsvShareGateway _shareGateway;
  final Future<Directory> Function() _documentsDirectory;

  @override
  Future<ClinicalCsvExportFile> exportAndShare({
    required int ownerUserId,
    Rect? sharePositionOrigin,
  }) async {
    final documents = await _documentsDirectory();
    final export = await _service.exportAll(
      ownerUserId: ownerUserId,
      outputDirectory: Directory(p.join(documents.path, 'exports')),
    );
    await _shareGateway.share(
      export,
      sharePositionOrigin: sharePositionOrigin,
    );
    return export;
  }
}

final clinicalCsvExportRepositoryProvider =
    Provider<ClinicalCsvExportRepository>((ref) {
  return DriftClinicalCsvExportRepository(ref.watch(databaseProvider));
});

final clinicalCsvExportServiceProvider = Provider<ClinicalCsvExportService>(
  (ref) => ClinicalCsvExportService(
    ref.watch(clinicalCsvExportRepositoryProvider),
  ),
);

final clinicalCsvExportGatewayProvider = Provider<ClinicalCsvExportGateway>(
  (ref) => ClinicalCsvExportCoordinator(
    service: ref.watch(clinicalCsvExportServiceProvider),
    shareGateway: const SharePlusClinicalCsvGateway(),
  ),
);
