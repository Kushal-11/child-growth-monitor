import 'package:drift/drift.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../models/child.dart';
import '../models/child_detail.dart';
import 'database_provider.dart';

/// Watches all children from the local DB, with visit counts joined in.
final childrenProvider = StreamProvider<List<ChildSummary>>((ref) {
  final childDao = ref.watch(childDaoProvider);
  final db = ref.watch(databaseProvider);

  return childDao.watchAll().asyncMap((rows) async {
    return Future.wait(rows.map((c) async {
      final countExpr = db.visits.id.count();
      final visitCount = await (db.selectOnly(db.visits)
            ..addColumns([countExpr])
            ..where(db.visits.childId.equals(c.id)))
          .map((row) => row.read(countExpr) ?? 0)
          .getSingle();
      return ChildSummary(
        id: c.id,
        name: c.name,
        dateOfBirth: c.dateOfBirth,
        sex: c.sex,
        visitCount: visitCount,
      );
    }));
  });
});

/// Watches a single child + their visit history.
final childDetailProvider =
    StreamProvider.family<ChildDetail, int>((ref, childId) {
  final db = ref.watch(databaseProvider);
  final visitDao = ref.watch(visitDaoProvider);

  final childQuery = db.select(db.children)..where((t) => t.id.equals(childId));
  return childQuery.watchSingleOrNull().asyncMap((child) async {
    if (child == null) {
      throw StateError('Child $childId not found');
    }

    final visitRows = await visitDao.watchByChildId(childId).first;

    final visits = await Future.wait(visitRows.map((pair) async {
      final v = pair.visit;
      final m = pair.measurement;
      final cameraRows = await (db.select(db.cameraResults)
            ..where((row) => row.visitId.equals(v.id))
            ..orderBy([(row) => OrderingTerm.desc(row.version)])
            ..limit(1))
          .get();
      final assetRows = await (db.select(db.captureAssets)
            ..where(
              (row) =>
                  row.visitId.equals(v.id) &
                  row.qualityVerdict.equals('accepted'),
            ))
          .get();
      final requiredAssetAcknowledgement = {
        for (final role in const ['front', 'side'])
          role: _assetAcknowledgementState(assetRows, role),
      };
      final camera = cameraRows.isEmpty ? null : cameraRows.first;
      final displayedHeight =
          m?.manualHeightCm ?? m?.effectiveHeightCm ?? m?.predictedHeightCm;
      final displayedWeight =
          m?.manualWeightKg ?? m?.effectiveWeightKg ?? m?.predictedWeightKg;
      final heightMethod =
          m?.heightMethod ?? (m?.manualHeightCm != null ? 'manual' : null);
      final weightMethod =
          m?.weightMethod ?? (m?.manualWeightKg != null ? 'manual' : null);
      final heightIsDirect =
          heightMethod == 'manual' || heightMethod == 'reference_object';
      final weightIsDirect = weightMethod == 'manual';
      return ChildVisit(
        visitId: v.id,
        localUuid: v.localUuid,
        visitDate: v.visitDate.toIso8601String(),
        ageMonths: v.ageMonths,
        entryMethod: v.entryMethod,
        captureState: v.captureState,
        cameraResultSummary: camera == null
            ? null
            : CameraResultSummary(
                resultUuid: camera.resultUuid,
                version: camera.version,
                estimatedHeightCm: camera.estimatedHeightCm,
                estimatedWeightKg: camera.estimatedWeightKg,
                estimatedStuntingStatus: camera.estimatedStuntingStatus,
                estimatedWastingStatus: camera.estimatedWastingStatus,
                experimentalOverallCategory: camera.experimentalOverallCategory,
                method: camera.method,
                modelVersion: camera.modelVersion,
                nonClinical: camera.nonClinical,
              ),
        hasMeasuredReport: v.captureState == 'measured_report' && m != null,
        requiredAssetAcknowledgement: requiredAssetAcknowledgement,
        requiredAssetsAcknowledged: requiredAssetAcknowledgement.values
            .every((state) => state == 'acknowledged'),
        mediaDeletedAt: v.mediaDeletedAt?.toIso8601String(),
        measurement: m == null
            ? null
            : ChildVisitMeasurement(
                predictedHeightCm: displayedHeight,
                predictedWeightKg: displayedWeight,
                heightMethod: heightMethod,
                weightMethod: weightMethod,
                muacCm: m.muacCm,
                muacMethod: m.muacMethod,
                hazZscore: heightIsDirect ? m.hazZscore : null,
                whzZscore:
                    heightIsDirect && weightIsDirect ? m.whzZscore : null,
                hazStatus: heightIsDirect ? m.hazStatus : null,
                whzStatus:
                    heightIsDirect && weightIsDirect ? m.whzStatus : null,
                confidenceScore: m.confidenceScore,
              ),
      );
    }));

    return ChildDetail(
      id: child.id,
      name: child.name,
      dateOfBirth: child.dateOfBirth,
      sex: child.sex,
      guardianName: child.guardianName,
      location: child.location,
      visits: visits,
    );
  });
});

String _assetAcknowledgementState(
  List<CaptureAsset> assets,
  String role,
) {
  final matching = assets.where((asset) => asset.role == role);
  if (matching.any(
    (asset) =>
        asset.serverAcknowledgedAt != null || asset.syncState == 'synced',
  )) {
    return 'acknowledged';
  }
  return matching.isEmpty ? 'missing' : 'pending';
}
