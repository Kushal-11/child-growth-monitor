import 'package:drift/drift.dart';

import '../database.dart';
import 'sync_outbox_dao.dart';

class AcceptedCaptureAsset {
  const AcceptedCaptureAsset({
    required this.assetUuid,
    required this.role,
    required this.localPath,
    required this.capturedAt,
    required this.payloadJson,
    this.selectedRank,
    this.poseScore,
    this.coverageScore,
    this.orientationScore,
    this.sharpnessScore,
    this.lightingScore,
    this.overallScore,
    this.qualityThresholdVersion,
    this.imageWidth,
    this.imageHeight,
    this.exifOrientation,
    this.displayOrientation,
    this.deviceCameraMetadataJson,
  });

  final String assetUuid;
  final String role;
  final String localPath;
  final DateTime capturedAt;
  final String payloadJson;
  final int? selectedRank;
  final double? poseScore;
  final double? coverageScore;
  final double? orientationScore;
  final double? sharpnessScore;
  final double? lightingScore;
  final double? overallScore;
  final String? qualityThresholdVersion;
  final int? imageWidth;
  final int? imageHeight;
  final int? exifOrientation;
  final int? displayOrientation;
  final String? deviceCameraMetadataJson;
}

class CaptureAssetDao {
  CaptureAssetDao(this._db);
  final AppDatabase _db;

  Future<List<CaptureAsset>> saveAcceptedAssets({
    required int ownerUserId,
    required String visitUuid,
    required List<AcceptedCaptureAsset> assets,
  }) {
    return _db.transaction(() async {
      final visit = await (_db.select(_db.visits)
            ..where(
              (row) =>
                  row.localUuid.equals(visitUuid) &
                  row.ownerUserId.equals(ownerUserId),
            ))
          .getSingleOrNull();
      if (visit == null) {
        throw StateError('Owner-scoped visit was not found');
      }
      if (assets.isEmpty) {
        throw ArgumentError.value(assets, 'assets', 'must not be empty');
      }

      final insertedIds = <int>[];
      for (final asset in assets) {
        final id = await _db.into(_db.captureAssets).insert(
              CaptureAssetsCompanion.insert(
                assetUuid: asset.assetUuid,
                visitId: visit.id,
                role: asset.role,
                localPath: Value(asset.localPath),
                capturedAt: asset.capturedAt,
                selectedRank: Value(asset.selectedRank),
                poseScore: Value(asset.poseScore),
                coverageScore: Value(asset.coverageScore),
                orientationScore: Value(asset.orientationScore),
                sharpnessScore: Value(asset.sharpnessScore),
                lightingScore: Value(asset.lightingScore),
                overallScore: Value(asset.overallScore),
                qualityVerdict: const Value('accepted'),
                qualityThresholdVersion: Value(asset.qualityThresholdVersion),
                imageWidth: Value(asset.imageWidth),
                imageHeight: Value(asset.imageHeight),
                exifOrientation: Value(asset.exifOrientation),
                displayOrientation: Value(asset.displayOrientation),
                deviceCameraMetadataJson: Value(asset.deviceCameraMetadataJson),
              ),
            );
        insertedIds.add(id);
        await SyncOutboxDao(_db).enqueue(
          ownerUserId: ownerUserId,
          visitUuid: visitUuid,
          entityType: SyncOutboxEntityType.captureAsset,
          entityUuid: asset.assetUuid,
          dependencyEntityUuid: visitUuid,
          payloadJson: asset.payloadJson,
        );
      }
      return (_db.select(_db.captureAssets)
            ..where((row) => row.id.isIn(insertedIds))
            ..orderBy([(row) => OrderingTerm.asc(row.capturedAt)]))
          .get();
    });
  }

  Future<List<CaptureAsset>> getForVisit({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    final visit = await (_db.select(_db.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingleOrNull();
    if (visit == null) return [];
    return (_db.select(_db.captureAssets)
          ..where((row) => row.visitId.equals(visit.id))
          ..orderBy([(row) => OrderingTerm.asc(row.capturedAt)]))
        .get();
  }
}
