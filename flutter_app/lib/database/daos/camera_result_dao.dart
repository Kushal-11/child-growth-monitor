import 'package:drift/drift.dart';

import '../database.dart';
import 'sync_outbox_dao.dart';

class CameraResultDao {
  CameraResultDao(this._db);
  final AppDatabase _db;

  Future<CameraResult> appendCameraResult({
    required int ownerUserId,
    required String visitUuid,
    required CameraResultsCompanion result,
    required String payloadJson,
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
      if (!result.resultUuid.present ||
          !result.version.present ||
          (result.nonClinical.present && !result.nonClinical.value)) {
        throw ArgumentError(
            'A non-clinical result UUID and version are required');
      }
      final existing = await (_db.select(_db.cameraResults)
            ..where(
              (row) => row.resultUuid.equals(result.resultUuid.value),
            ))
          .getSingleOrNull();
      if (existing != null) {
        throw StateError(
          'Camera results are immutable; append a new UUID and version',
        );
      }
      if (visit.captureState != 'processing' &&
          visit.captureState != 'measured_report') {
        throw StateError(
          'Camera results require processing or measured reprocessing state',
        );
      }
      final maxVersion = _db.cameraResults.version.max();
      final latest = await (_db.selectOnly(_db.cameraResults)
            ..addColumns([maxVersion])
            ..where(_db.cameraResults.visitId.equals(visit.id)))
          .map((row) => row.read(maxVersion))
          .getSingle();
      final expectedVersion = (latest ?? 0) + 1;
      if (result.version.value != expectedVersion) {
        throw StateError('Camera result version must be $expectedVersion');
      }

      final id = await _db.into(_db.cameraResults).insert(
            result.copyWith(visitId: Value(visit.id)),
          );
      if (visit.captureState == 'processing') {
        await (_db.update(_db.visits)..where((row) => row.id.equals(visit.id)))
            .write(
          const VisitsCompanion(
            captureState: Value('estimated_report'),
          ),
        );
      }
      await SyncOutboxDao(_db).enqueue(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        entityType: SyncOutboxEntityType.cameraResult,
        entityUuid: result.resultUuid.value,
        dependencyEntityUuid: visitUuid,
        payloadJson: payloadJson,
      );
      return (_db.select(_db.cameraResults)..where((row) => row.id.equals(id)))
          .getSingle();
    });
  }

  Future<List<CameraResult>> getVersions({
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
    return (_db.select(_db.cameraResults)
          ..where((row) => row.visitId.equals(visit.id))
          ..orderBy([(row) => OrderingTerm.asc(row.version)]))
        .get();
  }
}
