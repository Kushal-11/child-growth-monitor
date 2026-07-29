import 'dart:convert';

import 'package:drift/drift.dart';

import '../database.dart';
import 'sync_outbox_dao.dart';

class MeasuredDetailRevisionDao {
  MeasuredDetailRevisionDao(this._db);
  final AppDatabase _db;

  Future<Measurement> saveMeasuredReport({
    required int ownerUserId,
    required String visitUuid,
    required String revisionUuid,
    required String beforeJson,
    required String afterJson,
    required MeasurementsCompanion measurement,
    required String payloadJson,
    String? reason,
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
      if (visit.captureState != 'estimated_report' &&
          visit.captureState != 'measured_report') {
        throw StateError(
          'Measured details require an estimated or measured report',
        );
      }

      final existing = await (_db.select(_db.measurements)
            ..where((row) => row.visitId.equals(visit.id)))
          .getSingleOrNull();
      if (existing == null) {
        await _db.into(_db.measurements).insert(
              measurement.copyWith(visitId: Value(visit.id)),
            );
      } else {
        await (_db.update(_db.measurements)
              ..where((row) => row.id.equals(existing.id)))
            .write(measurement.copyWith(visitId: Value(visit.id)));
      }

      final maxRevision = _db.measuredDetailRevisions.revisionNumber.max();
      final latest = await (_db.selectOnly(_db.measuredDetailRevisions)
            ..addColumns([maxRevision])
            ..where(_db.measuredDetailRevisions.visitId.equals(visit.id)))
          .map((row) => row.read(maxRevision))
          .getSingle();
      await _db.into(_db.measuredDetailRevisions).insert(
            MeasuredDetailRevisionsCompanion.insert(
              revisionUuid: revisionUuid,
              visitId: visit.id,
              revisionNumber: (latest ?? 0) + 1,
              beforeJson: beforeJson,
              afterJson: afterJson,
              editorUserId: Value(ownerUserId),
              reason: Value(reason),
            ),
          );
      await (_db.update(_db.visits)..where((row) => row.id.equals(visit.id)))
          .write(
        const VisitsCompanion(captureState: Value('measured_report')),
      );
      final visitOutbox = await (_db.select(_db.syncOutbox)
            ..where(
              (row) =>
                  row.ownerUserId.equals(ownerUserId) &
                  row.entityType.equals(SyncOutboxEntityType.visit) &
                  row.entityUuid.equals(visitUuid),
            ))
          .getSingleOrNull();
      if (visitOutbox == null) {
        throw StateError('Visit outbox record was not found');
      }
      final visitPayload =
          jsonDecode(visitOutbox.payloadJson) as Map<String, dynamic>;
      visitPayload['capture_state'] = 'measured_report';
      await SyncOutboxDao(_db).refreshPayload(
        ownerUserId: ownerUserId,
        entityType: SyncOutboxEntityType.visit,
        entityUuid: visitUuid,
        payloadJson: jsonEncode(visitPayload),
      );
      await SyncOutboxDao(_db).enqueue(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        entityType: SyncOutboxEntityType.measuredRevision,
        entityUuid: revisionUuid,
        dependencyEntityUuid: visitUuid,
        payloadJson: payloadJson,
      );
      return (_db.select(_db.measurements)
            ..where((row) => row.visitId.equals(visit.id)))
          .getSingle();
    });
  }

  Future<List<MeasuredDetailRevision>> getForVisit({
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
    return (_db.select(_db.measuredDetailRevisions)
          ..where((row) => row.visitId.equals(visit.id))
          ..orderBy([(row) => OrderingTerm.asc(row.revisionNumber)]))
        .get();
  }
}
