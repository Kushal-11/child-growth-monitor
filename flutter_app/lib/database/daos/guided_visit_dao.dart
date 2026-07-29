import 'dart:convert';

import 'package:drift/drift.dart';

import '../database.dart';
import 'sync_outbox_dao.dart';

class GuidedVisitDao {
  GuidedVisitDao(this._db);
  final AppDatabase _db;

  Future<Visit> createDraft({
    required int childId,
    required int ownerUserId,
    required String localUuid,
    required DateTime visitDate,
    required double ageMonths,
    required String deviceMetadataJson,
    required String consentVersion,
    required DateTime consentTimestamp,
    required String consentOperatorIdentifier,
  }) {
    return _db.transaction(() async {
      final child = await (_db.select(_db.children)
            ..where(
              (row) =>
                  row.id.equals(childId) & row.ownerUserId.equals(ownerUserId),
            ))
          .getSingleOrNull();
      if (child == null) {
        throw StateError('Owner-scoped child was not found');
      }
      final existing = await (_db.select(_db.visits)
            ..where(
              (row) =>
                  row.localUuid.equals(localUuid) &
                  row.ownerUserId.equals(ownerUserId),
            ))
          .getSingleOrNull();
      if (existing != null) return existing;

      final visitId = await _db.into(_db.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: localUuid,
              visitDate: Value(visitDate),
              ageMonths: ageMonths,
              ownerUserId: Value(ownerUserId),
              entryMethod: const Value('guided_capture'),
              captureState: const Value('draft_capture'),
              captureStartedAt: Value(DateTime.now()),
              deviceMetadataJson: Value(deviceMetadataJson),
              consentVersion: Value(consentVersion),
              consentTimestamp: Value(consentTimestamp),
              consentOperatorIdentifier: Value(consentOperatorIdentifier),
            ),
          );
      final payloadJson = jsonEncode({
        'local_uuid': localUuid,
        'child_id': childId,
        'visit_date': visitDate.toIso8601String(),
        'age_months': ageMonths,
        'capture_state': 'draft_capture',
        'device_metadata': jsonDecode(deviceMetadataJson),
        'consent_version': consentVersion,
        'consent_timestamp': consentTimestamp.toIso8601String(),
        'consent_operator_identifier': consentOperatorIdentifier,
      });
      await SyncOutboxDao(_db).enqueue(
        ownerUserId: ownerUserId,
        visitUuid: localUuid,
        entityType: SyncOutboxEntityType.visit,
        entityUuid: localUuid,
        payloadJson: payloadJson,
      );
      return (_db.select(_db.visits)..where((row) => row.id.equals(visitId)))
          .getSingle();
    });
  }

  Future<Visit?> getByUuid({
    required int ownerUserId,
    required String visitUuid,
  }) {
    return (_db.select(_db.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingleOrNull();
  }

  Future<Visit> markIncompleteCapture({
    required int ownerUserId,
    required String visitUuid,
  }) {
    return _db.transaction(() async {
      final visit = await getByUuid(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
      );
      if (visit == null) {
        throw StateError('Owner-scoped visit was not found');
      }
      if (visit.captureState == 'incomplete_capture') return visit;
      if (visit.captureState != 'draft_capture') {
        throw StateError(
          'Only a draft capture can be saved as incomplete',
        );
      }

      final completedAt = DateTime.now();
      await (_db.update(_db.visits)..where((row) => row.id.equals(visit.id)))
          .write(
        VisitsCompanion(
          captureState: const Value('incomplete_capture'),
          captureCompletedAt: Value(completedAt),
        ),
      );

      final outbox = await (_db.select(_db.syncOutbox)
            ..where(
              (row) =>
                  row.ownerUserId.equals(ownerUserId) &
                  row.entityType.equals(SyncOutboxEntityType.visit) &
                  row.entityUuid.equals(visitUuid),
            ))
          .getSingleOrNull();
      if (outbox == null) {
        throw StateError('Visit outbox record was not found');
      }
      final payload = jsonDecode(outbox.payloadJson) as Map<String, dynamic>;
      payload['capture_state'] = 'incomplete_capture';
      payload['capture_completed_at'] = completedAt.toIso8601String();
      await SyncOutboxDao(_db).refreshPayload(
        ownerUserId: ownerUserId,
        entityType: SyncOutboxEntityType.visit,
        entityUuid: visitUuid,
        payloadJson: jsonEncode(payload),
      );

      return (_db.select(_db.visits)..where((row) => row.id.equals(visit.id)))
          .getSingle();
    });
  }
}
