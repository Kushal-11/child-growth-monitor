import 'dart:convert';

import 'package:drift/drift.dart';

import '../../../database/daos/sync_outbox_dao.dart';
import '../../../database/database.dart';
import '../domain/ar_scan_models.dart';

abstract interface class ArScanRepository {
  Future<ArScanVisitContext> getVisitContext({
    required int ownerUserId,
    required String visitUuid,
  });

  Future<void> saveExperimentalResult({
    required int ownerUserId,
    required String visitUuid,
    required FullArScanResult result,
  });
}

class ArScanVisitContext {
  const ArScanVisitContext({
    required this.ageMonths,
    required this.sex,
    required this.entryMethod,
  });

  final double ageMonths;
  final String sex;
  final String entryMethod;
}

class DriftArScanRepository implements ArScanRepository {
  DriftArScanRepository(this._database);
  final AppDatabase _database;

  @override
  Future<ArScanVisitContext> getVisitContext({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    final visit =
        await (_database.select(_database.visits)..where(
              (row) =>
                  row.localUuid.equals(visitUuid) &
                  row.ownerUserId.equals(ownerUserId),
            ))
            .getSingleOrNull();
    if (visit == null) throw StateError('Owner-scoped visit was not found');
    final child =
        await (_database.select(_database.children)..where(
              (row) =>
                  row.id.equals(visit.childId) &
                  row.ownerUserId.equals(ownerUserId),
            ))
            .getSingleOrNull();
    if (child == null) throw StateError('Owner-scoped child was not found');
    return ArScanVisitContext(
      ageMonths: visit.ageMonths,
      sex: child.sex,
      entryMethod: visit.entryMethod,
    );
  }

  @override
  Future<void> saveExperimentalResult({
    required int ownerUserId,
    required String visitUuid,
    required FullArScanResult result,
  }) => _database.transaction(() async {
    final visit =
        await (_database.select(_database.visits)..where(
              (row) =>
                  row.localUuid.equals(visitUuid) &
                  row.ownerUserId.equals(ownerUserId),
            ))
            .getSingleOrNull();
    if (visit == null) throw StateError('Owner-scoped visit was not found');
    final metadata = _decodeObject(visit.deviceMetadataJson);
    metadata['arcore_depth_scan'] = result.toJson();
    final encoded = jsonEncode(metadata);
    await (_database.update(_database.visits)
          ..where((row) => row.id.equals(visit.id)))
        .write(VisitsCompanion(deviceMetadataJson: Value(encoded)));

    final outbox =
        await (_database.select(_database.syncOutbox)..where(
              (row) =>
                  row.ownerUserId.equals(ownerUserId) &
                  row.entityType.equals(SyncOutboxEntityType.visit) &
                  row.entityUuid.equals(visitUuid),
            ))
            .getSingleOrNull();
    // Guided-capture visits have a typed outbox entry. Standard assessment
    // visits use the legacy sync queue, which reads the visit row later.
    // Persist the depth result in both paths and refresh an outbox payload
    // only when that payload exists.
    if (outbox != null) {
      final payload = _decodeObject(outbox.payloadJson);
      payload['device_metadata'] = metadata;
      await SyncOutboxDao(_database).refreshPayload(
        ownerUserId: ownerUserId,
        entityType: SyncOutboxEntityType.visit,
        entityUuid: visitUuid,
        payloadJson: jsonEncode(payload),
      );
    }
  });

  Map<String, dynamic> _decodeObject(String? value) {
    if (value == null || value.isEmpty) return <String, dynamic>{};
    final decoded = jsonDecode(value);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Expected JSON object');
    }
    return Map<String, dynamic>.from(decoded);
  }
}
