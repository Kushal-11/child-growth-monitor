import 'dart:convert';

import 'package:drift/drift.dart';

import '../../../database/daos/sync_outbox_dao.dart';
import '../../../database/database.dart';
import '../domain/ar_scan_models.dart';

abstract interface class ArScanRepository {
  Future<void> saveExperimentalResult({
    required int ownerUserId,
    required String visitUuid,
    required SparseArScanResult result,
  });
}

class DriftArScanRepository implements ArScanRepository {
  DriftArScanRepository(this._database);
  final AppDatabase _database;

  @override
  Future<void> saveExperimentalResult({
    required int ownerUserId,
    required String visitUuid,
    required SparseArScanResult result,
  }) => _database.transaction(() async {
        final visit = await (_database.select(_database.visits)
              ..where((row) =>
                  row.localUuid.equals(visitUuid) &
                  row.ownerUserId.equals(ownerUserId)))
            .getSingleOrNull();
        if (visit == null) throw StateError('Owner-scoped visit was not found');
        final metadata = _decodeObject(visit.deviceMetadataJson);
        metadata['sparse_ar_scan'] = result.toJson();
        final encoded = jsonEncode(metadata);
        await (_database.update(_database.visits)
              ..where((row) => row.id.equals(visit.id)))
            .write(VisitsCompanion(deviceMetadataJson: Value(encoded)));

        final outbox = await (_database.select(_database.syncOutbox)
              ..where((row) =>
                  row.ownerUserId.equals(ownerUserId) &
                  row.entityType.equals(SyncOutboxEntityType.visit) &
                  row.entityUuid.equals(visitUuid)))
            .getSingleOrNull();
        if (outbox == null) throw StateError('Visit outbox record was not found');
        final payload = _decodeObject(outbox.payloadJson);
        payload['device_metadata'] = metadata;
        await SyncOutboxDao(_database).refreshPayload(
          ownerUserId: ownerUserId,
          entityType: SyncOutboxEntityType.visit,
          entityUuid: visitUuid,
          payloadJson: jsonEncode(payload),
        );
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
