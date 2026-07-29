import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:drift/drift.dart';

import '../database.dart';

abstract final class SyncOutboxEntityType {
  static const visit = 'visit';
  static const captureAsset = 'capture_asset';
  static const cameraResult = 'camera_result';
  static const measuredRevision = 'measured_revision';
  static const mediaDeletion = 'media_deletion';
}

class SyncOutboxDao {
  SyncOutboxDao(this._db);
  final AppDatabase _db;

  static const maxRetryCount = 5;

  static String checksumForPayload(String payloadJson) =>
      sha256.convert(utf8.encode(payloadJson)).toString();

  Future<SyncOutboxData> enqueue({
    required int ownerUserId,
    required String visitUuid,
    required String entityType,
    required String entityUuid,
    required String payloadJson,
    String operation = 'upsert',
    String? dependencyEntityUuid,
  }) async {
    final id = await _db.into(_db.syncOutbox).insert(
          SyncOutboxCompanion.insert(
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
            entityType: entityType,
            entityUuid: entityUuid,
            operation: Value(operation),
            dependencyEntityUuid: Value(dependencyEntityUuid),
            payloadJson: payloadJson,
            payloadChecksum: checksumForPayload(payloadJson),
          ),
        );
    return (_db.select(_db.syncOutbox)..where((row) => row.id.equals(id)))
        .getSingle();
  }

  Future<List<SyncOutboxData>> readyForSync(int ownerUserId) async {
    final allOwnerRows = await (_db.select(_db.syncOutbox)
          ..where((row) => row.ownerUserId.equals(ownerUserId))
          ..orderBy([
            (row) => OrderingTerm.asc(row.createdAt),
            (row) => OrderingTerm.asc(row.id),
          ]))
        .get();
    final acknowledgedUuids = allOwnerRows
        .where((row) => row.status == 'acknowledged')
        .map((row) => row.entityUuid)
        .toSet();

    bool visitAcknowledged(SyncOutboxData row) => allOwnerRows.any(
          (candidate) =>
              candidate.entityType == SyncOutboxEntityType.visit &&
              candidate.entityUuid == row.visitUuid &&
              candidate.status == 'acknowledged',
        );

    bool allAssetsAcknowledged(SyncOutboxData row) {
      final assets = allOwnerRows
          .where(
            (candidate) =>
                candidate.visitUuid == row.visitUuid &&
                candidate.entityType == SyncOutboxEntityType.captureAsset,
          )
          .toList();
      return assets.isNotEmpty &&
          assets.every((candidate) => candidate.status == 'acknowledged');
    }

    return allOwnerRows.where((row) {
      final retryable = (row.status == 'pending' || row.status == 'failed') &&
          row.retryCount < maxRetryCount;
      if (!retryable) return false;
      if (row.dependencyEntityUuid != null &&
          !acknowledgedUuids.contains(row.dependencyEntityUuid)) {
        return false;
      }
      return switch (row.entityType) {
        SyncOutboxEntityType.visit => true,
        SyncOutboxEntityType.captureAsset => visitAcknowledged(row),
        SyncOutboxEntityType.measuredRevision => visitAcknowledged(row),
        SyncOutboxEntityType.cameraResult =>
          visitAcknowledged(row) && allAssetsAcknowledged(row),
        SyncOutboxEntityType.mediaDeletion => allAssetsAcknowledged(row),
        _ => false,
      };
    }).toList();
  }

  Future<void> refreshPayload({
    required int ownerUserId,
    required String entityType,
    required String entityUuid,
    required String payloadJson,
  }) async {
    final updated = await (_db.update(_db.syncOutbox)
          ..where(
            (row) =>
                row.ownerUserId.equals(ownerUserId) &
                row.entityType.equals(entityType) &
                row.entityUuid.equals(entityUuid),
          ))
        .write(
      SyncOutboxCompanion(
        payloadJson: Value(payloadJson),
        payloadChecksum: Value(checksumForPayload(payloadJson)),
        status: const Value('pending'),
        retryCount: const Value(0),
        lastAttemptAt: const Value(null),
        acknowledgedAt: const Value(null),
        acknowledgementPayloadJson: const Value(null),
        errorMessage: const Value(null),
      ),
    );
    if (updated != 1) {
      throw StateError('Owner-scoped outbox entity was not found');
    }
  }

  Future<void> markSyncing(int ownerUserId, int id) {
    return (_db.update(_db.syncOutbox)
          ..where(
            (row) => row.id.equals(id) & row.ownerUserId.equals(ownerUserId),
          ))
        .write(
      SyncOutboxCompanion(
        status: const Value('syncing'),
        lastAttemptAt: Value(DateTime.now()),
        errorMessage: const Value(null),
      ),
    );
  }

  Future<void> resetSyncing(int ownerUserId) {
    return (_db.update(_db.syncOutbox)
          ..where(
            (row) =>
                row.ownerUserId.equals(ownerUserId) &
                row.status.equals('syncing'),
          ))
        .write(
      const SyncOutboxCompanion(
        status: Value('pending'),
        lastAttemptAt: Value(null),
        errorMessage: Value(null),
      ),
    );
  }

  Future<void> acknowledge(
    int ownerUserId,
    int id,
    String acknowledgementPayloadJson,
  ) {
    return (_db.update(_db.syncOutbox)
          ..where(
            (row) => row.id.equals(id) & row.ownerUserId.equals(ownerUserId),
          ))
        .write(
      SyncOutboxCompanion(
        status: const Value('acknowledged'),
        acknowledgementPayloadJson: Value(acknowledgementPayloadJson),
        acknowledgedAt: Value(DateTime.now()),
        lastAttemptAt: Value(DateTime.now()),
        errorMessage: const Value(null),
      ),
    );
  }

  Future<void> markFailed(
    int ownerUserId,
    int id,
    String error,
  ) async {
    final entry = await (_db.select(_db.syncOutbox)
          ..where(
            (row) => row.id.equals(id) & row.ownerUserId.equals(ownerUserId),
          ))
        .getSingle();
    await (_db.update(_db.syncOutbox)
          ..where(
            (row) => row.id.equals(id) & row.ownerUserId.equals(ownerUserId),
          ))
        .write(
      SyncOutboxCompanion(
        status: const Value('failed'),
        retryCount: Value(entry.retryCount + 1),
        errorMessage: Value(error),
        lastAttemptAt: Value(DateTime.now()),
      ),
    );
  }
}
