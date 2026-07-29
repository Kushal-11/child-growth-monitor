import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:drift/drift.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

import '../../../database/daos/sync_outbox_dao.dart';
import '../../../database/database.dart';
import '../../../services/image_storage_service.dart';

class GuidedMediaStatus {
  const GuidedMediaStatus({
    required this.acknowledged,
    required this.pending,
    required this.failed,
    required this.deletionRequested,
  });

  final int acknowledged;
  final int pending;
  final int failed;
  final int deletionRequested;

  int get total => acknowledged + pending + failed + deletionRequested;
}

abstract interface class GuidedSyncGateway {
  Future<void> runOnce(int ownerUserId);

  Future<GuidedMediaStatus> mediaStatus(int ownerUserId);

  Future<int> cleanupAcknowledgedMedia(int ownerUserId);

  Future<void> requestMediaDeletion({
    required int ownerUserId,
    required String visitUuid,
    required String assetUuid,
  });
}

class GuidedSyncService implements GuidedSyncGateway {
  GuidedSyncService({
    required AppDatabase database,
    required SyncOutboxDao outboxDao,
    required String baseUrl,
    required ImageStorageService imageStorage,
    http.Client? httpClient,
    String? authToken,
    void Function()? onUnauthorized,
    Duration requestTimeout = const Duration(seconds: 60),
  })  : _database = database,
        _outboxDao = outboxDao,
        _baseUrl = baseUrl.replaceFirst(RegExp(r'/+$'), ''),
        _imageStorage = imageStorage,
        _client = httpClient ?? http.Client(),
        _authToken = authToken,
        _onUnauthorized = onUnauthorized,
        _requestTimeout = requestTimeout;

  final AppDatabase _database;
  final SyncOutboxDao _outboxDao;
  final String _baseUrl;
  final ImageStorageService _imageStorage;
  final http.Client _client;
  final String? _authToken;
  final void Function()? _onUnauthorized;
  final Duration _requestTimeout;

  @override
  Future<void> runOnce(int ownerUserId) async {
    await _outboxDao.resetSyncing(ownerUserId);
    final attempted = <int>{};
    while (true) {
      final ready = (await _outboxDao.readyForSync(ownerUserId))
          .where((entry) => !attempted.contains(entry.id))
          .toList();
      if (ready.isEmpty) return;
      for (final entry in ready) {
        attempted.add(entry.id);
        await _syncOne(ownerUserId, entry);
      }
    }
  }

  Future<void> _syncOne(int ownerUserId, SyncOutboxData entry) async {
    await _outboxDao.markSyncing(ownerUserId, entry.id);
    try {
      final request = await _requestFor(ownerUserId, entry);
      final streamed = await _client.send(request).timeout(_requestTimeout);
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode == 401) {
        _onUnauthorized?.call();
        throw const _GuidedSyncFailure(
          'Unauthorized (401) — re-login required',
        );
      }
      if (response.statusCode == 409) {
        throw _GuidedSyncFailure(
          'Immutable checksum conflict (409): ${_responseDetail(response)}',
        );
      }
      if (response.statusCode != 200) {
        throw _GuidedSyncFailure(
          'HTTP ${response.statusCode}: ${_responseDetail(response)}',
        );
      }
      final acknowledgement = _parseAcknowledgement(response.body);
      final expectedType = _wireEntityType(entry.entityType);
      if (acknowledgement.entityUuid != entry.entityUuid ||
          acknowledgement.entityType != expectedType) {
        throw _GuidedSyncFailure(
          'Server did not acknowledge ${entry.entityType} '
          '${entry.entityUuid}',
        );
      }
      await _applyAcknowledgement(
        ownerUserId,
        entry,
        acknowledgement,
        response.body,
      );
    } on Object catch (error) {
      await _outboxDao.markFailed(ownerUserId, entry.id, error.toString());
    }
  }

  Future<http.Request> _requestFor(
    int ownerUserId,
    SyncOutboxData entry,
  ) async {
    final payload = _decodePayload(entry.payloadJson);
    late final String path;
    late final String method;
    switch (entry.entityType) {
      case SyncOutboxEntityType.visit:
        method = 'PUT';
        path = '/api/v1/sync/guided/visits/${entry.visitUuid}';
        break;
      case SyncOutboxEntityType.captureAsset:
        method = 'PUT';
        path = '/api/v1/sync/guided/visits/${entry.visitUuid}'
            '/assets/${entry.entityUuid}';
        final asset = await _ownerScopedAsset(ownerUserId, entry);
        final localPath = asset.localPath;
        if (localPath == null || !await File(localPath).exists()) {
          throw const _GuidedSyncFailure(
            'Retained asset file was not found',
          );
        }
        final bytes = await File(localPath).readAsBytes();
        payload['content_base64'] = base64Encode(bytes);
        payload['content_checksum'] = sha256.convert(bytes).toString();
        payload['content_type'] = _contentType(localPath);
        break;
      case SyncOutboxEntityType.cameraResult:
        method = 'PUT';
        path = '/api/v1/sync/guided/visits/${entry.visitUuid}'
            '/camera-results/${entry.entityUuid}';
        payload['visit_uuid'] = entry.visitUuid;
        break;
      case SyncOutboxEntityType.measuredRevision:
        method = 'PUT';
        path = '/api/v1/sync/guided/visits/${entry.visitUuid}'
            '/measured-revisions/${entry.entityUuid}';
        final revision = await _ownerScopedRevision(ownerUserId, entry);
        payload['visit_uuid'] = entry.visitUuid;
        payload['revision_number'] = revision.revisionNumber;
        break;
      case SyncOutboxEntityType.mediaDeletion:
        method = 'DELETE';
        path = '/api/v1/sync/guided/visits/${entry.visitUuid}'
            '/media/${entry.entityUuid}';
        break;
      default:
        throw _GuidedSyncFailure(
          'Unsupported outbox entity type ${entry.entityType}',
        );
    }

    final request = http.Request(method, Uri.parse('$_baseUrl$path'));
    request.headers['Accept'] = 'application/json';
    if (_authToken != null) {
      request.headers['Authorization'] = 'Bearer $_authToken';
    }
    if (method != 'DELETE') {
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode(payload);
    }
    return request;
  }

  Future<void> _applyAcknowledgement(
    int ownerUserId,
    SyncOutboxData entry,
    _GuidedAcknowledgement acknowledgement,
    String rawAcknowledgement,
  ) async {
    String? mediaPathToDelete;
    await _database.transaction(() async {
      switch (entry.entityType) {
        case SyncOutboxEntityType.visit:
          await (_database.update(_database.visits)
                ..where(
                  (row) =>
                      row.localUuid.equals(entry.entityUuid) &
                      row.ownerUserId.equals(ownerUserId),
                ))
              .write(
            VisitsCompanion(
              serverId: Value(acknowledgement.serverId),
            ),
          );
          break;
        case SyncOutboxEntityType.captureAsset:
          final asset = await _ownerScopedAsset(ownerUserId, entry);
          await (_database.update(_database.captureAssets)
                ..where((row) => row.id.equals(asset.id)))
              .write(
            CaptureAssetsCompanion(
              serverId: Value(acknowledgement.serverId),
              serverObjectId: Value(acknowledgement.serverObjectId),
              syncState: const Value('synced'),
              serverAcknowledgedAt: Value(acknowledgement.acknowledgedAt),
            ),
          );
          break;
        case SyncOutboxEntityType.cameraResult:
          final result = await _ownerScopedCameraResult(ownerUserId, entry);
          await (_database.update(_database.cameraResults)
                ..where((row) => row.id.equals(result.id)))
              .write(
            CameraResultsCompanion(
              serverId: Value(acknowledgement.serverId),
            ),
          );
          break;
        case SyncOutboxEntityType.measuredRevision:
          final revision = await _ownerScopedRevision(ownerUserId, entry);
          await (_database.update(_database.measuredDetailRevisions)
                ..where((row) => row.id.equals(revision.id)))
              .write(
            MeasuredDetailRevisionsCompanion(
              serverId: Value(acknowledgement.serverId),
            ),
          );
          break;
        case SyncOutboxEntityType.mediaDeletion:
          final asset = await _ownerScopedAsset(ownerUserId, entry);
          mediaPathToDelete = asset.localPath;
          await (_database.update(_database.captureAssets)
                ..where((row) => row.id.equals(asset.id)))
              .write(
            const CaptureAssetsCompanion(localPath: Value(null)),
          );
          await _markVisitMediaDeletedWhenEmpty(asset.visitId);
          break;
      }
      await _outboxDao.acknowledge(
        ownerUserId,
        entry.id,
        rawAcknowledgement,
      );
    });
    if (mediaPathToDelete != null) {
      await _imageStorage.deleteAcknowledged([mediaPathToDelete!]);
    }
  }

  @override
  Future<GuidedMediaStatus> mediaStatus(int ownerUserId) async {
    final assets = await _ownerAssets(ownerUserId);
    final outbox = await (_database.select(_database.syncOutbox)
          ..where((row) => row.ownerUserId.equals(ownerUserId)))
        .get();
    var acknowledged = 0;
    var pending = 0;
    var failed = 0;
    var deletionRequested = 0;
    for (final asset in assets.where((asset) => asset.localPath != null)) {
      final deletion = outbox
          .where(
            (entry) =>
                entry.entityType == SyncOutboxEntityType.mediaDeletion &&
                entry.entityUuid == asset.assetUuid &&
                entry.status != 'acknowledged',
          )
          .firstOrNull;
      if (deletion != null) {
        deletionRequested += 1;
        continue;
      }
      final upload = outbox
          .where(
            (entry) =>
                entry.entityType == SyncOutboxEntityType.captureAsset &&
                entry.entityUuid == asset.assetUuid,
          )
          .firstOrNull;
      if (upload?.status == 'failed') {
        failed += 1;
      } else if (asset.serverAcknowledgedAt != null ||
          upload?.status == 'acknowledged') {
        acknowledged += 1;
      } else {
        pending += 1;
      }
    }
    return GuidedMediaStatus(
      acknowledged: acknowledged,
      pending: pending,
      failed: failed,
      deletionRequested: deletionRequested,
    );
  }

  @override
  Future<int> cleanupAcknowledgedMedia(int ownerUserId) async {
    final assets = await _ownerAssets(ownerUserId);
    final deletions = await (_database.select(_database.syncOutbox)
          ..where(
            (row) =>
                row.ownerUserId.equals(ownerUserId) &
                row.entityType.equals(SyncOutboxEntityType.mediaDeletion) &
                row.status.isNotValue('acknowledged'),
          ))
        .get();
    final protectedUuids = deletions.map((entry) => entry.entityUuid).toSet();
    final eligible = assets
        .where(
          (asset) =>
              asset.localPath != null &&
              asset.serverAcknowledgedAt != null &&
              !protectedUuids.contains(asset.assetUuid),
        )
        .toList();
    final deleted = await _imageStorage.deleteAcknowledged(
      eligible.map((asset) => asset.localPath!),
    );
    await _database.transaction(() async {
      for (final asset in eligible) {
        await (_database.update(_database.captureAssets)
              ..where((row) => row.id.equals(asset.id)))
            .write(
          const CaptureAssetsCompanion(localPath: Value(null)),
        );
        await _markVisitMediaDeletedWhenEmpty(asset.visitId);
      }
    });
    return deleted;
  }

  @override
  Future<void> requestMediaDeletion({
    required int ownerUserId,
    required String visitUuid,
    required String assetUuid,
  }) async {
    final visit = await (_database.select(_database.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingleOrNull();
    if (visit == null) {
      throw StateError('Owner-scoped visit was not found');
    }
    final asset = await (_database.select(_database.captureAssets)
          ..where(
            (row) =>
                row.visitId.equals(visit.id) & row.assetUuid.equals(assetUuid),
          ))
        .getSingleOrNull();
    if (asset == null) {
      throw StateError('Owner-scoped asset was not found');
    }
    final existing = await (_database.select(_database.syncOutbox)
          ..where(
            (row) =>
                row.ownerUserId.equals(ownerUserId) &
                row.entityType.equals(SyncOutboxEntityType.mediaDeletion) &
                row.entityUuid.equals(assetUuid),
          ))
        .getSingleOrNull();
    if (existing != null) return;
    await _outboxDao.enqueue(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      entityType: SyncOutboxEntityType.mediaDeletion,
      entityUuid: assetUuid,
      operation: 'delete',
      dependencyEntityUuid: assetUuid,
      payloadJson: jsonEncode({
        'visit_uuid': visitUuid,
        'asset_uuid': assetUuid,
      }),
    );
  }

  Future<CaptureAsset> _ownerScopedAsset(
    int ownerUserId,
    SyncOutboxData entry,
  ) async {
    final visit = await (_database.select(_database.visits)
          ..where(
            (row) =>
                row.localUuid.equals(entry.visitUuid) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingleOrNull();
    if (visit == null) throw const _GuidedSyncFailure('Visit not found');
    final asset = await (_database.select(_database.captureAssets)
          ..where(
            (row) =>
                row.visitId.equals(visit.id) &
                row.assetUuid.equals(entry.entityUuid),
          ))
        .getSingleOrNull();
    if (asset == null) throw const _GuidedSyncFailure('Asset not found');
    return asset;
  }

  Future<CameraResult> _ownerScopedCameraResult(
    int ownerUserId,
    SyncOutboxData entry,
  ) async {
    final visit = await _ownerVisit(ownerUserId, entry.visitUuid);
    final result = await (_database.select(_database.cameraResults)
          ..where(
            (row) =>
                row.visitId.equals(visit.id) &
                row.resultUuid.equals(entry.entityUuid),
          ))
        .getSingleOrNull();
    if (result == null) {
      throw const _GuidedSyncFailure('Camera result not found');
    }
    return result;
  }

  Future<MeasuredDetailRevision> _ownerScopedRevision(
    int ownerUserId,
    SyncOutboxData entry,
  ) async {
    final visit = await _ownerVisit(ownerUserId, entry.visitUuid);
    final revision = await (_database.select(_database.measuredDetailRevisions)
          ..where(
            (row) =>
                row.visitId.equals(visit.id) &
                row.revisionUuid.equals(entry.entityUuid),
          ))
        .getSingleOrNull();
    if (revision == null) {
      throw const _GuidedSyncFailure('Measured revision not found');
    }
    return revision;
  }

  Future<Visit> _ownerVisit(int ownerUserId, String visitUuid) async {
    final visit = await (_database.select(_database.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingleOrNull();
    if (visit == null) throw const _GuidedSyncFailure('Visit not found');
    return visit;
  }

  Future<List<CaptureAsset>> _ownerAssets(int ownerUserId) async {
    final visits = await (_database.select(_database.visits)
          ..where((row) => row.ownerUserId.equals(ownerUserId)))
        .get();
    if (visits.isEmpty) return const [];
    return (_database.select(_database.captureAssets)
          ..where(
            (row) => row.visitId.isIn(visits.map((visit) => visit.id)),
          ))
        .get();
  }

  Future<void> _markVisitMediaDeletedWhenEmpty(int visitId) async {
    final remaining = await (_database.select(_database.captureAssets)
          ..where(
            (row) => row.visitId.equals(visitId) & row.localPath.isNotNull(),
          ))
        .get();
    if (remaining.isEmpty) {
      await (_database.update(_database.visits)
            ..where((row) => row.id.equals(visitId)))
          .write(
        VisitsCompanion(mediaDeletedAt: Value(DateTime.now())),
      );
    }
  }

  static Map<String, dynamic> _decodePayload(String raw) {
    final decoded = jsonDecode(raw);
    if (decoded is! Map<String, dynamic>) {
      throw const _GuidedSyncFailure('Outbox payload must be a JSON object');
    }
    return Map<String, dynamic>.from(decoded);
  }

  static String _contentType(String filePath) {
    return switch (p.extension(filePath).toLowerCase()) {
      '.png' => 'image/png',
      '.webp' => 'image/webp',
      _ => 'image/jpeg',
    };
  }

  static String _wireEntityType(String entityType) => switch (entityType) {
        SyncOutboxEntityType.visit => 'visit',
        SyncOutboxEntityType.captureAsset => 'capture_asset',
        SyncOutboxEntityType.cameraResult => 'camera_result',
        SyncOutboxEntityType.measuredRevision => 'measured_revision',
        SyncOutboxEntityType.mediaDeletion => 'media_deletion',
        _ => entityType,
      };

  static _GuidedAcknowledgement _parseAcknowledgement(String raw) {
    final body = _decodePayload(raw);
    final status = body['status'];
    final entityType = body['entity_type'];
    final entityUuid = body['entity_uuid'];
    final acknowledgedAt = body['acknowledged_at'];
    if ((status != 'accepted' && status != 'already_accepted') ||
        entityType is! String ||
        entityUuid is! String ||
        acknowledgedAt is! String) {
      throw const _GuidedSyncFailure(
        'Server acknowledgement is incomplete',
      );
    }
    return _GuidedAcknowledgement(
      entityType: entityType,
      entityUuid: entityUuid,
      status: status as String,
      serverId: body['server_id'] as int?,
      serverObjectId: body['server_object_id'] as String?,
      acknowledgedAt: DateTime.parse(acknowledgedAt),
    );
  }

  static String _responseDetail(http.Response response) {
    try {
      final body = jsonDecode(response.body);
      if (body is Map && body['detail'] is String) {
        return body['detail'] as String;
      }
    } on FormatException {
      // Fall through to the raw response.
    }
    return response.body;
  }
}

class _GuidedAcknowledgement {
  const _GuidedAcknowledgement({
    required this.entityType,
    required this.entityUuid,
    required this.status,
    required this.serverId,
    required this.serverObjectId,
    required this.acknowledgedAt,
  });

  final String entityType;
  final String entityUuid;
  final String status;
  final int? serverId;
  final String? serverObjectId;
  final DateTime acknowledgedAt;
}

class _GuidedSyncFailure implements Exception {
  const _GuidedSyncFailure(this.message);

  final String message;

  @override
  String toString() => message;
}
