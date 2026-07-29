import 'dart:convert';
import 'dart:io';

import 'package:child_growth_monitor_app/database/daos/sync_outbox_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_sync_service.dart';
import 'package:child_growth_monitor_app/services/image_storage_service.dart';
import 'package:crypto/crypto.dart';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  late AppDatabase db;
  late SyncOutboxDao outboxDao;
  late Directory root;
  const ownerUserId = 7;
  const visitUuid = '10000000-0000-0000-0000-000000000001';

  setUp(() async {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    outboxDao = SyncOutboxDao(db);
    root = await Directory.systemTemp.createTemp('guided-sync-flutter-');
  });

  tearDown(() async {
    await db.close();
    if (await root.exists()) await root.delete(recursive: true);
  });

  test('drains typed entities in dependency order with exact request shapes',
      () async {
    final fixture = await seedFullOutbox(
      db,
      outboxDao,
      root,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final requests = <http.Request>[];
    final client = MockClient((request) async {
      requests.add(request);
      final identity = identityFor(request);
      return http.Response(
        jsonEncode({
          'entity_type': identity.type,
          'entity_uuid': identity.uuid,
          'status': 'accepted',
          'server_id': requests.length + 40,
          if (identity.type == 'capture_asset')
            'server_object_id': '7/$visitUuid/${identity.uuid}.jpg',
          'checksum': 'a' * 64,
          'acknowledged_at': '2026-07-29T10:00:00Z',
        }),
        200,
      );
    });
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      authToken: 'token',
      httpClient: client,
      imageStorage: ImageStorageService(rootOverride: root),
    );

    await service.runOnce(ownerUserId);

    expect(
      requests.map((request) => identityFor(request).type),
      [
        'visit',
        'capture_asset',
        'capture_asset',
        'measured_revision',
        'camera_result',
      ],
    );
    expect(requests.every((request) => request.method == 'PUT'), isTrue);
    expect(
      requests.every(
        (request) => request.headers['Authorization'] == 'Bearer token',
      ),
      isTrue,
    );
    final frontBody = jsonDecode(requests[1].body) as Map<String, dynamic>;
    expect(frontBody['content_base64'], base64Encode(fixture.frontBytes));
    expect(
      frontBody['content_checksum'],
      sha256.convert(fixture.frontBytes).toString(),
    );
    final revisionBody = jsonDecode(requests[3].body) as Map<String, dynamic>;
    expect(revisionBody['revision_number'], 1);
    final resultBody = jsonDecode(requests[4].body) as Map<String, dynamic>;
    expect(resultBody['visit_uuid'], visitUuid);

    final entries = await db.select(db.syncOutbox).get();
    expect(entries.every((entry) => entry.status == 'acknowledged'), isTrue);
    final assets = await db.select(db.captureAssets).get();
    expect(assets.every((asset) => asset.serverId != null), isTrue);
    expect(assets.every((asset) => asset.serverObjectId != null), isTrue);
    expect(assets.every((asset) => asset.serverAcknowledgedAt != null), isTrue);
    expect(File(fixture.frontPath).existsSync(), isTrue);
    expect(File(fixture.sidePath).existsSync(), isTrue);
  });

  test('offline failure increments retry and retains local media', () async {
    final fixture = await seedAssetOutbox(
      db,
      outboxDao,
      root,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      acknowledgeVisit: true,
    );
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      authToken: 'token',
      httpClient: MockClient((_) => throw const SocketException('offline')),
      imageStorage: ImageStorageService(rootOverride: root),
    );

    await service.runOnce(ownerUserId);

    final entries = await (db.select(db.syncOutbox)
          ..where((row) => row.entityType.equals('capture_asset')))
        .get();
    expect(entries, hasLength(2));
    expect(entries.every((entry) => entry.status == 'failed'), isTrue);
    expect(entries.every((entry) => entry.retryCount == 1), isTrue);
    expect(
      entries
          .every((entry) => entry.errorMessage?.contains('offline') ?? false),
      isTrue,
    );
    expect(File(fixture.frontPath).existsSync(), isTrue);
  });

  test('timeout fails the entity without losing its retryable payload',
      () async {
    final original = await seedVisitOutbox(
      db,
      outboxDao,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      httpClient: MockClient((_) async {
        await Future<void>.delayed(const Duration(milliseconds: 50));
        return http.Response('{}', 200);
      }),
      requestTimeout: const Duration(milliseconds: 5),
      imageStorage: ImageStorageService(rootOverride: root),
    );

    await service.runOnce(ownerUserId);

    final entry = await db.select(db.syncOutbox).getSingle();
    expect(entry.status, 'failed');
    expect(entry.retryCount, 1);
    expect(entry.payloadJson, original.payloadJson);
    expect(entry.payloadChecksum, original.payloadChecksum);
    expect(entry.errorMessage, contains('TimeoutException'));
  });

  test('401 triggers logout callback and 409 remains a failed conflict',
      () async {
    await seedVisitOutbox(
      db,
      outboxDao,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    var unauthorizedCalls = 0;
    var status = 401;
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      authToken: 'token',
      httpClient: MockClient((_) async => http.Response('conflict', status)),
      imageStorage: ImageStorageService(rootOverride: root),
      onUnauthorized: () => unauthorizedCalls += 1,
    );

    await service.runOnce(ownerUserId);
    expect(unauthorizedCalls, 1);
    var entry = await db.select(db.syncOutbox).getSingle();
    expect(entry.status, 'failed');
    expect(entry.errorMessage, contains('401'));

    await outboxDao.refreshPayload(
      ownerUserId: ownerUserId,
      entityType: 'visit',
      entityUuid: visitUuid,
      payloadJson: entry.payloadJson,
    );
    status = 409;
    await service.runOnce(ownerUserId);
    entry = await db.select(db.syncOutbox).getSingle();
    expect(entry.status, 'failed');
    expect(entry.errorMessage, contains('checksum conflict'));
  });

  test('wrong acknowledgement UUID cannot mark an entity synced', () async {
    await seedVisitOutbox(
      db,
      outboxDao,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      httpClient: MockClient(
        (_) async => http.Response(
          jsonEncode({
            'entity_type': 'visit',
            'entity_uuid': '10000000-0000-0000-0000-000000000009',
            'status': 'accepted',
            'server_id': 42,
            'acknowledged_at': '2026-07-29T10:00:00Z',
          }),
          200,
        ),
      ),
      imageStorage: ImageStorageService(rootOverride: root),
    );

    await service.runOnce(ownerUserId);

    final entry = await db.select(db.syncOutbox).getSingle();
    expect(entry.status, 'failed');
    expect(entry.errorMessage, contains('did not acknowledge'));
  });

  test('partial acknowledgement retains the failed asset and blocks camera',
      () async {
    final fixture = await seedFullOutbox(
      db,
      outboxDao,
      root,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final requests = <EntityIdentity>[];
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      httpClient: MockClient((request) async {
        final identity = identityFor(request);
        requests.add(identity);
        if (identity.uuid == '20000000-0000-0000-0000-000000000002') {
          return http.Response('temporarily unavailable', 503);
        }
        return http.Response(
          jsonEncode({
            'entity_type': identity.type,
            'entity_uuid': identity.uuid,
            'status': 'accepted',
            'server_id': 42,
            if (identity.type == 'capture_asset')
              'server_object_id': '7/$visitUuid/${identity.uuid}.jpg',
            'acknowledged_at': '2026-07-29T10:00:00Z',
          }),
          200,
        );
      }),
      imageStorage: ImageStorageService(rootOverride: root),
    );

    await service.runOnce(ownerUserId);

    final rows = await db.select(db.syncOutbox).get();
    String statusOf(String entityType, [String? entityUuid]) => rows
        .singleWhere(
          (row) =>
              row.entityType == entityType &&
              (entityUuid == null || row.entityUuid == entityUuid),
        )
        .status;
    expect(
      statusOf(
        'capture_asset',
        '20000000-0000-0000-0000-000000000001',
      ),
      'acknowledged',
    );
    expect(
      statusOf(
        'capture_asset',
        '20000000-0000-0000-0000-000000000002',
      ),
      'failed',
    );
    expect(statusOf('camera_result'), 'pending');
    expect(statusOf('measured_revision'), 'acknowledged');
    expect(
        requests.where((request) => request.type == 'camera_result'), isEmpty);
    expect(File(fixture.frontPath).existsSync(), isTrue);
    expect(File(fixture.sidePath).existsSync(), isTrue);
  });

  test('normal cleanup deletes only individually acknowledged media', () async {
    final fixture = await seedAssetOutbox(
      db,
      outboxDao,
      root,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      acknowledgeVisit: true,
    );
    final frontOutbox = await (db.select(db.syncOutbox)
          ..where(
            (row) =>
                row.entityType.equals('capture_asset') &
                row.entityUuid.equals(
                  '20000000-0000-0000-0000-000000000001',
                ),
          ))
        .getSingle();
    await outboxDao.acknowledge(
      ownerUserId,
      frontOutbox.id,
      '{"status":"accepted"}',
    );
    await (db.update(db.captureAssets)
          ..where(
            (row) => row.assetUuid.equals(
              '20000000-0000-0000-0000-000000000001',
            ),
          ))
        .write(
      CaptureAssetsCompanion(
        serverId: const Value(51),
        serverAcknowledgedAt: Value(DateTime.utc(2026, 7, 29, 10)),
      ),
    );
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      httpClient: MockClient((_) async => http.Response('{}', 500)),
      imageStorage: ImageStorageService(rootOverride: root),
    );

    final before = await service.mediaStatus(ownerUserId);
    expect(before.acknowledged, 1);
    expect(before.pending, 1);
    expect(await service.cleanupAcknowledgedMedia(ownerUserId), 1);

    expect(File(fixture.frontPath).existsSync(), isFalse);
    expect(File(fixture.sidePath).existsSync(), isTrue);
    final assets = await db.select(db.captureAssets).get();
    expect(
      assets.singleWhere((asset) => asset.role == 'front').localPath,
      equals(null),
    );
    expect(
      assets.singleWhere((asset) => asset.role == 'side').localPath,
      fixture.sidePath,
    );
  });

  test('explicit deletion waits for its acknowledgement and preserves history',
      () async {
    final fixture = await seedFullOutbox(
      db,
      outboxDao,
      root,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final methods = <String>[];
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      httpClient: MockClient((request) async {
        methods.add(request.method);
        final identity = identityFor(request);
        return http.Response(
          jsonEncode({
            'entity_type': identity.type,
            'entity_uuid': identity.uuid,
            'status': 'accepted',
            'server_id': 42,
            if (identity.type == 'capture_asset')
              'server_object_id': '7/$visitUuid/${identity.uuid}.jpg',
            'acknowledged_at': '2026-07-29T10:00:00Z',
          }),
          200,
        );
      }),
      imageStorage: ImageStorageService(rootOverride: root),
    );
    await service.runOnce(ownerUserId);

    await service.requestMediaDeletion(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      assetUuid: '20000000-0000-0000-0000-000000000001',
    );
    expect(File(fixture.frontPath).existsSync(), isTrue);
    expect((await service.mediaStatus(ownerUserId)).deletionRequested, 1);

    await service.runOnce(ownerUserId);

    expect(methods.last, 'DELETE');
    expect(File(fixture.frontPath).existsSync(), isFalse);
    expect(File(fixture.sidePath).existsSync(), isTrue);
    expect(await db.select(db.captureAssets).get(), hasLength(2));
    expect(await db.select(db.cameraResults).get(), hasLength(1));
    expect(await db.select(db.measuredDetailRevisions).get(), hasLength(1));
    final deletion = await (db.select(db.syncOutbox)
          ..where((row) => row.entityType.equals('media_deletion')))
        .getSingle();
    expect(deletion.status, 'acknowledged');
  });

  test('process-death recovery resets syncing and retry exhaustion is bounded',
      () async {
    final entry = await seedVisitOutbox(
      db,
      outboxDao,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    await outboxDao.markSyncing(ownerUserId, entry.id);
    var requests = 0;
    final service = GuidedSyncService(
      database: db,
      outboxDao: outboxDao,
      baseUrl: 'http://server.test',
      httpClient: MockClient((request) async {
        requests += 1;
        final identity = identityFor(request);
        return http.Response(
          jsonEncode({
            'entity_type': identity.type,
            'entity_uuid': identity.uuid,
            'status': 'already_accepted',
            'server_id': 42,
            'acknowledged_at': '2026-07-29T10:00:00Z',
          }),
          200,
        );
      }),
      imageStorage: ImageStorageService(rootOverride: root),
    );

    await service.runOnce(ownerUserId);
    expect(requests, 1);
    expect((await db.select(db.syncOutbox).getSingle()).status, 'acknowledged');

    final exhausted = await outboxDao.enqueue(
      ownerUserId: ownerUserId,
      visitUuid: '10000000-0000-0000-0000-000000000002',
      entityType: 'visit',
      entityUuid: '10000000-0000-0000-0000-000000000002',
      payloadJson: '{}',
    );
    for (var count = 0; count < SyncOutboxDao.maxRetryCount; count++) {
      await outboxDao.markFailed(ownerUserId, exhausted.id, 'offline');
    }
    await service.runOnce(ownerUserId);
    expect(requests, 1);
  });
}

typedef EntityIdentity = ({String type, String uuid});

EntityIdentity identityFor(http.Request request) {
  final segments = request.url.pathSegments;
  if (segments.contains('assets')) {
    return (type: 'capture_asset', uuid: segments.last);
  }
  if (segments.contains('camera-results')) {
    return (type: 'camera_result', uuid: segments.last);
  }
  if (segments.contains('measured-revisions')) {
    return (type: 'measured_revision', uuid: segments.last);
  }
  if (segments.contains('media')) {
    return (type: 'media_deletion', uuid: segments.last);
  }
  return (type: 'visit', uuid: segments.last);
}

class SeededAsset {
  const SeededAsset({
    required this.frontPath,
    required this.sidePath,
    required this.frontBytes,
  });

  final String frontPath;
  final String sidePath;
  final List<int> frontBytes;
}

Future<SeededAsset> seedFullOutbox(
  AppDatabase db,
  SyncOutboxDao dao,
  Directory root, {
  required int ownerUserId,
  required String visitUuid,
}) async {
  final visitEntry = await seedVisitOutbox(
    db,
    dao,
    ownerUserId: ownerUserId,
    visitUuid: visitUuid,
  );
  final asset = await seedAssetOutbox(
    db,
    dao,
    root,
    ownerUserId: ownerUserId,
    visitUuid: visitUuid,
    acknowledgeVisit: false,
  );
  await db.into(db.cameraResults).insert(
        CameraResultsCompanion.insert(
          resultUuid: '30000000-0000-0000-0000-000000000001',
          visitId: 1,
          version: 1,
          method: 'camera_screening_v1',
          modelVersion: 'camera-v1',
          manifestChecksum: 'a' * 64,
          trainingDataLabel: 'research_only',
        ),
      );
  await dao.enqueue(
    ownerUserId: ownerUserId,
    visitUuid: visitUuid,
    entityType: 'camera_result',
    entityUuid: '30000000-0000-0000-0000-000000000001',
    dependencyEntityUuid: visitUuid,
    payloadJson: jsonEncode({
      'result_uuid': '30000000-0000-0000-0000-000000000001',
      'version': 1,
      'estimated_height_cm': 88,
      'estimated_weight_kg': 11,
      'component_probabilities': <String, double>{},
      'body_proportion_features': <String, dynamic>{},
      'capture_quality_summary': {
        'used_views': ['front', 'side'],
      },
      'method': 'camera_screening_v1',
      'model_version': 'camera-v1',
      'manifest_checksum': 'a' * 64,
      'training_data_label': 'research_only',
      'non_clinical': true,
      'created_at': '2026-07-29T09:10:00Z',
    }),
  );
  await db.into(db.measuredDetailRevisions).insert(
        MeasuredDetailRevisionsCompanion.insert(
          revisionUuid: '40000000-0000-0000-0000-000000000001',
          visitId: 1,
          revisionNumber: 1,
          beforeJson: '{}',
          afterJson: jsonEncode({
            'height_cm': 83.58,
            'measurement_mode': 'standing_height',
            'oedema': 'not_checked',
            'measured_at': '2026-07-29T10:00:00Z',
          }),
        ),
      );
  await dao.enqueue(
    ownerUserId: ownerUserId,
    visitUuid: visitUuid,
    entityType: 'measured_revision',
    entityUuid: '40000000-0000-0000-0000-000000000001',
    dependencyEntityUuid: visitUuid,
    payloadJson: jsonEncode({
      'revision_uuid': '40000000-0000-0000-0000-000000000001',
      'visit_uuid': visitUuid,
      'before': <String, dynamic>{},
      'after': {
        'height_cm': 83.58,
        'measurement_mode': 'standing_height',
        'oedema': 'not_checked',
        'measured_at': '2026-07-29T10:00:00Z',
      },
      'created_at': '2026-07-29T10:00:00Z',
    }),
  );
  expect(visitEntry.status, 'pending');
  return asset;
}

Future<SyncOutboxData> seedVisitOutbox(
  AppDatabase db,
  SyncOutboxDao dao, {
  required int ownerUserId,
  required String visitUuid,
}) async {
  if (await db.select(db.children).getSingleOrNull() == null) {
    await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Child 001',
            dateOfBirth: '2024-01-29',
            sex: 'F',
            ownerUserId: Value(ownerUserId),
          ),
        );
    await db.into(db.visits).insert(
          VisitsCompanion.insert(
            childId: 1,
            localUuid: visitUuid,
            visitDate: Value(DateTime(2026, 7, 29)),
            ageMonths: 30,
            ownerUserId: Value(ownerUserId),
            entryMethod: const Value('guided_capture'),
            captureState: const Value('draft_capture'),
            deviceMetadataJson: const Value('{}'),
            consentVersion: const Value('guided_capture_consent_v1'),
            consentTimestamp: Value(DateTime.utc(2026, 7, 29, 9)),
            consentOperatorIdentifier: const Value('operator-7'),
          ),
        );
  }
  return dao.enqueue(
    ownerUserId: ownerUserId,
    visitUuid: visitUuid,
    entityType: 'visit',
    entityUuid: visitUuid,
    payloadJson: jsonEncode({
      'local_uuid': visitUuid,
      'child_id': 1,
      'visit_date': '2026-07-29T00:00:00',
      'age_months': 30,
      'capture_state': 'draft_capture',
      'device_metadata': <String, dynamic>{},
      'consent_version': 'guided_capture_consent_v1',
      'consent_timestamp': '2026-07-29T09:00:00Z',
      'consent_operator_identifier': 'operator-7',
    }),
  );
}

Future<SeededAsset> seedAssetOutbox(
  AppDatabase db,
  SyncOutboxDao dao,
  Directory root, {
  required int ownerUserId,
  required String visitUuid,
  required bool acknowledgeVisit,
}) async {
  SyncOutboxData? visitEntry;
  if (await db.select(db.visits).getSingleOrNull() == null) {
    visitEntry = await seedVisitOutbox(
      db,
      dao,
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
  } else {
    visitEntry = await (db.select(db.syncOutbox)
          ..where((row) => row.entityType.equals('visit')))
        .getSingleOrNull();
  }
  if (acknowledgeVisit && visitEntry != null) {
    await dao.acknowledge(
      ownerUserId,
      visitEntry.id,
      jsonEncode({
        'entity_type': 'visit',
        'entity_uuid': visitUuid,
        'status': 'accepted',
        'server_id': 42,
      }),
    );
  }
  final frontBytes = utf8.encode('front image');
  final sideBytes = utf8.encode('side image');
  final front = File('${root.path}/images/front.jpg');
  final side = File('${root.path}/images/side.jpg');
  await front.parent.create(recursive: true);
  await front.writeAsBytes(frontBytes);
  await side.writeAsBytes(sideBytes);
  for (final item in [
    (
      uuid: '20000000-0000-0000-0000-000000000001',
      role: 'front',
      path: front.path,
    ),
    (
      uuid: '20000000-0000-0000-0000-000000000002',
      role: 'side',
      path: side.path,
    ),
  ]) {
    await db.into(db.captureAssets).insert(
          CaptureAssetsCompanion.insert(
            assetUuid: item.uuid,
            visitId: 1,
            role: item.role,
            localPath: Value(item.path),
            capturedAt: DateTime.utc(2026, 7, 29, 9, 5),
            qualityVerdict: const Value('accepted'),
          ),
        );
    await dao.enqueue(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      entityType: 'capture_asset',
      entityUuid: item.uuid,
      dependencyEntityUuid: visitUuid,
      payloadJson: jsonEncode({
        'asset_uuid': item.uuid,
        'visit_uuid': visitUuid,
        'role': item.role,
        'captured_at': '2026-07-29T09:05:00Z',
        'selected_rank': 0,
        'quality': {
          'pose': 0.9,
          'coverage': 0.9,
          'orientation': 0.9,
          'sharpness': 0.9,
          'lighting': 0.9,
          'overall': 0.9,
          'threshold_version': 'guided_capture_quality_v1',
        },
        'device_camera_metadata': <String, dynamic>{},
      }),
    );
  }
  return SeededAsset(
    frontPath: front.path,
    sidePath: side.path,
    frontBytes: frontBytes,
  );
}
