import 'package:child_growth_monitor_app/database/daos/sync_outbox_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late AppDatabase db;
  late SyncOutboxDao dao;

  const ownerUserId = 7;
  const visitUuid = '10000000-0000-0000-0000-000000000001';

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    dao = SyncOutboxDao(db);
  });

  tearDown(() => db.close());

  Future<SyncOutboxData> enqueue(
    String type,
    String uuid, {
    String? dependency,
  }) {
    return dao.enqueue(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      entityType: type,
      entityUuid: uuid,
      dependencyEntityUuid: dependency,
      payloadJson: '{"entity":"$uuid"}',
    );
  }

  test('dependencies enforce visit, asset, result and deletion order',
      () async {
    final visit = await enqueue('visit', visitUuid);
    final front = await enqueue(
      'capture_asset',
      '20000000-0000-0000-0000-000000000001',
      dependency: visitUuid,
    );
    final side = await enqueue(
      'capture_asset',
      '20000000-0000-0000-0000-000000000002',
      dependency: visitUuid,
    );
    final result = await enqueue(
      'camera_result',
      '30000000-0000-0000-0000-000000000001',
      dependency: visitUuid,
    );
    final revision = await enqueue(
      'measured_revision',
      '40000000-0000-0000-0000-000000000001',
      dependency: visitUuid,
    );
    final deletion = await enqueue(
      'media_deletion',
      '50000000-0000-0000-0000-000000000001',
      dependency: front.entityUuid,
    );

    expect(
      (await dao.readyForSync(ownerUserId)).map((row) => row.entityUuid),
      [visitUuid],
    );

    await dao.acknowledge(
      ownerUserId,
      visit.id,
      '{"server_visit_id":42}',
    );
    expect(
      (await dao.readyForSync(ownerUserId))
          .map((row) => row.entityUuid)
          .toSet(),
      {front.entityUuid, side.entityUuid, revision.entityUuid},
    );

    await dao.acknowledge(ownerUserId, front.id, '{"asset":"front"}');
    await dao.acknowledge(ownerUserId, side.id, '{"asset":"side"}');
    await dao.acknowledge(ownerUserId, revision.id, '{"revision":1}');
    expect(
      (await dao.readyForSync(ownerUserId))
          .map((row) => row.entityUuid)
          .toSet(),
      {result.entityUuid, deletion.entityUuid},
    );
  });

  test('retry preserves entity UUID, payload and checksum', () async {
    final entry = await enqueue('visit', visitUuid);
    await dao.markFailed(ownerUserId, entry.id, 'offline');
    final failed = await (db.select(db.syncOutbox)
          ..where((row) => row.id.equals(entry.id)))
        .getSingle();

    expect(failed.entityUuid, entry.entityUuid);
    expect(failed.payloadJson, entry.payloadJson);
    expect(failed.payloadChecksum, entry.payloadChecksum);
    expect(failed.retryCount, 1);
    expect(failed.status, 'failed');
  });

  test('owner scope excludes another worker outbox', () async {
    await enqueue('visit', visitUuid);
    await dao.enqueue(
      ownerUserId: 8,
      visitUuid: '10000000-0000-0000-0000-000000000008',
      entityType: 'visit',
      entityUuid: '10000000-0000-0000-0000-000000000008',
      payloadJson: '{}',
    );

    final ready = await dao.readyForSync(ownerUserId);
    expect(ready.map((row) => row.ownerUserId).toSet(), {ownerUserId});
  });
}
