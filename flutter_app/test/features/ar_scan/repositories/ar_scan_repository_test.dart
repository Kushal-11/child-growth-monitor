import 'dart:convert';

import 'package:child_growth_monitor_app/database/daos/sync_outbox_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:child_growth_monitor_app/features/ar_scan/repositories/ar_scan_repository.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('full AR evidence updates owner-scoped visit and outbox atomically',
      () async {
    final database = AppDatabase.forTesting(NativeDatabase.memory());
    addTearDown(database.close);
    const ownerUserId = 7;
    const visitUuid = '10000000-0000-0000-0000-000000000001';
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'Child 001',
            dateOfBirth: '2023-01-01',
            sex: 'F',
            ownerUserId: const Value(ownerUserId),
          ),
        );
    await database.into(database.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: visitUuid,
            ageMonths: 36,
            ownerUserId: const Value(ownerUserId),
            deviceMetadataJson: const Value('{"device":"pixel"}'),
          ),
        );
    await SyncOutboxDao(database).enqueue(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      entityType: SyncOutboxEntityType.visit,
      entityUuid: visitUuid,
      payloadJson: '{"device_metadata":{"device":"pixel"}}',
    );
    const result = FullArScanResult(
      estimatedHeightCm: 88.1,
      uncertaintyCm: 0.6,
      acceptedKeyframes: 20,
      validDepthFraction: 0.45,
      meanDepthConfidence: 0.82,
      scanCoverageDegrees: 41,
      cameraTravelMeters: 0.7,
      floorStabilityCm: 1.2,
      capturedBodyPoints: 5000,
      durationMs: 14000,
      qualityScore: 0.9,
      depthMode: 'raw_depth_with_confidence',
    );

    final repository = DriftArScanRepository(database);
    final context = await repository.getVisitContext(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    expect(context.ageMonths, 36);
    expect(context.sex, 'F');

    await repository.saveExperimentalResult(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      result: result,
    );

    final visit = await (database.select(database.visits)
          ..where((row) => row.localUuid.equals(visitUuid)))
        .getSingle();
    final visitMetadata =
        jsonDecode(visit.deviceMetadataJson!) as Map<String, dynamic>;
    final persisted =
        visitMetadata['arcore_depth_scan'] as Map<String, dynamic>;
    expect(persisted['method'], contactlessArMethodV3);
    expect(persisted['clinical_measurement_eligible'], isFalse);
    expect(persisted['raw_media_retained'], isFalse);

    final outbox = await (database.select(database.syncOutbox)
          ..where((row) => row.entityUuid.equals(visitUuid)))
        .getSingle();
    final outboxPayload =
        jsonDecode(outbox.payloadJson) as Map<String, dynamic>;
    expect(
      outboxPayload['device_metadata']['arcore_depth_scan']['quality_score'],
      0.9,
    );
    expect(
      outbox.payloadChecksum,
      SyncOutboxDao.checksumForPayload(outbox.payloadJson),
    );
  });

  test('owner mismatch cannot write AR evidence', () async {
    final database = AppDatabase.forTesting(NativeDatabase.memory());
    addTearDown(database.close);
    const visitUuid = '10000000-0000-0000-0000-000000000002';
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'Child 002',
            dateOfBirth: '2023-01-01',
            sex: 'M',
            ownerUserId: const Value(7),
          ),
        );
    await database.into(database.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: visitUuid,
            ageMonths: 36,
            ownerUserId: const Value(7),
          ),
        );
    const result = FullArScanResult(
      estimatedHeightCm: 88.1,
      uncertaintyCm: 0.6,
      acceptedKeyframes: 20,
      validDepthFraction: 0.45,
      meanDepthConfidence: 0.82,
      scanCoverageDegrees: 41,
      cameraTravelMeters: 0.7,
      floorStabilityCm: 1.2,
      capturedBodyPoints: 5000,
      durationMs: 14000,
      qualityScore: 0.9,
      depthMode: 'raw_depth_with_confidence',
    );

    expect(
      () => DriftArScanRepository(database).saveExperimentalResult(
        ownerUserId: 8,
        visitUuid: visitUuid,
        result: result,
      ),
      throwsStateError,
    );
  });
}
