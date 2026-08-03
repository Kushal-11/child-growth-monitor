import 'package:child_growth_monitor_app/database/daos/capture_asset_dao.dart';
import 'package:child_growth_monitor_app/database/daos/guided_visit_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/repositories/guided_capture_repository.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('draft stores calendar-aware age at a month-end anniversary', () async {
    final database = AppDatabase.forTesting(NativeDatabase.memory());
    addTearDown(database.close);
    final visitDao = GuidedVisitDao(database);
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'Child 001',
            dateOfBirth: '2024-01-31',
            sex: 'F',
            ownerUserId: const Value(7),
          ),
        );
    final repository = DriftGuidedCaptureRepository(
      database: database,
      visitDao: visitDao,
      captureAssetDao: CaptureAssetDao(database),
    );
    final child = await repository.getOwnerChild(
      childId: childId,
      ownerUserId: 7,
    );

    await repository.createDraft(
      child: child!,
      visitUuid: '10000000-0000-0000-0000-000000000001',
      visitDate: DateTime(2024, 2, 29),
      deviceMetadataJson: '{}',
      consentVersion: 'guided_capture_consent_v1',
      consentTimestamp: DateTime.utc(2024, 2, 29),
      consentOperatorIdentifier: 'worker-7',
    );

    final visit = await visitDao.getByUuid(
      ownerUserId: 7,
      visitUuid: '10000000-0000-0000-0000-000000000001',
    );
    expect(visit!.ageMonths, 1);
  });
}
