import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/manual_visit_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/children_provider.dart';

void main() {
  test('childDetailProvider surfaces manual height/weight', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final childDao = ChildDao(db);
    final manualDao = ManualVisitDao(db);
    final childId = await childDao.createChild(
        name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await manualDao.createManualVisit(
      childId: childId,
      ownerUserId: 1,
      ageMonths: 18.0,
      visitDate: DateTime(2026, 6, 1),
      heightCm: 80.0,
      weightKg: 10.5,
      hazZscore: -1.0,
      whzZscore: -0.5,
      hazStatus: 'Normal',
      whzStatus: 'Normal',
    );
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
    ]);
    addTearDown(() {
      container.dispose();
      db.close();
    });
    final detail = await container.read(childDetailProvider(childId).future);
    expect(detail.visits, isNotEmpty);
    final m = detail.visits.first.measurement;
    expect(m, isNotNull);
    // effective height/weight must reflect the manual values
    expect(m!.predictedHeightCm, 80.0);
    expect(m.predictedWeightKg, 10.5);
    expect(m.heightMethod, 'manual');
    expect(m.weightMethod, 'manual');
  });

  test('childDetailProvider retains labelled photo estimates', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final childDao = ChildDao(db);
    final visitDao = VisitDao(db);
    final childId = await childDao.createChild(
      name: 'Kid',
      dateOfBirth: '2023-01-01',
      sex: 'F',
      ownerUserId: 1,
    );
    await visitDao.createWithMeasurement(
      childId: childId,
      ageMonths: 42,
      imagePath: '/tmp/front.jpg',
      measurement: const MeasurementsCompanion(
        predictedHeightCm: Value(98.2),
        effectiveHeightCm: Value(98.2),
        heightMethod: Value('who_statistical'),
        predictedWeightKg: Value(14.1),
        effectiveWeightKg: Value(14.1),
        weightMethod: Value('ml_estimated'),
        muacCm: Value(15.2),
        muacMethod: Value('landmark_estimated'),
        confidenceScore: Value(0.94),
      ),
    );
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
    ]);
    addTearDown(() {
      container.dispose();
      db.close();
    });

    final detail = await container.read(childDetailProvider(childId).future);
    final m = detail.visits.single.measurement;
    expect(m?.predictedHeightCm, 98.2);
    expect(m?.heightMethod, 'who_statistical');
    expect(m?.predictedWeightKg, 14.1);
    expect(m?.weightMethod, 'ml_estimated');
    expect(m?.muacCm, 15.2);
    expect(m?.muacMethod, 'landmark_estimated');
    expect(m?.confidenceScore, 0.94);
    expect(m?.hazStatus, isNull);
    expect(m?.whzStatus, isNull);
  });
}
