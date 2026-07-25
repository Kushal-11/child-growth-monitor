import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/manual_visit_dao.dart';
import 'package:child_growth_monitor_app/models/child_detail.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/children_provider.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

class _FakeAuth implements AuthService {
  _FakeAuth([this.userId = 1]);
  final int userId;
  @override
  String get baseUrl => 'http://test';
  @override
  Future<AuthLoginResult> login(String username, String password) async =>
      throw UnimplementedError();
  @override
  Future<void> logout() async {}
  @override
  Future<String?> readToken() async => 'token';
  @override
  Future<AuthUser?> readUser() async => AuthUser(
        id: userId,
        username: 'owner',
        fullName: 'Owner',
        role: 'worker',
      );
}

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
      authServiceProvider.overrideWithValue(_FakeAuth()),
    ]);
    await container.read(authProvider.notifier).restore();
    addTearDown(() {
      container.dispose();
      db.close();
    });
    final detail = await container.read(childDetailProvider(childId).future);
    expect(detail.visits, isNotEmpty);
    final m = detail.visits.first.measurement;
    expect(m, isNotNull);
    expect(m!.predictedHeightCm, isNull);
    expect(m.predictedWeightKg, isNull);
    expect(m.manualHeightCm, 80.0);
    expect(m.manualWeightKg, 10.5);
    expect(m.effectiveHeightCm, 80.0);
    expect(m.effectiveWeightKg, 10.5);
    expect(m.displayHeightCm, 80.0);
    expect(m.displayWeightKg, 10.5);
    expect(m.heightSource, 'manual');
    expect(m.weightSource, 'manual');
    expect(m.bmi, isNotNull);
    expect(m.muacStatus, 'Indeterminate');
    expect(m.poshanStatus, 'Indeterminate');
    expect(m.classificationMethod, 'poshan_setu_v1');
  });

  test('childDetailProvider reacts when a visit is inserted', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final childDao = ChildDao(db);
    final manualDao = ManualVisitDao(db);
    final childId = await childDao.createChild(
      name: 'Reactive child',
      dateOfBirth: '2024-01-01',
      sex: 'M',
      ownerUserId: 1,
    );
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      authServiceProvider.overrideWithValue(_FakeAuth()),
    ]);
    await container.read(authProvider.notifier).restore();
    addTearDown(() {
      container.dispose();
      db.close();
    });

    final visitArrived = Completer<ChildDetail>();
    final subscription = container.listen(
      childDetailProvider(childId),
      (_, next) {
        next.whenData((detail) {
          if (detail.visits.length == 1 && !visitArrived.isCompleted) {
            visitArrived.complete(detail);
          }
        });
      },
      fireImmediately: true,
    );
    addTearDown(subscription.close);
    expect(
      (await container.read(childDetailProvider(childId).future)).visits,
      isEmpty,
    );

    await manualDao.createManualVisit(
      childId: childId,
      ownerUserId: 1,
      ageMonths: 18,
      visitDate: DateTime(2025, 7, 1),
      heightCm: 80,
      weightKg: 10,
    );

    final updated = await visitArrived.future.timeout(
      const Duration(seconds: 2),
    );
    expect(updated.visits.single.measurement!.displayHeightCm, 80);
  });

  test('childDetailProvider rejects another owner deep link', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final childId = await ChildDao(db).createChild(
      name: 'Owner one',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      ownerUserId: 1,
    );
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      authServiceProvider.overrideWithValue(_FakeAuth(2)),
    ]);
    await container.read(authProvider.notifier).restore();
    addTearDown(() {
      container.dispose();
      db.close();
    });

    await expectLater(
      container.read(childDetailProvider(childId).future),
      throwsStateError,
    );
  });
}
