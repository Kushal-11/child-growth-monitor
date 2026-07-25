import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/screens/child_management/manual_measurement_screen.dart';
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
  testWidgets('manual measurement requires height and weight', (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final childId = await ChildDao(db).createChild(
      name: 'Owned child',
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

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: MaterialApp(home: ManualMeasurementScreen(childId: childId)),
    ));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const Key('measure_save')));
    await tester.tap(find.byKey(const Key('measure_save')));
    await tester.pumpAndSettle();
    expect(find.text('Required'), findsNWidgets(2)); // height + weight
  });

  testWidgets('manual-measure deep link rejects another owner child',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final childId = await ChildDao(db).createChild(
      name: 'Private child',
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

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: MaterialApp(home: ManualMeasurementScreen(childId: childId)),
    ));
    await tester.pumpAndSettle();
    expect(
      find.text('Child not found for the signed-in user.'),
      findsOneWidget,
    );
    expect(find.byKey(const Key('measure_height')), findsNothing);
  });
}
