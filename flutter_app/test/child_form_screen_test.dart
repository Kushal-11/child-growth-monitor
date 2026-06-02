import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';
import 'package:child_growth_monitor_app/screens/child_management/child_form_screen.dart';

class _FakeAuth implements AuthService {
  @override
  String get baseUrl => 'http://t';
  @override
  Future<AuthLoginResult> login(String u, String p) async =>
      throw UnimplementedError();
  @override
  Future<String?> readToken() async => 't';
  @override
  Future<AuthUser?> readUser() async =>
      AuthUser(id: 1, username: 'a', fullName: 'A', role: 'worker');
  @override
  Future<void> logout() async {}
}

void main() {
  testWidgets('new child form requires a name', (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
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
      child: const MaterialApp(home: ChildFormScreen()),
    ));
    await tester.pumpAndSettle();
    // Tap save with empty name -> validation error shown.
    await tester.ensureVisible(find.byKey(const Key('child_save')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('child_save')));
    await tester.pumpAndSettle();
    expect(find.text('Name is required'), findsOneWidget);
  });
}
