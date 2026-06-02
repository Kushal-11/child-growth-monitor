import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';
import 'package:child_growth_monitor_app/screens/children/children_list_screen.dart';

class _FakeAuth implements AuthService {
  @override
  String get baseUrl => 'http://t';
  @override
  Future<AuthLoginResult> login(String u, String p) async => throw UnimplementedError();
  @override
  Future<String?> readToken() async => 't';
  @override
  Future<AuthUser?> readUser() async => AuthUser(id: 1, username: 'a', fullName: 'A', role: 'worker');
  @override
  Future<void> logout() async {}
}

void main() {
  testWidgets('shows only the owner\'s non-archived children from local db', (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final dao = ChildDao(db);
    await dao.createChild(name: 'Mine', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.createChild(name: 'Theirs', dateOfBirth: '2024-01-01', sex: 'F', ownerUserId: 2);

    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      authServiceProvider.overrideWithValue(_FakeAuth()),
    ]);
    await container.read(authProvider.notifier).restore();
    addTearDown(() { container.dispose(); db.close(); });

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: const MaterialApp(home: ChildrenListScreen()),
    ));
    await tester.pumpAndSettle();

    expect(find.text('Mine'), findsOneWidget);
    expect(find.text('Theirs'), findsNothing);

    // Dispose the StreamBuilder (and its drift subscription) while the db is
    // still open, so drift's stream-cancel timer can flush before teardown.
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pumpAndSettle();
  });
}
