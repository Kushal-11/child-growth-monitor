import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:child_growth_monitor_app/screens/auth/login_screen.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

class _FakeAuthService implements AuthService {
  @override
  String get baseUrl => 'http://test';
  @override
  Future<AuthLoginResult> login(String u, String p) async {
    if (p != 'good') throw AuthException('Invalid username or password', statusCode: 401);
    return AuthLoginResult(token: 't', user: AuthUser(id: 1, username: u, fullName: 'X', role: 'worker'));
  }
  @override
  Future<String?> readToken() async => null;
  @override
  Future<AuthUser?> readUser() async => null;
  @override
  Future<void> logout() async {}
}

void main() {
  testWidgets('shows error on bad login', (tester) async {
    await tester.pumpWidget(ProviderScope(
      overrides: [authServiceProvider.overrideWithValue(_FakeAuthService())],
      child: const MaterialApp(home: LoginScreen()),
    ));
    await tester.enterText(find.byKey(const Key('login_username')), 'asha');
    await tester.enterText(find.byKey(const Key('login_password')), 'bad');
    await tester.tap(find.byKey(const Key('login_submit')));
    await tester.pumpAndSettle();
    expect(find.textContaining('Invalid'), findsOneWidget);
  });
}
