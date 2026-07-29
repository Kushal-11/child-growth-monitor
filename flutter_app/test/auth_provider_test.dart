import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

class _FakeAuthService implements AuthService {
  _FakeAuthService();
  String? _token;
  AuthUser? _user;

  @override
  String get baseUrl => 'http://test';

  @override
  Future<AuthLoginResult> login(String username, String password) async {
    if (password != 'good') throw AuthException('bad', statusCode: 401);
    _token = 'tok';
    _user = AuthUser(id: 1, username: username, fullName: 'X', role: 'worker');
    return AuthLoginResult(token: _token!, user: _user!);
  }

  @override
  Future<String?> readToken() async => _token;
  @override
  Future<AuthUser?> readUser() async => _user;
  @override
  Future<void> logout() async {
    _token = null;
    _user = null;
  }
}

void main() {
  test('initial restore with no token => unauthenticated', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_FakeAuthService()),
    ]);
    addTearDown(container.dispose);
    await container.read(authProvider.notifier).restore();
    expect(container.read(authProvider).status, AuthStatus.unauthenticated);
  });

  test('login success => authenticated with user', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_FakeAuthService()),
    ]);
    addTearDown(container.dispose);
    await container.read(authProvider.notifier).login('asha', 'good');
    final state = container.read(authProvider);
    expect(state.status, AuthStatus.authenticated);
    expect(state.user?.username, 'asha');
    expect(state.token, 'tok');
  });

  test('login failure keeps unauthenticated and surfaces error', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_FakeAuthService()),
    ]);
    addTearDown(container.dispose);
    await expectLater(
      container.read(authProvider.notifier).login('asha', 'bad'),
      throwsA(isA<AuthException>()),
    );
    expect(container.read(authProvider).status, AuthStatus.unauthenticated);
  });

  test('restore tolerates storage errors and lands unauthenticated', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_ThrowingAuthService()),
    ]);
    addTearDown(container.dispose);
    await container.read(authProvider.notifier).restore();
    expect(container.read(authProvider).status, AuthStatus.unauthenticated);
  });
}

class _ThrowingAuthService implements AuthService {
  @override
  String get baseUrl => 'http://test';
  @override
  Future<AuthLoginResult> login(String u, String p) async =>
      throw AuthException('x');
  @override
  Future<String?> readToken() async => throw Exception('keystore boom');
  @override
  Future<AuthUser?> readUser() async => null;
  @override
  Future<void> logout() async {}
}
