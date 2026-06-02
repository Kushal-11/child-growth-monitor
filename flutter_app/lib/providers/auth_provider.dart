import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/auth_service.dart';
import 'api_provider.dart';

enum AuthStatus { unknown, unauthenticated, authenticated }

class AuthState {
  const AuthState({this.status = AuthStatus.unknown, this.user, this.token});

  final AuthStatus status;
  final AuthUser? user;
  final String? token;

  AuthState copyWith({AuthStatus? status, AuthUser? user, String? token}) =>
      AuthState(
        status: status ?? this.status,
        user: user ?? this.user,
        token: token ?? this.token,
      );

  static const unauthenticated = AuthState(status: AuthStatus.unauthenticated);
}

/// Overridable so widgets/tests can inject a configured AuthService.
final authServiceProvider = Provider<AuthService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  return AuthService(baseUrl: effectiveBaseUrl(baseUrl));
});

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._service) : super(const AuthState());

  final AuthService _service;

  /// Loads any cached token/user. Offline-tolerant: presence of a token = authenticated.
  Future<void> restore() async {
    try {
      final token = await _service.readToken();
      final user = await _service.readUser();
      if (token != null && user != null) {
        state = AuthState(status: AuthStatus.authenticated, user: user, token: token);
      } else {
        state = AuthState.unauthenticated;
      }
    } catch (_) {
      // Corrupt/unavailable secure storage must not strand the app on a
      // protected screen — fall back to requiring login.
      state = AuthState.unauthenticated;
    }
  }

  Future<void> login(String username, String password) async {
    try {
      final result = await _service.login(username, password);
      state = AuthState(
        status: AuthStatus.authenticated,
        user: result.user,
        token: result.token,
      );
    } catch (_) {
      // Surface the error to the caller, but ensure the auth state reflects
      // that no session is active (do not leave it in the `unknown` state).
      state = AuthState.unauthenticated;
      rethrow;
    }
  }

  Future<void> logout() async {
    await _service.logout();
    state = AuthState.unauthenticated;
  }

  /// Called when the backend rejects the token (401 during sync).
  Future<void> onTokenRejected() async {
    await _service.logout();
    state = AuthState.unauthenticated;
  }
}

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier(ref.watch(authServiceProvider));
});
