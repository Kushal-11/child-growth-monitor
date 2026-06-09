import 'auth_service.dart';

/// Offline, hardcoded credential source for field-test builds.
///
/// WARNING: This password is compiled into the app. It is acceptable for a
/// field-test build ONLY and is NOT a real secret. Remove or gate behind a
/// debug/flavor flag before any production or public release.
class LocalAuth {
  LocalAuth._();

  static const String _username = 'cgmtester@test.com';
  static const String _password = 'cgmtester';
  static const int _userId = 9001;

  /// Fixed identity for the offline field tester. The stable [_userId] (9001)
  /// is used to owner-scope locally created data so it can be reconciled with a
  /// real account once online sync is enabled.
  static final AuthUser _fixedUser = AuthUser(
    id: _userId,
    username: _username,
    fullName: 'CGM Field Tester',
    role: 'field_worker',
  );

  /// Synthetic, clearly-non-server token. A real backend will reject this with
  /// a 401, which the sync layer already handles gracefully.
  static const String _localToken = 'local-$_userId';

  /// Returns a login result for the hardcoded tester, or null if the
  /// credential does not match. Username is trimmed and compared
  /// case-insensitively; password is exact. Pure function, no I/O.
  static AuthLoginResult? tryLogin(String username, String password) {
    final normalized = username.trim().toLowerCase();
    if (normalized == _username && password == _password) {
      return AuthLoginResult(token: _localToken, user: _fixedUser);
    }
    return null;
  }
}
