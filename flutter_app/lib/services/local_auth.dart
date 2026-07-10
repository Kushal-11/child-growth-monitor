import 'package:flutter/foundation.dart';

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

  /// Compile-time opt-in for offline login in a non-debug build. Build a
  /// release/profile field APK with `--dart-define=FIELD_OFFLINE_AUTH=true` to
  /// keep the offline tester login available without shipping a slow, insecure
  /// debug build. Defaults to `false`, so an ordinary release build still
  /// contains no usable backdoor.
  static const bool _fieldOfflineAuth =
      bool.fromEnvironment('FIELD_OFFLINE_AUTH');

  /// The default gate for [tryLogin]: offline login is available when the build
  /// is a debug build OR was explicitly opted in via the field flag. Pure and
  /// side-effect-free so the release-build truth table is unit-testable (in a
  /// test run [kDebugMode] is always true, which would otherwise mask the flag).
  static bool computeOfflineAuthEnabled(bool debugMode, bool fieldFlag) =>
      debugMode || fieldFlag;

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
  ///
  /// The backdoor is gated by [enabled]. When omitted it resolves to
  /// [computeOfflineAuthEnabled] of [kDebugMode] and the `FIELD_OFFLINE_AUTH`
  /// compile flag: available in debug builds, and in release/profile builds
  /// only when explicitly built with `--dart-define=FIELD_OFFLINE_AUTH=true`.
  /// A plain release build authenticates no one. Pass [enabled] explicitly to
  /// override (e.g. `false` in tests).
  static AuthLoginResult? tryLogin(
    String username,
    String password, {
    bool? enabled,
  }) {
    final gate =
        enabled ?? computeOfflineAuthEnabled(kDebugMode, _fieldOfflineAuth);
    if (!gate) return null;
    final normalized = username.trim().toLowerCase();
    if (normalized == _username && password == _password) {
      return AuthLoginResult(token: _localToken, user: _fixedUser);
    }
    return null;
  }
}
