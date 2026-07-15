import 'package:flutter/foundation.dart';

/// Compile-time feature gates, following the LocalAuth FIELD_OFFLINE_AUTH
/// pattern: on by default in debug builds, opt-in per release build via
/// `--dart-define`.
class FeatureFlags {
  FeatureFlags._();

  /// Build a release/profile field APK with `--dart-define=LIVE_CAPTURE=true`
  /// to enable the in-app live capture screen (camera preview with real-time
  /// pose guidance). Plain release builds keep the proven image_picker flow
  /// until live capture is field-validated.
  static const bool _liveCaptureFlag = bool.fromEnvironment('LIVE_CAPTURE');

  /// Pure gate so the release-build truth table is unit-testable (in a test
  /// run [kDebugMode] is always true, which would otherwise mask the flag).
  static bool computeLiveCaptureEnabled(bool debugMode, bool fieldFlag) =>
      debugMode || fieldFlag;

  static bool get liveCaptureEnabled =>
      computeLiveCaptureEnabled(kDebugMode, _liveCaptureFlag);
}
