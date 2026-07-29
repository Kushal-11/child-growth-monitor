import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/constants/feature_flags.dart';

void main() {
  group('computeLiveCaptureEnabled', () {
    // In a test run kDebugMode is always true, which would mask the flag —
    // so the release-build truth table is exercised through the pure gate,
    // same as LocalAuth.computeOfflineAuthEnabled.
    test('enabled in debug builds regardless of flag', () {
      expect(FeatureFlags.computeLiveCaptureEnabled(true, false), isTrue);
      expect(FeatureFlags.computeLiveCaptureEnabled(true, true), isTrue);
    });

    test('release builds require the LIVE_CAPTURE dart-define', () {
      expect(FeatureFlags.computeLiveCaptureEnabled(false, true), isTrue);
      expect(FeatureFlags.computeLiveCaptureEnabled(false, false), isFalse);
    });

    test('test/debug runtime is enabled while the normal release stays gated',
        () {
      expect(FeatureFlags.liveCaptureEnabled, isTrue);
      expect(
        FeatureFlags.computeLiveCaptureEnabled(
          false,
          false,
        ),
        isFalse,
        reason: 'A plain release must not expose unvalidated live capture',
      );
    });
  });
}
