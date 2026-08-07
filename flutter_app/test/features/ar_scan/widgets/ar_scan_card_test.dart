import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:child_growth_monitor_app/features/ar_scan/providers/ar_scan_provider.dart';
import 'package:child_growth_monitor_app/features/ar_scan/repositories/ar_scan_repository.dart';
import 'package:child_growth_monitor_app/features/ar_scan/services/ar_scan_platform.dart';
import 'package:child_growth_monitor_app/features/ar_scan/widgets/ar_scan_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

class UnsupportedArPlatform implements ArScanPlatform {
  @override
  Future<ArScanCapability> checkCapability() async => const ArScanCapability(
        availability: 'unsupported_device_not_capable',
        arSupported: false,
        transient: false,
        ramMb: 256,
      );

  @override
  Future<SparseArScanResult?> startSparseScan() => throw UnimplementedError();
}

class NoopArRepository implements ArScanRepository {
  @override
  Future<void> saveExperimentalResult({
    required int ownerUserId,
    required String visitUuid,
    required SparseArScanResult result,
  }) async {}
}

void main() {
  testWidgets('unsupported phones retain lightweight guided fallback',
      (tester) async {
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          arScanPlatformProvider.overrideWithValue(UnsupportedArPlatform()),
          arScanRepositoryProvider.overrideWithValue(NoopArRepository()),
        ],
        child: const MaterialApp(
          home: Scaffold(
            body: ArScanCard(ownerUserId: 7, visitUuid: 'visit-uuid'),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('Standard guided photos'), findsOneWidget);
    expect(find.text('Start depth scan'), findsNothing);
  });
}
