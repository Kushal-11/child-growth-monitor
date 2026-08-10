import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:child_growth_monitor_app/features/ar_scan/providers/ar_scan_provider.dart';
import 'package:child_growth_monitor_app/features/ar_scan/repositories/ar_scan_repository.dart';
import 'package:child_growth_monitor_app/features/ar_scan/services/ar_scan_platform.dart';
import 'package:child_growth_monitor_app/features/ar_scan/widgets/ar_scan_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

const _result = FullArScanResult(
  estimatedHeightCm: 88.1,
  uncertaintyCm: 0.6,
  acceptedKeyframes: 20,
  validDepthFraction: 0.45,
  meanDepthConfidence: 0.82,
  scanCoverageDegrees: 41,
  cameraTravelMeters: 0.7,
  floorStabilityCm: 1.2,
  capturedBodyPoints: 5000,
  durationMs: 14000,
  qualityScore: 0.9,
  depthMode: 'raw_depth_with_confidence',
);

class FakeArPlatform implements ArScanPlatform {
  FakeArPlatform({required this.supported, this.result});

  final bool supported;
  final FullArScanResult? result;
  int scanCalls = 0;
  double? receivedAgeMonths;
  String? receivedSex;

  @override
  Future<ArScanCapability> checkCapability() async => ArScanCapability(
        availability: supported
            ? 'supported_installed'
            : 'unsupported_device_not_capable',
        arSupported: supported,
        transient: false,
        ramMb: 512,
      );

  @override
  Future<FullArScanResult?> startFullScan({
    double? ageMonths,
    String? sex,
  }) async {
    scanCalls++;
    receivedAgeMonths = ageMonths;
    receivedSex = sex;
    return result;
  }
}

class RecordingArRepository implements ArScanRepository {
  RecordingArRepository({this.entryMethod = 'guided_capture'});

  final String entryMethod;
  FullArScanResult? savedResult;

  @override
  Future<ArScanVisitContext> getVisitContext({
    required int ownerUserId,
    required String visitUuid,
  }) async =>
      ArScanVisitContext(ageMonths: 30, sex: 'F', entryMethod: entryMethod);

  @override
  Future<void> saveExperimentalResult({
    required int ownerUserId,
    required String visitUuid,
    required FullArScanResult result,
  }) async {
    savedResult = result;
  }
}

Widget testApp(
  FakeArPlatform platform,
  RecordingArRepository repository, {
  ArScanPostProcessor? postProcessor,
}) =>
    ProviderScope(
      overrides: [
        arScanPlatformProvider.overrideWithValue(platform),
        arScanRepositoryProvider.overrideWithValue(repository),
        arScanPostProcessorProvider.overrideWithValue(
          postProcessor ??
              ({required int ownerUserId, required String visitUuid}) async =>
                  null,
        ),
      ],
      child: const MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: ArScanCard(ownerUserId: 7, visitUuid: 'visit-uuid'),
          ),
        ),
      ),
    );

void main() {
  testWidgets('unsupported phones retain guided photo fallback',
      (tester) async {
    final platform = FakeArPlatform(supported: false);
    final repository = RecordingArRepository();
    await tester.pumpWidget(testApp(platform, repository));
    await tester.pumpAndSettle();
    expect(find.text('Standard guided photos'), findsOneWidget);
    expect(find.text('Start guided depth scan'), findsNothing);
  });

  testWidgets('successful scan is saved and disclosed as an estimate',
      (tester) async {
    final platform = FakeArPlatform(supported: true, result: _result);
    final repository = RecordingArRepository();
    await tester.pumpWidget(testApp(platform, repository));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Start guided depth scan'));
    await tester.pumpAndSettle();
    expect(platform.scanCalls, 1);
    expect(platform.receivedAgeMonths, 30);
    expect(platform.receivedSex, 'F');
    expect(repository.savedResult, same(_result));
    expect(find.textContaining('Estimated height 88.1'), findsOneWidget);
    expect(find.textContaining('These values are estimates'), findsOneWidget);
  });

  testWidgets('assessment scan displays the processed AR geometry weight',
      (tester) async {
    final platform = FakeArPlatform(supported: true, result: _result);
    final repository = RecordingArRepository(entryMethod: 'assessment');
    await tester.pumpWidget(
      testApp(
        platform,
        repository,
        postProcessor: ({
          required int ownerUserId,
          required String visitUuid,
        }) async =>
            const ArScanProcessedResult(
          estimatedWeightKg: 12.4,
          weightRangeLowerKg: 11.8,
          weightRangeUpperKg: 13,
          weightSource: 'arcore_geometry_ml_weight_v3',
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.text('Start guided depth scan'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Estimated weight 12.4 kg'), findsOneWidget);
    expect(find.textContaining('11.8–13.0 kg range'), findsOneWidget);
  });
}
