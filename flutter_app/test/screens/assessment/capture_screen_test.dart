import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/burst_frame_ranker.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/device_metadata_service.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/frame_quality_service.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_camera_controller.dart';
import 'package:child_growth_monitor_app/screens/assessment/capture_screen.dart';
import 'package:child_growth_monitor_app/services/capture_quality.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

class WidgetCameraGateway implements GuidedCameraGateway {
  WidgetCameraGateway({this.failInitialization = false});
  final bool failInitialization;
  GuidedLiveFrameCallback? callback;

  @override
  GuidedCameraDescription get description => const GuidedCameraDescription(
        identifier: 'widget-camera',
        lensDirection: 'back',
        sensorOrientation: 90,
      );

  @override
  Widget buildPreview() =>
      const ColoredBox(key: Key('fake-camera-preview'), color: Colors.black);

  @override
  Future<void> initialize(DeviceOrientation orientation) async {
    if (failInitialization) {
      throw const GuidedCameraException('init', 'widget camera unavailable');
    }
  }

  @override
  Future<void> startLiveStream(GuidedLiveFrameCallback onFrame) async {
    callback = onFrame;
  }

  @override
  Future<void> stopLiveStream() async {}

  @override
  Future<GuidedTemporaryFrame> takePicture() async => GuidedTemporaryFrame(
        path: '/tmp/widget.jpg',
        capturedAt: DateTime.utc(2026, 7, 29),
        width: 1080,
        height: 1920,
        exifOrientation: 1,
      );

  @override
  Future<void> setCaptureOrientation(DeviceOrientation orientation) async {}

  @override
  Future<void> setTorch(bool enabled) async {}

  @override
  Future<void> dispose() async {}
}

class WidgetPoseEvaluator implements GuidedPoseEvaluator {
  CaptureQuality stillQuality = const CaptureQuality.accepted(
    poseScore: 1,
    coverageScore: 1,
    orientationScore: 1,
  );

  @override
  Future<CaptureQuality> evaluateLive(
    GuidedLiveFrame frame, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  }) async =>
      stillQuality;

  @override
  Future<CaptureQuality> evaluateStill(
    String path, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  }) async =>
      stillQuality;

  @override
  Future<void> dispose() async {}
}

class WidgetFrameQuality extends FrameQualityService {
  const WidgetFrameQuality();

  @override
  Future<FrameQualityResult> evaluateFile(String path) async =>
      const FrameQualityResult(
        brightnessScore: 1,
        contrastScore: 1,
        lightingScore: 1,
        sharpnessScore: 1,
        overallScore: 1,
        accepted: true,
        rejectionReason: null,
      );
}

class WidgetStorage implements GuidedCaptureStorage {
  @override
  Future<String> retain(
    GuidedTemporaryFrame frame, {
    required CaptureAssetRole role,
    required int selectedRank,
  }) async =>
      '/visit/${role.wireValue}-$selectedRank.jpg';

  @override
  Future<void> deleteRetained(String path) async {}

  @override
  Future<void> deleteTemporary(String path) async {}
}

GuidedCameraController widgetController(
  WidgetCameraGateway gateway,
  WidgetPoseEvaluator pose,
) =>
    GuidedCameraController(
      role: CaptureAssetRole.front,
      cameraGateway: gateway,
      poseEvaluator: pose,
      frameQualityService: const WidgetFrameQuality(),
      frameRanker: const BurstFrameRanker(),
      storage: WidgetStorage(),
      deviceMetadataService: DeviceMetadataService(
        tiltSource: const NoopDeviceTiltSource(),
        platformDescription: () => 'widget-test',
      ),
      stableFrameCount: 1,
      burstSize: 1,
      retainedFrameCount: 1,
    );

Widget app(GuidedCameraController controller, {SystemCameraPicker? picker}) {
  return ProviderScope(
    child: MaterialApp(
      home: CaptureScreen(
        role: 'front',
        controller: controller,
        systemCameraPicker: picker,
      ),
    ),
  );
}

void main() {
  testWidgets('renders gateway preview after successful initialization',
      (tester) async {
    final gateway = WidgetCameraGateway();
    final controller = widgetController(gateway, WidgetPoseEvaluator());

    await tester.pumpWidget(app(controller));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('fake-camera-preview')), findsOneWidget);
    expect(find.text('Position the child in the frame'), findsOneWidget);
  });

  testWidgets('validated burst shows retained photo confirmation',
      (tester) async {
    final gateway = WidgetCameraGateway();
    final controller = widgetController(gateway, WidgetPoseEvaluator());
    await tester.pumpWidget(app(controller));
    await tester.pumpAndSettle();

    await gateway.callback!(
      const GuidedLiveFrame(payload: 'frame', width: 720, height: 1280),
    );
    await tester.pumpAndSettle();

    expect(find.text('Use photo'), findsOneWidget);
    expect(find.text('Retake'), findsOneWidget);
  });

  testWidgets('system-camera fallback cannot bypass failed pose validation',
      (tester) async {
    final gateway = WidgetCameraGateway(failInitialization: true);
    final pose = WidgetPoseEvaluator()
      ..stillQuality =
          const CaptureQuality.blocked(CaptureIssue.wrongOrientation);
    final controller = widgetController(gateway, pose);

    await tester.pumpWidget(
      app(controller, picker: () async => '/tmp/system.jpg'),
    );
    await tester.pumpAndSettle();
    expect(find.text('Camera unavailable'), findsOneWidget);

    await tester.tap(find.text('Use system camera'));
    await tester.pumpAndSettle();

    expect(find.text('Use photo'), findsNothing);
    expect(find.text('Turn the child to match the requested view'),
        findsOneWidget);
  });
}
