import 'dart:io';

import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_thresholds.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/burst_frame_ranker.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/device_metadata_service.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/frame_quality_service.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_camera_controller.dart';
import 'package:child_growth_monitor_app/services/capture_quality.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

const goodLive = CaptureQuality.accepted(
  poseScore: 0.9,
  coverageScore: 0.8,
  orientationScore: 0.85,
);

const goodStill = FrameQualityResult(
  brightnessScore: 0.8,
  contrastScore: 0.8,
  lightingScore: 0.8,
  sharpnessScore: 0.9,
  overallScore: 0.85,
  accepted: true,
  rejectionReason: null,
);

class FakeCameraGateway implements GuidedCameraGateway {
  bool failInitialization = false;
  int? failBurstAt;
  int initializeCount = 0;
  int disposeCount = 0;
  int takeCount = 0;
  bool streaming = false;
  GuidedLiveFrameCallback? callback;
  final events = <String>[];
  final orientations = <DeviceOrientation>[];

  @override
  GuidedCameraDescription get description => const GuidedCameraDescription(
        identifier: 'rear-0',
        lensDirection: 'back',
        sensorOrientation: 90,
      );

  @override
  Widget buildPreview() => const ColoredBox(color: Color(0xff111111));

  @override
  Future<void> initialize(DeviceOrientation orientation) async {
    events.add('initialize');
    initializeCount++;
    if (failInitialization) {
      throw const GuidedCameraException('init_failed', 'camera unavailable');
    }
    orientations.add(orientation);
  }

  @override
  Future<void> startLiveStream(GuidedLiveFrameCallback onFrame) async {
    events.add('start');
    streaming = true;
    callback = onFrame;
  }

  @override
  Future<void> stopLiveStream() async {
    events.add('stop');
    streaming = false;
  }

  Future<void> emit() async {
    await callback?.call(
      const GuidedLiveFrame(payload: 'frame', width: 720, height: 1280),
    );
  }

  @override
  Future<GuidedTemporaryFrame> takePicture() async {
    final index = takeCount++;
    events.add('take:$index');
    if (failBurstAt == index) {
      throw const GuidedCameraException('burst_failed', 'capture failed');
    }
    return GuidedTemporaryFrame(
      path: '/tmp/frame-$index.jpg',
      capturedAt: DateTime.utc(2026, 7, 29, 10, 0, index),
      width: 1080,
      height: 1920,
      exifOrientation: 6,
    );
  }

  @override
  Future<void> setCaptureOrientation(DeviceOrientation orientation) async {
    events.add('rotate:${orientation.name}');
    orientations.add(orientation);
  }

  @override
  Future<void> setTorch(bool enabled) async {
    events.add('torch:$enabled');
  }

  @override
  Future<void> dispose() async {
    events.add('dispose');
    disposeCount++;
    streaming = false;
  }
}

class FakePoseEvaluator implements GuidedPoseEvaluator {
  CaptureQuality live = goodLive;
  CaptureQuality still = goodLive;
  int disposeCount = 0;

  @override
  Future<CaptureQuality> evaluateLive(
    GuidedLiveFrame frame, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  }) async =>
      live;

  @override
  Future<CaptureQuality> evaluateStill(
    String path, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  }) async =>
      still;

  @override
  Future<void> dispose() async {
    disposeCount++;
  }
}

class FakeFrameQualityService extends FrameQualityService {
  const FakeFrameQualityService();

  @override
  Future<FrameQualityResult> evaluateFile(String path) async => goodStill;
}

class FakeStorage implements GuidedCaptureStorage {
  final events = <String>[];
  int? failRetainAt;
  int _retainCount = 0;

  @override
  Future<String> retain(
    GuidedTemporaryFrame frame, {
    required CaptureAssetRole role,
    required int selectedRank,
  }) async {
    events.add('retain:${frame.path}:$selectedRank');
    if (failRetainAt == _retainCount++) {
      throw const FileSystemException('durable copy failed');
    }
    return '/visits/visit-1/${role.wireValue}-$selectedRank.jpg';
  }

  @override
  Future<void> deleteTemporary(String path) async {
    events.add('delete-temp:$path');
  }

  @override
  Future<void> deleteRetained(String path) async {
    events.add('delete-retained:$path');
  }
}

GuidedCameraController buildController({
  FakeCameraGateway? gateway,
  FakePoseEvaluator? pose,
  FakeStorage? storage,
}) {
  return GuidedCameraController(
    role: CaptureAssetRole.front,
    cameraGateway: gateway ?? FakeCameraGateway(),
    poseEvaluator: pose ?? FakePoseEvaluator(),
    frameQualityService: const FakeFrameQualityService(),
    frameRanker: const BurstFrameRanker(),
    storage: storage ?? FakeStorage(),
    deviceMetadataService: DeviceMetadataService(
      tiltSource: const NoopDeviceTiltSource(),
      platformDescription: () => 'test-device',
    ),
    stableFrameCount: 2,
    burstSize: 3,
    retainedFrameCount: 2,
  );
}

void main() {
  test('pauses and reinitializes camera across lifecycle changes', () async {
    final gateway = FakeCameraGateway();
    final controller = buildController(gateway: gateway);

    await controller.initialize();
    expect(controller.state, GuidedCameraState.streaming);

    await controller.handleLifecycleState(AppLifecycleState.inactive);
    expect(controller.state, GuidedCameraState.paused);
    expect(gateway.disposeCount, 1);

    await controller.handleLifecycleState(AppLifecycleState.resumed);
    expect(controller.state, GuidedCameraState.streaming);
    expect(gateway.initializeCount, 2);
  });

  test('forwards display rotation to the gateway and metadata', () async {
    final gateway = FakeCameraGateway();
    final controller = buildController(gateway: gateway);
    await controller.initialize();

    await controller.updateOrientation(DeviceOrientation.landscapeLeft);

    expect(gateway.orientations.last, DeviceOrientation.landscapeLeft);
    expect(controller.displayOrientationDegrees, 90);
  });

  test('camera initialization error exposes only controlled fallback',
      () async {
    final gateway = FakeCameraGateway()..failInitialization = true;
    final controller = buildController(gateway: gateway);

    await controller.initialize();

    expect(controller.state, GuidedCameraState.error);
    expect(controller.errorMessage, contains('camera unavailable'));
    expect(controller.fallbackAllowed, isTrue);
  });

  test('stable quality streak captures, ranks, and durably retains a burst',
      () async {
    final gateway = FakeCameraGateway();
    final storage = FakeStorage();
    final controller = buildController(gateway: gateway, storage: storage);
    await controller.initialize();

    await gateway.emit();
    expect(controller.state, GuidedCameraState.streaming);
    await gateway.emit();

    expect(controller.state, GuidedCameraState.review);
    expect(controller.retainedFrames, hasLength(2));
    expect(
      controller.retainedFrames.map((frame) => frame.selectedRank),
      [1, 2],
    );
    expect(
      controller.retainedFrames.first.qualityThresholdVersion,
      captureThresholdVersion,
    );
    expect(controller.retainedFrames.first.imageWidth, 1080);
    expect(controller.retainedFrames.first.imageHeight, 1920);
    expect(controller.retainedFrames.first.cameraIdentifier, 'rear-0');
    expect(controller.retainedFrames.first.lensDirection, 'back');
    expect(controller.retainedFrames.first.deviceMetadataJson,
        contains('test-device'));

    final lastRetain = storage.events.lastIndexWhere(
      (event) => event.startsWith('retain:'),
    );
    final firstDelete = storage.events.indexWhere(
      (event) => event.startsWith('delete-temp:'),
    );
    expect(lastRetain, lessThan(firstDelete));
    expect(
      storage.events.where((event) => event.startsWith('delete-temp:')),
      hasLength(3),
    );
  });

  test('burst failure preserves a retryable error and cleans captured temps',
      () async {
    final gateway = FakeCameraGateway()..failBurstAt = 1;
    final storage = FakeStorage();
    final controller = buildController(gateway: gateway, storage: storage);
    await controller.initialize();

    await gateway.emit();
    await gateway.emit();

    expect(controller.state, GuidedCameraState.error);
    expect(controller.errorMessage, contains('capture failed'));
    expect(storage.events, contains('delete-temp:/tmp/frame-0.jpg'));
    expect(controller.retainedFrames, isEmpty);
  });

  test('durable-copy failure preserves every temporary burst source', () async {
    final gateway = FakeCameraGateway();
    final storage = FakeStorage()..failRetainAt = 1;
    final controller = buildController(gateway: gateway, storage: storage);
    await controller.initialize();

    await gateway.emit();
    await gateway.emit();

    expect(controller.state, GuidedCameraState.error);
    expect(controller.errorMessage, contains('durable copy failed'));
    expect(
      storage.events.where((event) => event.startsWith('delete-temp:')),
      isEmpty,
    );
  });

  test('retake deletes unconfirmed retained files and resumes the stream',
      () async {
    final gateway = FakeCameraGateway();
    final storage = FakeStorage();
    final controller = buildController(gateway: gateway, storage: storage);
    await controller.initialize();
    await gateway.emit();
    await gateway.emit();

    await controller.retake();

    expect(controller.state, GuidedCameraState.streaming);
    expect(
      storage.events.where((event) => event.startsWith('delete-retained:')),
      hasLength(2),
    );
    expect(controller.retainedFrames, isEmpty);
  });

  test('cancellation removes unconfirmed retained files and disposes camera',
      () async {
    final gateway = FakeCameraGateway();
    final storage = FakeStorage();
    final controller = buildController(gateway: gateway, storage: storage);
    await controller.initialize();
    await gateway.emit();
    await gateway.emit();

    await controller.cancel();

    expect(controller.state, GuidedCameraState.cancelled);
    expect(gateway.disposeCount, 1);
    expect(
      storage.events.where((event) => event.startsWith('delete-retained:')),
      hasLength(2),
    );
  });

  test('system-camera fallback must pass pose and still quality gates',
      () async {
    final gateway = FakeCameraGateway()..failInitialization = true;
    final pose = FakePoseEvaluator()
      ..still = const CaptureQuality.blocked(CaptureIssue.wrongOrientation);
    final storage = FakeStorage();
    final controller = buildController(
      gateway: gateway,
      pose: pose,
      storage: storage,
    );
    await controller.initialize();

    final rejected =
        await controller.validateSystemCameraFile('/tmp/system.jpg');
    expect(rejected, isFalse);
    expect(controller.retainedFrames, isEmpty);
    expect(storage.events, isEmpty);

    pose.still = goodLive;
    final accepted =
        await controller.validateSystemCameraFile('/tmp/system.jpg');
    expect(accepted, isTrue);
    expect(controller.state, GuidedCameraState.review);
    expect(controller.retainedFrames.single.cameraIdentifier, 'system_camera');
  });
}
