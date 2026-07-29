import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;

import '../../../services/camera_frame_converter.dart';
import '../../../services/capture_quality.dart';
import '../domain/capture_models.dart';
import '../domain/capture_thresholds.dart';
import 'burst_frame_ranker.dart';
import 'device_metadata_service.dart';
import 'frame_quality_service.dart';

enum GuidedCameraState {
  idle,
  initializing,
  streaming,
  capturing,
  review,
  paused,
  error,
  cancelled,
}

class GuidedCameraException implements Exception {
  const GuidedCameraException(this.code, this.message);

  final String code;
  final String message;

  @override
  String toString() => '$code: $message';
}

class GuidedCameraDescription {
  const GuidedCameraDescription({
    required this.identifier,
    required this.lensDirection,
    required this.sensorOrientation,
  });

  final String identifier;
  final String lensDirection;
  final int sensorOrientation;
}

class GuidedLiveFrame {
  const GuidedLiveFrame({
    required this.payload,
    required this.width,
    required this.height,
  });

  final Object payload;
  final double width;
  final double height;
}

class GuidedTemporaryFrame {
  const GuidedTemporaryFrame({
    required this.path,
    required this.capturedAt,
    required this.width,
    required this.height,
    required this.exifOrientation,
  });

  final String path;
  final DateTime capturedAt;
  final int width;
  final int height;
  final int? exifOrientation;
}

typedef GuidedLiveFrameCallback = Future<void> Function(GuidedLiveFrame frame);

abstract interface class GuidedCameraGateway {
  GuidedCameraDescription get description;
  Widget buildPreview();
  Future<void> initialize(DeviceOrientation orientation);
  Future<void> startLiveStream(GuidedLiveFrameCallback onFrame);
  Future<void> stopLiveStream();
  Future<GuidedTemporaryFrame> takePicture();
  Future<void> setCaptureOrientation(DeviceOrientation orientation);
  Future<void> setTorch(bool enabled);
  Future<void> dispose();
}

abstract interface class GuidedPoseEvaluator {
  Future<CaptureQuality> evaluateLive(
    GuidedLiveFrame frame, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  });

  Future<CaptureQuality> evaluateStill(
    String path, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  });

  Future<void> dispose();
}

abstract interface class GuidedCaptureStorage {
  Future<String> retain(
    GuidedTemporaryFrame frame, {
    required CaptureAssetRole role,
    required int selectedRank,
  });

  Future<void> deleteTemporary(String path);
  Future<void> deleteRetained(String path);
}

class GuidedRetainedFrame {
  const GuidedRetainedFrame({
    required this.localPath,
    required this.role,
    required this.capturedAt,
    required this.selectedRank,
    required this.poseScore,
    required this.coverageScore,
    required this.orientationScore,
    required this.sharpnessScore,
    required this.lightingScore,
    required this.overallScore,
    required this.qualityThresholdVersion,
    required this.imageWidth,
    required this.imageHeight,
    required this.exifOrientation,
    required this.displayOrientation,
    required this.cameraIdentifier,
    required this.lensDirection,
    required this.deviceMetadataJson,
  });

  final String localPath;
  final CaptureAssetRole role;
  final DateTime capturedAt;
  final int selectedRank;
  final double poseScore;
  final double coverageScore;
  final double orientationScore;
  final double sharpnessScore;
  final double lightingScore;
  final double overallScore;
  final String qualityThresholdVersion;
  final int imageWidth;
  final int imageHeight;
  final int? exifOrientation;
  final int displayOrientation;
  final String cameraIdentifier;
  final String lensDirection;
  final String deviceMetadataJson;
}

class PluginGuidedCameraGateway implements GuidedCameraGateway {
  CameraController? _controller;
  CameraDescription? _camera;
  bool _handlingFrame = false;

  @override
  GuidedCameraDescription get description {
    final camera = _camera;
    if (camera == null) {
      throw const GuidedCameraException(
        'not_initialized',
        'Camera description is unavailable',
      );
    }
    return GuidedCameraDescription(
      identifier: camera.name,
      lensDirection: camera.lensDirection.name,
      sensorOrientation: camera.sensorOrientation,
    );
  }

  @override
  Widget buildPreview() {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const SizedBox.shrink();
    }
    return CameraPreview(controller);
  }

  @override
  Future<void> initialize(DeviceOrientation orientation) async {
    await dispose();
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw const GuidedCameraException(
          'no_camera',
          'No camera is available on this device',
        );
      }
      final camera = cameras.firstWhere(
        (candidate) => candidate.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      await controller.initialize();
      await controller.lockCaptureOrientation(orientation);
      _camera = camera;
      _controller = controller;
    } on GuidedCameraException {
      rethrow;
    } on CameraException catch (error) {
      throw GuidedCameraException(
        error.code,
        error.description ?? 'Camera initialization failed',
      );
    }
  }

  @override
  Future<void> startLiveStream(GuidedLiveFrameCallback onFrame) async {
    final controller = _requireController();
    if (controller.value.isStreamingImages) return;
    await controller.startImageStream((image) {
      if (_handlingFrame) return;
      final camera = _camera;
      if (camera == null) return;
      final converted = convertCameraFrame(
        image,
        camera,
        controller.value.deviceOrientation,
      );
      if (converted == null) return;
      _handlingFrame = true;
      unawaited(
        onFrame(
          GuidedLiveFrame(
            payload: converted.inputImage,
            width: converted.uprightSize.width,
            height: converted.uprightSize.height,
          ),
        ).whenComplete(() => _handlingFrame = false),
      );
    });
  }

  @override
  Future<void> stopLiveStream() async {
    final controller = _controller;
    if (controller != null && controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
  }

  @override
  Future<GuidedTemporaryFrame> takePicture() async {
    final controller = _requireController();
    try {
      final file = await controller.takePicture();
      final decoded = await img.decodeImageFile(file.path);
      if (decoded == null) {
        throw const GuidedCameraException(
          'decode_failed',
          'Captured image could not be decoded',
        );
      }
      return GuidedTemporaryFrame(
        path: file.path,
        capturedAt: DateTime.now().toUtc(),
        width: decoded.width,
        height: decoded.height,
        exifOrientation: null,
      );
    } on GuidedCameraException {
      rethrow;
    } on CameraException catch (error) {
      throw GuidedCameraException(
        error.code,
        error.description ?? 'Still capture failed',
      );
    }
  }

  @override
  Future<void> setCaptureOrientation(DeviceOrientation orientation) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    await controller.lockCaptureOrientation(orientation);
  }

  @override
  Future<void> setTorch(bool enabled) async {
    final controller = _requireController();
    await controller.setFlashMode(enabled ? FlashMode.torch : FlashMode.off);
  }

  @override
  Future<void> dispose() async {
    final controller = _controller;
    _controller = null;
    _camera = null;
    _handlingFrame = false;
    await controller?.dispose();
  }

  CameraController _requireController() {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      throw const GuidedCameraException(
        'not_initialized',
        'Camera is not initialized',
      );
    }
    return controller;
  }
}

class MlKitGuidedPoseEvaluator implements GuidedPoseEvaluator {
  MlKitGuidedPoseEvaluator()
      : _liveDetector = PoseDetector(
          options: PoseDetectorOptions(
            mode: PoseDetectionMode.stream,
            model: PoseDetectionModel.base,
          ),
        ),
        _stillDetector = PoseDetector(
          options: PoseDetectorOptions(
            mode: PoseDetectionMode.single,
            model: PoseDetectionModel.accurate,
          ),
        );

  final PoseDetector _liveDetector;
  final PoseDetector _stillDetector;

  @override
  Future<CaptureQuality> evaluateLive(
    GuidedLiveFrame frame, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  }) async {
    final poses = await _liveDetector.processImage(frame.payload as InputImage);
    return evaluateCaptureQuality(
      poses.isEmpty
          ? const []
          : poses.first.landmarks.values.toList(growable: false),
      poseCount: poses.length,
      role: role,
      imageWidth: frame.width,
      imageHeight: frame.height,
      tiltDegrees: tiltDegrees,
    );
  }

  @override
  Future<CaptureQuality> evaluateStill(
    String path, {
    required CaptureAssetRole role,
    required double? tiltDegrees,
  }) async {
    final image = await img.decodeImageFile(path);
    if (image == null) {
      return const CaptureQuality.blocked(CaptureIssue.noPose);
    }
    final poses = await _stillDetector.processImage(
      InputImage.fromFilePath(path),
    );
    return evaluateCaptureQuality(
      poses.isEmpty
          ? const []
          : poses.first.landmarks.values.toList(growable: false),
      poseCount: poses.length,
      role: role,
      imageWidth: image.width.toDouble(),
      imageHeight: image.height.toDouble(),
      tiltDegrees: tiltDegrees,
    );
  }

  @override
  Future<void> dispose() async {
    await _liveDetector.close();
    await _stillDetector.close();
  }
}

class FileGuidedCaptureStorage implements GuidedCaptureStorage {
  FileGuidedCaptureStorage(this.visitDirectory);

  final String visitDirectory;

  @override
  Future<String> retain(
    GuidedTemporaryFrame frame, {
    required CaptureAssetRole role,
    required int selectedRank,
  }) async {
    final roleDirectory = Directory(path.join(visitDirectory, role.wireValue));
    await roleDirectory.create(recursive: true);
    final extension = path.extension(frame.path).isEmpty
        ? '.jpg'
        : path.extension(frame.path);
    final fileName =
        '${frame.capturedAt.microsecondsSinceEpoch}_$selectedRank$extension';
    final destination = path.join(roleDirectory.path, fileName);
    final retained = await File(frame.path).copy(destination);
    final handle = await retained.open(mode: FileMode.append);
    try {
      await handle.flush();
    } finally {
      await handle.close();
    }
    return retained.path;
  }

  @override
  Future<void> deleteTemporary(String filePath) async {
    final file = File(filePath);
    if (await file.exists()) await file.delete();
  }

  @override
  Future<void> deleteRetained(String filePath) async {
    final file = File(filePath);
    if (await file.exists()) await file.delete();
  }
}

class GuidedCameraController extends ChangeNotifier {
  GuidedCameraController({
    required this.role,
    required this.cameraGateway,
    required this.poseEvaluator,
    required this.frameQualityService,
    required this.frameRanker,
    required this.storage,
    required this.deviceMetadataService,
    this.stableFrameCount = captureStableFrameCount,
    this.burstSize = captureBurstFrameCount,
    this.retainedFrameCount = captureRetainedFrameCount,
  }) : _gate = AutoCaptureGate(requiredGoodFrames: stableFrameCount);

  final CaptureAssetRole role;
  final GuidedCameraGateway cameraGateway;
  final GuidedPoseEvaluator poseEvaluator;
  final FrameQualityService frameQualityService;
  final BurstFrameRanker frameRanker;
  final GuidedCaptureStorage storage;
  final DeviceMetadataService deviceMetadataService;
  final int stableFrameCount;
  final int burstSize;
  final int retainedFrameCount;
  final AutoCaptureGate _gate;

  GuidedCameraState _state = GuidedCameraState.idle;
  CaptureQuality _currentQuality =
      const CaptureQuality.blocked(CaptureIssue.noPose);
  final List<GuidedRetainedFrame> _retainedFrames = [];
  DeviceOrientation _orientation = DeviceOrientation.portraitUp;
  String? _errorMessage;
  bool _fallbackAllowed = false;
  bool _evaluatingFrame = false;
  bool _captureInFlight = false;
  bool _torchEnabled = false;
  bool _confirmed = false;
  bool _isDisposed = false;

  GuidedCameraState get state => _state;
  CaptureQuality get currentQuality => _currentQuality;
  double get gateProgress => _gate.progress;
  String? get errorMessage => _errorMessage;
  bool get fallbackAllowed => _fallbackAllowed;
  bool get torchEnabled => _torchEnabled;
  List<GuidedRetainedFrame> get retainedFrames =>
      List.unmodifiable(_retainedFrames);
  int get displayOrientationDegrees => _orientationDegrees(_orientation);
  Widget buildPreview() => cameraGateway.buildPreview();

  Future<void> initialize() async {
    if (_state == GuidedCameraState.initializing || _captureInFlight) return;
    _setState(GuidedCameraState.initializing);
    _errorMessage = null;
    _fallbackAllowed = false;
    deviceMetadataService.start();
    try {
      await cameraGateway.initialize(_orientation);
      await cameraGateway.startLiveStream(_onLiveFrame);
      _setState(GuidedCameraState.streaming);
    } catch (error) {
      _errorMessage = _messageFor(error);
      _fallbackAllowed = true;
      _setState(GuidedCameraState.error);
    }
  }

  Future<void> handleLifecycleState(AppLifecycleState lifecycleState) async {
    if (lifecycleState == AppLifecycleState.resumed) {
      if (_state == GuidedCameraState.paused) await initialize();
      return;
    }
    if (lifecycleState == AppLifecycleState.inactive ||
        lifecycleState == AppLifecycleState.paused ||
        lifecycleState == AppLifecycleState.detached ||
        lifecycleState == AppLifecycleState.hidden) {
      await cameraGateway.stopLiveStream();
      await cameraGateway.dispose();
      await deviceMetadataService.stop();
      _setState(GuidedCameraState.paused);
    }
  }

  Future<void> updateOrientation(DeviceOrientation orientation) async {
    if (_orientation == orientation) return;
    _orientation = orientation;
    await cameraGateway.setCaptureOrientation(orientation);
    _notify();
  }

  Future<void> captureNow() async {
    if (!_currentQuality.ready) return;
    await _captureBurst(_currentQuality);
  }

  Future<void> toggleTorch() async {
    final next = !_torchEnabled;
    try {
      await cameraGateway.setTorch(next);
      _torchEnabled = next;
      _notify();
    } catch (_) {
      // Torch support is optional.
    }
  }

  Future<bool> validateSystemCameraFile(String filePath) async {
    final tilt = deviceMetadataService.tiltDegrees;
    final poseQuality = await poseEvaluator.evaluateStill(
      filePath,
      role: role,
      tiltDegrees: tilt,
    );
    _currentQuality = poseQuality;
    if (!poseQuality.ready) {
      _errorMessage = 'System-camera photo failed pose quality';
      _fallbackAllowed = true;
      _setState(GuidedCameraState.error);
      return false;
    }

    final stillQuality = await frameQualityService.evaluateFile(filePath);
    if (!stillQuality.accepted) {
      _errorMessage =
          'System-camera photo failed still quality: ${stillQuality.rejectionReason?.name}';
      _fallbackAllowed = true;
      _setState(GuidedCameraState.error);
      return false;
    }

    img.Image? decoded;
    try {
      decoded = await img.decodeImageFile(filePath);
    } on FileSystemException {
      // The real frame-quality service rejects unreadable files. Injectable
      // test evaluators may intentionally provide metadata-only paths.
    }
    final temporary = GuidedTemporaryFrame(
      path: filePath,
      capturedAt: DateTime.now().toUtc(),
      width: decoded?.width ?? 0,
      height: decoded?.height ?? 0,
      exifOrientation: null,
    );
    final storedPath = await storage.retain(
      temporary,
      role: role,
      selectedRank: 1,
    );
    _retainedFrames
      ..clear()
      ..add(
        _retainedFrame(
          storedPath: storedPath,
          temporary: temporary,
          rank: 1,
          liveQuality: poseQuality,
          stillQuality: stillQuality,
          description: const GuidedCameraDescription(
            identifier: 'system_camera',
            lensDirection: 'unknown',
            sensorOrientation: 0,
          ),
        ),
      );
    await _deleteTempsBestEffort([temporary]);
    _fallbackAllowed = false;
    _errorMessage = null;
    _setState(GuidedCameraState.review);
    return true;
  }

  Future<void> retake() async {
    await _deleteRetained();
    _currentQuality = const CaptureQuality.blocked(CaptureIssue.noPose);
    _errorMessage = null;
    _fallbackAllowed = false;
    _gate.reset();
    try {
      await cameraGateway.startLiveStream(_onLiveFrame);
      _setState(GuidedCameraState.streaming);
    } catch (error) {
      _errorMessage = _messageFor(error);
      _fallbackAllowed = true;
      _setState(GuidedCameraState.error);
    }
  }

  List<GuidedRetainedFrame> confirm() {
    _confirmed = true;
    return List.unmodifiable(_retainedFrames);
  }

  Future<void> cancel() async {
    if (!_confirmed) await _deleteRetained();
    await _shutdownResources();
    _setState(GuidedCameraState.cancelled);
  }

  Future<void> shutdown() => _shutdownResources();

  Future<void> _onLiveFrame(GuidedLiveFrame frame) async {
    if (_state != GuidedCameraState.streaming ||
        _evaluatingFrame ||
        _captureInFlight) {
      return;
    }
    _evaluatingFrame = true;
    try {
      final quality = await poseEvaluator.evaluateLive(
        frame,
        role: role,
        tiltDegrees: deviceMetadataService.tiltDegrees,
      );
      _currentQuality = quality;
      final shouldCapture = _gate.onFrame(quality.ready);
      _notify();
      if (shouldCapture) await _captureBurst(quality);
    } catch (error) {
      _errorMessage = _messageFor(error);
      _setState(GuidedCameraState.error);
    } finally {
      _evaluatingFrame = false;
    }
  }

  Future<void> _captureBurst(CaptureQuality liveQuality) async {
    if (_captureInFlight) return;
    _captureInFlight = true;
    _setState(GuidedCameraState.capturing);
    final temporaryFrames = <GuidedTemporaryFrame>[];
    final partiallyRetainedPaths = <String>[];
    var durableRetentionStarted = false;
    var durableRetentionCompleted = false;
    try {
      await cameraGateway.stopLiveStream();
      for (var index = 0; index < burstSize; index++) {
        temporaryFrames.add(await cameraGateway.takePicture());
      }

      final candidates = <BurstFrameCandidate<GuidedTemporaryFrame>>[];
      for (var index = 0; index < temporaryFrames.length; index++) {
        candidates.add(
          BurstFrameCandidate(
            value: temporaryFrames[index],
            captureIndex: index,
            liveQuality: liveQuality,
            stillQuality: await frameQualityService.evaluateFile(
              temporaryFrames[index].path,
            ),
          ),
        );
      }
      final ranked =
          frameRanker.rank(candidates).take(retainedFrameCount).toList();
      if (ranked.isEmpty) {
        throw const GuidedCameraException(
          'quality_rejected',
          'No burst frame passed the still-image quality gate',
        );
      }

      final description = cameraGateway.description;
      final retained = <GuidedRetainedFrame>[];
      durableRetentionStarted = true;
      for (final frame in ranked) {
        final storedPath = await storage.retain(
          frame.value,
          role: role,
          selectedRank: frame.rank,
        );
        partiallyRetainedPaths.add(storedPath);
        retained.add(
          _retainedFrame(
            storedPath: storedPath,
            temporary: frame.value,
            rank: frame.rank,
            liveQuality: frame.liveQuality,
            stillQuality: frame.stillQuality,
            description: description,
          ),
        );
      }
      durableRetentionCompleted = true;

      _retainedFrames
        ..clear()
        ..addAll(retained);
      await _deleteTempsBestEffort(temporaryFrames);
      _errorMessage = null;
      _fallbackAllowed = false;
      _setState(GuidedCameraState.review);
    } catch (error) {
      if (!durableRetentionStarted) {
        await _deleteTempsBestEffort(temporaryFrames);
      } else if (!durableRetentionCompleted) {
        for (final retainedPath in partiallyRetainedPaths) {
          try {
            await storage.deleteRetained(retainedPath);
          } catch (_) {
            // The temporary sources remain available for recovery.
          }
        }
      }
      _errorMessage = _messageFor(error);
      _setState(GuidedCameraState.error);
    } finally {
      _captureInFlight = false;
    }
  }

  GuidedRetainedFrame _retainedFrame({
    required String storedPath,
    required GuidedTemporaryFrame temporary,
    required int rank,
    required CaptureQuality liveQuality,
    required FrameQualityResult stillQuality,
    required GuidedCameraDescription description,
  }) {
    final overallScore = captureLiveScoreWeight * liveQuality.overallScore +
        captureStillScoreWeight * stillQuality.overallScore;
    return GuidedRetainedFrame(
      localPath: storedPath,
      role: role,
      capturedAt: temporary.capturedAt,
      selectedRank: rank,
      poseScore: liveQuality.poseScore,
      coverageScore: liveQuality.coverageScore,
      orientationScore: liveQuality.orientationScore,
      sharpnessScore: stillQuality.sharpnessScore,
      lightingScore: stillQuality.lightingScore,
      overallScore: overallScore.clamp(0.0, 1.0),
      qualityThresholdVersion: captureThresholdVersion,
      imageWidth: temporary.width,
      imageHeight: temporary.height,
      exifOrientation: temporary.exifOrientation,
      displayOrientation: displayOrientationDegrees,
      cameraIdentifier: description.identifier,
      lensDirection: description.lensDirection,
      deviceMetadataJson: deviceMetadataService.snapshotJson(
        displayOrientationDegrees: displayOrientationDegrees,
        cameraIdentifier: description.identifier,
        lensDirection: description.lensDirection,
        sensorOrientationDegrees: description.sensorOrientation,
      ),
    );
  }

  Future<void> _deleteTempsBestEffort(
    Iterable<GuidedTemporaryFrame> frames,
  ) async {
    for (final frame in frames) {
      try {
        await storage.deleteTemporary(frame.path);
      } catch (_) {
        // A durable retained copy is never discarded because temp cleanup
        // failed. Later media maintenance can retry this bounded cleanup.
      }
    }
  }

  Future<void> _deleteRetained() async {
    for (final frame in _retainedFrames) {
      await storage.deleteRetained(frame.localPath);
    }
    _retainedFrames.clear();
  }

  Future<void> _shutdownResources() async {
    try {
      await cameraGateway.stopLiveStream();
    } catch (_) {
      // The camera may already be disposed after an init/lifecycle failure.
    }
    await cameraGateway.dispose();
    await poseEvaluator.dispose();
    await deviceMetadataService.dispose();
  }

  void _setState(GuidedCameraState value) {
    _state = value;
    _notify();
  }

  void _notify() {
    if (!_isDisposed) notifyListeners();
  }

  String _messageFor(Object error) =>
      error is GuidedCameraException ? error.message : error.toString();

  @override
  void dispose() {
    _isDisposed = true;
    unawaited(_shutdownResources());
    super.dispose();
  }
}

int _orientationDegrees(DeviceOrientation orientation) => switch (orientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
    };
