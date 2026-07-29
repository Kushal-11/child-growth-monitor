import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import '../../features/guided_capture/domain/capture_models.dart';
import '../../l10n/l10n_provider.dart';
import '../../services/camera_frame_converter.dart';
import '../../services/capture_quality.dart';

/// What the capture screen pops with.
class CaptureResult {
  const CaptureResult.image(String this.imagePath) : useSystemCamera = false;
  const CaptureResult.systemCamera()
      : imagePath = null,
        useSystemCamera = true;

  /// Path of the captured photo, when one was taken and confirmed.
  final String? imagePath;

  /// True when the camera could not be used and the caller should fall back
  /// to the system camera via image_picker.
  final bool useSystemCamera;
}

/// In-app live capture: full-screen camera preview with real-time pose
/// guidance. Streams frames through a fast base-model pose detector purely to
/// gate capture quality; the confirmed still goes through the existing
/// accurate static pipeline (PoseService), so measurements are unaffected.
class CaptureScreen extends ConsumerStatefulWidget {
  const CaptureScreen({super.key, required this.role});

  /// Which assessment photo this capture is for: 'front' | 'side' | 'back'.
  final String role;

  @override
  ConsumerState<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends ConsumerState<CaptureScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  late final PoseDetector _liveDetector;
  final AutoCaptureGate _gate = AutoCaptureGate();

  CaptureQuality _quality = const CaptureQuality.blocked(CaptureIssue.noPose);
  double _gateProgress = 0;
  bool _detecting = false;
  bool _capturing = false;
  bool _autoCapture = true;
  bool _torchOn = false;
  XFile? _captured;
  String? _initError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Base model in stream mode: fast enough for live gating. The accurate
    // model stays where it was — in PoseService, run on the confirmed still.
    _liveDetector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseDetectionModel.base,
      ),
    );
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _liveDetector.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      controller.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw CameraException('noCamera', 'No camera available');
      }
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        // 1080p balances still quality for the measurement pipeline against
        // stream-analysis cost on low-end field devices.
        ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      await controller.initialize();
      await controller.lockCaptureOrientation(DeviceOrientation.portraitUp);
      await controller.startImageStream(_onFrame);
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _controller = controller;
        _initError = null;
      });
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() => _initError = e.description ?? e.code);
    } catch (e) {
      if (!mounted) return;
      setState(() => _initError = e.toString());
    }
  }

  Future<void> _onFrame(CameraImage image) async {
    final controller = _controller;
    if (controller == null || _detecting || _capturing || _captured != null) {
      return;
    }
    _detecting = true;
    try {
      final frame = convertCameraFrame(
        image,
        controller.description,
        controller.value.deviceOrientation,
      );
      if (frame == null) return;
      final poses = await _liveDetector.processImage(frame.inputImage);
      final landmarks = poses.isEmpty
          ? const <PoseLandmark>[]
          : poses.first.landmarks.values.toList();
      final quality = evaluateCaptureQuality(
        landmarks,
        poseCount: poses.length,
        role: CaptureAssetRole.fromWire(widget.role),
        imageWidth: frame.uprightSize.width,
        imageHeight: frame.uprightSize.height,
        tiltDegrees: null,
      );
      final shouldFire = _autoCapture && _gate.onFrame(quality.ready);
      if (!mounted) return;
      setState(() {
        _quality = quality;
        _gateProgress = _gate.progress;
      });
      if (shouldFire) await _takePicture();
    } finally {
      _detecting = false;
    }
  }

  Future<void> _takePicture() async {
    final controller = _controller;
    if (controller == null || _capturing || _captured != null) return;
    setState(() => _capturing = true);
    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
      final file = await controller.takePicture();
      if (!mounted) return;
      setState(() => _captured = file);
    } on CameraException {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(t('cap_capture_failed', ref))),
      );
      await _resumeStream();
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<void> _retake() async {
    _gate.reset();
    setState(() {
      _captured = null;
      _quality = const CaptureQuality.blocked(CaptureIssue.noPose);
      _gateProgress = 0;
    });
    await _resumeStream();
  }

  Future<void> _resumeStream() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (!controller.value.isStreamingImages) {
      try {
        await controller.startImageStream(_onFrame);
      } on CameraException {
        // Preview keeps working; the worker can still capture manually.
      }
    }
  }

  Future<void> _toggleTorch() async {
    final controller = _controller;
    if (controller == null) return;
    try {
      await controller.setFlashMode(_torchOn ? FlashMode.off : FlashMode.torch);
      setState(() => _torchOn = !_torchOn);
    } on CameraException {
      // Device has no torch; ignore.
    }
  }

  String _roleTitle() {
    switch (widget.role) {
      case 'side':
        return t('side_view', ref);
      case 'back':
        return t('back_view', ref);
      default:
        return t('front_view_photo', ref);
    }
  }

  String _instruction() {
    switch (_quality.issue) {
      case CaptureIssue.noPose:
        return t('cap_no_pose', ref);
      case CaptureIssue.multiplePoses:
        return t('cap_multiple_poses', ref);
      case CaptureIssue.wrongOrientation:
        return t('cap_wrong_orientation', ref);
      case CaptureIssue.cutOffTop:
        return t('cap_cut_top', ref);
      case CaptureIssue.cutOffBottom:
        return t('cap_cut_bottom', ref);
      case CaptureIssue.missingRequiredLandmark:
        return t('cap_missing_landmark', ref);
      case CaptureIssue.tooFar:
        return t('cap_too_far', ref);
      case CaptureIssue.offCenter:
        return t('cap_center', ref);
      case CaptureIssue.lowVisibility:
        return t('cap_low_visibility', ref);
      case CaptureIssue.excessiveTilt:
        return t('cap_tilt', ref);
      case null:
        return t('cap_ready', ref);
    }
  }

  Color _statusColor() {
    if (_quality.ready) return Colors.green;
    if (_quality.issue == CaptureIssue.noPose) return Colors.white54;
    return Colors.orange;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(child: _buildBody()),
    );
  }

  Widget _buildBody() {
    if (_initError != null) return _buildError();
    if (_captured != null) return _buildConfirm();
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        Center(child: CameraPreview(controller)),
        IgnorePointer(
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: _statusColor(), width: 4),
            ),
          ),
        ),
        _buildTopBar(),
        _buildBottomPanel(),
      ],
    );
  }

  Widget _buildTopBar() {
    return Align(
      alignment: Alignment.topCenter,
      child: Container(
        color: Colors.black45,
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
        child: Row(
          children: [
            IconButton(
              icon: const Icon(Icons.close, color: Colors.white),
              onPressed: () => context.pop(),
            ),
            Expanded(
              child: Text(
                _roleTitle(),
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
            IconButton(
              icon: Icon(
                _torchOn ? Icons.flash_on : Icons.flash_off,
                color: Colors.white,
              ),
              onPressed: _toggleTorch,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomPanel() {
    return Align(
      alignment: Alignment.bottomCenter,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 24),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              _instruction(),
              textAlign: TextAlign.center,
              style: TextStyle(color: _statusColor(), fontSize: 14),
            ),
          ),
          if (_autoCapture && _quality.ready)
            Padding(
              padding: const EdgeInsets.only(top: 8, left: 80, right: 80),
              child: LinearProgressIndicator(
                value: _gateProgress,
                color: Colors.green,
                backgroundColor: Colors.white24,
                minHeight: 4,
              ),
            ),
          Container(
            color: Colors.black45,
            margin: const EdgeInsets.only(top: 12),
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Row(
              children: [
                Expanded(
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Switch(
                        value: _autoCapture,
                        activeThumbColor: Colors.green,
                        onChanged: (v) {
                          _gate.reset();
                          setState(() {
                            _autoCapture = v;
                            _gateProgress = 0;
                          });
                        },
                      ),
                      Text(
                        t('cap_auto', ref),
                        style:
                            const TextStyle(color: Colors.white, fontSize: 12),
                      ),
                    ],
                  ),
                ),
                FloatingActionButton(
                  heroTag: 'shutter',
                  backgroundColor: _quality.ready ? Colors.green : Colors.white,
                  onPressed: _capturing ? null : _takePicture,
                  child: _capturing
                      ? const SizedBox(
                          width: 24,
                          height: 24,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.camera_alt, color: Colors.black),
                ),
                const Expanded(child: SizedBox()),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfirm() {
    return Column(
      children: [
        Expanded(
          child: Center(child: Image.file(File(_captured!.path))),
        ),
        Container(
          color: Colors.black45,
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              OutlinedButton.icon(
                icon: const Icon(Icons.refresh, color: Colors.white),
                label: Text(
                  t('cap_retake', ref),
                  style: const TextStyle(color: Colors.white),
                ),
                onPressed: _retake,
              ),
              FilledButton.icon(
                style: FilledButton.styleFrom(backgroundColor: Colors.green),
                icon: const Icon(Icons.check),
                label: Text(t('cap_use_photo', ref)),
                onPressed: () =>
                    context.pop(CaptureResult.image(_captured!.path)),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildError() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.no_photography, color: Colors.white54, size: 48),
            const SizedBox(height: 12),
            Text(
              t('cap_camera_error', ref),
              style: const TextStyle(color: Colors.white, fontSize: 16),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 4),
            Text(
              _initError!,
              style: const TextStyle(color: Colors.white54, fontSize: 12),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                setState(() => _initError = null);
                _initCamera();
              },
              child: Text(t('cap_retry', ref)),
            ),
            const SizedBox(height: 8),
            OutlinedButton(
              onPressed: () => context.pop(const CaptureResult.systemCamera()),
              child: Text(
                t('cap_open_system_camera', ref),
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
