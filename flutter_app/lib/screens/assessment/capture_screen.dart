import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

import '../../features/guided_capture/domain/capture_models.dart';
import '../../features/guided_capture/services/burst_frame_ranker.dart';
import '../../features/guided_capture/services/device_metadata_service.dart';
import '../../features/guided_capture/services/frame_quality_service.dart';
import '../../features/guided_capture/services/guided_camera_controller.dart';
import '../../l10n/l10n_provider.dart';
import '../../services/capture_quality.dart';

typedef SystemCameraPicker = Future<String?> Function();

class CaptureResult {
  const CaptureResult.image(
    String this.imagePath, {
    this.retainedFrames = const [],
  }) : useSystemCamera = false;

  const CaptureResult.systemCamera()
      : imagePath = null,
        retainedFrames = const [],
        useSystemCamera = true;

  final String? imagePath;
  final List<GuidedRetainedFrame> retainedFrames;
  final bool useSystemCamera;
}

/// Guided live camera UI backed entirely by an injectable controller.
///
/// Camera-plugin calls, pose evaluation, burst retention, and fallback
/// validation live behind the controller/gateway boundary so widget tests do
/// not require camera platform channels.
class CaptureScreen extends ConsumerStatefulWidget {
  const CaptureScreen({
    super.key,
    required this.role,
    this.controller,
    this.systemCameraPicker,
    this.visitStorageDirectory,
  });

  final String role;
  final GuidedCameraController? controller;
  final SystemCameraPicker? systemCameraPicker;
  final String? visitStorageDirectory;

  @override
  ConsumerState<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends ConsumerState<CaptureScreen>
    with WidgetsBindingObserver {
  GuidedCameraController? _controller;
  bool _ownsController = false;
  DeviceOrientation _orientation = DeviceOrientation.portraitUp;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    final injected = widget.controller;
    if (injected != null) {
      _attach(injected, ownsController: false);
      unawaited(injected.initialize());
    } else {
      unawaited(_createDefaultController());
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final next = MediaQuery.orientationOf(context) == Orientation.landscape
        ? DeviceOrientation.landscapeLeft
        : DeviceOrientation.portraitUp;
    if (next != _orientation) {
      _orientation = next;
      final controller = _controller;
      if (controller != null) {
        unawaited(controller.updateOrientation(next));
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller != null) {
      unawaited(controller.handleLifecycleState(state));
    }
  }

  Future<void> _createDefaultController() async {
    final root = widget.visitStorageDirectory ??
        path.join(
          (await getApplicationDocumentsDirectory()).path,
          'guided_capture',
          'staging',
          DateTime.now().toUtc().microsecondsSinceEpoch.toString(),
        );
    if (!mounted) return;
    final controller = GuidedCameraController(
      role: CaptureAssetRole.fromWire(widget.role),
      cameraGateway: PluginGuidedCameraGateway(),
      poseEvaluator: MlKitGuidedPoseEvaluator(),
      frameQualityService: const FrameQualityService(),
      frameRanker: const BurstFrameRanker(),
      storage: FileGuidedCaptureStorage(root),
      deviceMetadataService: DeviceMetadataService(),
    );
    _attach(controller, ownsController: true);
    await controller.updateOrientation(_orientation);
    await controller.initialize();
  }

  void _attach(
    GuidedCameraController controller, {
    required bool ownsController,
  }) {
    _controller = controller;
    _ownsController = ownsController;
    controller.addListener(_onControllerChanged);
    if (mounted) setState(() {});
  }

  void _onControllerChanged() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    final controller = _controller;
    controller?.removeListener(_onControllerChanged);
    if (_ownsController) controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(child: _buildBody()),
    );
  }

  Widget _buildBody() {
    final controller = _controller;
    if (controller == null ||
        controller.state == GuidedCameraState.idle ||
        controller.state == GuidedCameraState.initializing ||
        controller.state == GuidedCameraState.paused) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }
    if (controller.state == GuidedCameraState.review) {
      return _buildConfirm(controller);
    }
    if (controller.state == GuidedCameraState.error) {
      return _buildError(controller);
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        Center(child: controller.buildPreview()),
        IgnorePointer(
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: _statusColor(controller), width: 4),
            ),
          ),
        ),
        _buildTopBar(controller),
        _buildBottomPanel(controller),
      ],
    );
  }

  Widget _buildTopBar(GuidedCameraController controller) {
    return Align(
      alignment: Alignment.topCenter,
      child: Container(
        color: Colors.black45,
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
        child: Row(
          children: [
            IconButton(
              icon: const Icon(Icons.close, color: Colors.white),
              onPressed: () => _cancel(controller),
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
                controller.torchEnabled ? Icons.flash_on : Icons.flash_off,
                color: Colors.white,
              ),
              onPressed: controller.toggleTorch,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomPanel(GuidedCameraController controller) {
    final capturing = controller.state == GuidedCameraState.capturing;
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
              _instruction(controller.currentQuality),
              textAlign: TextAlign.center,
              style: TextStyle(
                color: _statusColor(controller),
                fontSize: 14,
              ),
            ),
          ),
          if (controller.currentQuality.ready && !capturing)
            Padding(
              padding: const EdgeInsets.only(top: 8, left: 80, right: 80),
              child: LinearProgressIndicator(
                value: controller.gateProgress,
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
                  child: Text(
                    t('cap_auto', ref),
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ),
                FloatingActionButton(
                  heroTag: 'shutter',
                  backgroundColor: controller.currentQuality.ready
                      ? Colors.green
                      : Colors.white,
                  onPressed: capturing || !controller.currentQuality.ready
                      ? null
                      : controller.captureNow,
                  child: capturing
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

  Widget _buildConfirm(GuidedCameraController controller) {
    final retained = controller.retainedFrames;
    if (retained.isEmpty) return _buildError(controller);
    return Column(
      children: [
        Expanded(
          child: Center(
            child: Image.file(
              File(retained.first.localPath),
              errorBuilder: (_, __, ___) => const Icon(
                Icons.photo,
                color: Colors.white54,
                size: 64,
              ),
            ),
          ),
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
                onPressed: controller.retake,
              ),
              FilledButton.icon(
                style: FilledButton.styleFrom(backgroundColor: Colors.green),
                icon: const Icon(Icons.check),
                label: Text(t('cap_use_photo', ref)),
                onPressed: () {
                  final confirmed = controller.confirm();
                  context.pop(
                    CaptureResult.image(
                      confirmed.first.localPath,
                      retainedFrames: confirmed,
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildError(GuidedCameraController controller) {
    final qualityInstruction = controller.currentQuality.issue == null ||
            controller.currentQuality.issue == CaptureIssue.noPose
        ? null
        : _instruction(controller.currentQuality);
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
              qualityInstruction ?? controller.errorMessage ?? '',
              style: const TextStyle(color: Colors.white70, fontSize: 12),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: controller.fallbackAllowed
                  ? controller.initialize
                  : controller.retake,
              child: Text(t('cap_retry', ref)),
            ),
            if (controller.fallbackAllowed) ...[
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: () => _useSystemCamera(controller),
                child: Text(
                  t('cap_open_system_camera', ref),
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _useSystemCamera(GuidedCameraController controller) async {
    final picker = widget.systemCameraPicker ?? _defaultSystemCameraPicker;
    final filePath = await picker();
    if (filePath == null) return;
    await controller.validateSystemCameraFile(filePath);
  }

  Future<String?> _defaultSystemCameraPicker() async {
    final image = await ImagePicker().pickImage(
      source: ImageSource.camera,
      imageQuality: 90,
    );
    return image?.path;
  }

  Future<void> _cancel(GuidedCameraController controller) async {
    await controller.cancel();
    if (mounted) context.pop();
  }

  String _roleTitle() {
    switch (widget.role) {
      case 'side':
      case 'arm_side':
        return t('side_view', ref);
      case 'back':
        return t('back_view', ref);
      default:
        return t('front_view_photo', ref);
    }
  }

  String _instruction(CaptureQuality quality) {
    switch (quality.issue) {
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

  Color _statusColor(GuidedCameraController controller) {
    if (controller.currentQuality.ready) return Colors.green;
    if (controller.currentQuality.issue == CaptureIssue.noPose) {
      return Colors.white54;
    }
    return Colors.orange;
  }
}
