import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

import '../../../providers/auth_provider.dart';
import '../../../screens/assessment/capture_screen.dart';
import '../../ar_scan/widgets/ar_scan_card.dart';
import '../domain/capture_models.dart';
import '../providers/guided_capture_provider.dart';
import '../services/guided_camera_controller.dart';
import '../widgets/capture_role_card.dart';

typedef GuidedCaptureLauncher = Future<List<GuidedRetainedFrame>?> Function(
  BuildContext context,
  CaptureAssetRole role,
  String visitUuid,
);

class GuidedCaptureFlowScreen extends ConsumerStatefulWidget {
  const GuidedCaptureFlowScreen({
    super.key,
    required this.visitUuid,
    this.captureLauncher,
    this.ownerUserId,
  });

  final String visitUuid;
  final GuidedCaptureLauncher? captureLauncher;
  final int? ownerUserId;

  @override
  ConsumerState<GuidedCaptureFlowScreen> createState() =>
      _GuidedCaptureFlowScreenState();
}

class _GuidedCaptureFlowScreenState
    extends ConsumerState<GuidedCaptureFlowScreen> {
  bool _launching = false;
  String? _localError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _resumeIfNeeded());
  }

  Future<void> _resumeIfNeeded() async {
    final state = ref.read(guidedCaptureProvider);
    if (state.visitUuid == widget.visitUuid) return;
    final ownerUserId = widget.ownerUserId ?? ref.read(authProvider).user?.id;
    if (ownerUserId == null) {
      setState(() => _localError = 'An authenticated operator is required');
      return;
    }
    try {
      await ref.read(guidedCaptureProvider.notifier).resume(
            visitUuid: widget.visitUuid,
            ownerUserId: ownerUserId,
          );
    } catch (_) {
      // Provider state contains the owner-scoped error.
    }
  }

  Future<void> _capture(CaptureAssetRole role) async {
    setState(() {
      _launching = true;
      _localError = null;
    });
    try {
      final launcher = widget.captureLauncher ?? _defaultCaptureLauncher;
      final frames = await launcher(context, role, widget.visitUuid);
      if (frames == null || frames.isEmpty) {
        await ref.read(guidedCaptureProvider.notifier).recordRoleFailure(role);
      } else {
        await ref.read(guidedCaptureProvider.notifier).acceptFrames(frames);
      }
    } catch (error) {
      if (mounted) setState(() => _localError = error.toString());
    } finally {
      if (mounted) setState(() => _launching = false);
    }
  }

  Future<List<GuidedRetainedFrame>?> _defaultCaptureLauncher(
    BuildContext context,
    CaptureAssetRole role,
    String visitUuid,
  ) async {
    final documents = await getApplicationDocumentsDirectory();
    if (!context.mounted) return null;
    final result = await Navigator.of(context).push<CaptureResult>(
      MaterialPageRoute(
        builder: (_) => CaptureScreen(
          role: role.wireValue,
          visitStorageDirectory: path.join(
            documents.path,
            'guided_capture',
            'visits',
            visitUuid,
          ),
        ),
      ),
    );
    return result?.retainedFrames;
  }

  void _review() {
    ref.read(guidedCaptureProvider.notifier).reviewRequiredPhotos();
    context.go('/visits/${widget.visitUuid}/capture/review');
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(guidedCaptureProvider);
    if (state.phase == GuidedCapturePhase.loading ||
        state.phase == GuidedCapturePhase.idle) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (state.phase == GuidedCapturePhase.incomplete) {
      return Scaffold(
        appBar: AppBar(title: const Text('Photo assessment')),
        body: const Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Text(
              'Incomplete capture saved. The visit can be resumed from the '
              'child profile without losing accepted photos.',
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    }
    if (state.phase == GuidedCapturePhase.error) {
      return Scaffold(
        appBar: AppBar(title: const Text('Photo assessment')),
        body: Center(child: Text(state.errorMessage ?? 'Capture failed')),
      );
    }
    if (state.phase == GuidedCapturePhase.review) {
      return Scaffold(
        appBar: AppBar(title: const Text('Photo assessment')),
        body: Center(
          child: FilledButton(
            onPressed: () => context.go(
              '/visits/${widget.visitUuid}/capture/review',
            ),
            child: const Text('Open capture review'),
          ),
        ),
      );
    }

    final role = state.currentRole;
    if (role == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final error = _localError ?? state.errorMessage;
    return Scaffold(
      appBar: AppBar(title: const Text('Guided photo assessment')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Visit ${widget.visitUuid.substring(0, 8)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            CaptureRoleCard(
              role: role,
              capturedFrameCount: state.acceptedFrames[role]?.length ?? 0,
            ),
            if (role == CaptureAssetRole.front && state.ownerUserId != null) ...[
              const SizedBox(height: 8),
              ArScanCard(
                ownerUserId: state.ownerUserId!,
                visitUuid: widget.visitUuid,
              ),
            ],
            if (error != null) ...[
              const SizedBox(height: 8),
              Text(error, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _launching ? null : () => _capture(role),
              icon: const Icon(Icons.camera_alt),
              label: Text(
                _launching
                    ? 'Saving accepted photos…'
                    : captureRoleActionLabel(role),
              ),
            ),
            if (!CaptureAssetRole.requiredRoles.contains(role)) ...[
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: _launching
                    ? null
                    : () async {
                        await ref
                            .read(guidedCaptureProvider.notifier)
                            .skipCurrentRole();
                        if (!context.mounted) return;
                        if (ref.read(guidedCaptureProvider).phase ==
                            GuidedCapturePhase.review) {
                          context.go(
                            '/visits/${widget.visitUuid}/capture/review',
                          );
                        }
                      },
                child: const Text('Skip optional view'),
              ),
            ],
            if (state.canReviewRequired) ...[
              const SizedBox(height: 8),
              TextButton(
                onPressed: _launching ? null : _review,
                child: const Text('Review required photos'),
              ),
            ],
            const SizedBox(height: 20),
            Text(
              'Accepted photos are saved immediately. You can leave and '
              'resume this visit later.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
