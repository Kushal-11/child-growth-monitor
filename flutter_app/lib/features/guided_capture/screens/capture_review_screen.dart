import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../providers/auth_provider.dart';
import '../providers/guided_capture_provider.dart';
import '../widgets/capture_role_card.dart';

class CaptureReviewScreen extends ConsumerStatefulWidget {
  const CaptureReviewScreen({
    super.key,
    required this.visitUuid,
    this.ownerUserId,
  });

  final String visitUuid;
  final int? ownerUserId;

  @override
  ConsumerState<CaptureReviewScreen> createState() =>
      _CaptureReviewScreenState();
}

class _CaptureReviewScreenState extends ConsumerState<CaptureReviewScreen> {
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

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(guidedCaptureProvider);
    final entries = state.acceptedFrames.entries.toList()
      ..sort((left, right) => left.key.index.compareTo(right.key.index));
    final error = _localError ?? state.errorMessage;

    return Scaffold(
      appBar: AppBar(title: const Text('Capture review')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Required photos are stored for visit '
              '${widget.visitUuid.substring(0, 8)}.',
            ),
            if (error != null) ...[
              const SizedBox(height: 8),
              Text(error, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 16),
            for (final entry in entries)
              Card(
                child: ListTile(
                  leading: SizedBox(
                    width: 48,
                    height: 48,
                    child: Image.file(
                      File(entry.value.first.localPath),
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) =>
                          const Icon(Icons.photo_outlined),
                    ),
                  ),
                  title: Text(captureRoleShortLabel(entry.key)),
                  subtitle: Text('${entry.value.length} accepted frame(s)'),
                  trailing: const Icon(Icons.check_circle, color: Colors.green),
                ),
              ),
            if (entries.isEmpty)
              const Card(
                child: ListTile(
                  leading: Icon(Icons.photo_outlined),
                  title: Text('No accepted photos found'),
                ),
              ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: state.requiredRolesComplete
                  ? () => context.push('/visits/${widget.visitUuid}/report')
                  : null,
              icon: const Icon(Icons.analytics_outlined),
              label: const Text('Generate estimated report'),
            ),
            const SizedBox(height: 8),
            Text(
              'Camera screening is connected in the next implementation task. '
              'These accepted images and their quality metadata are already '
              'saved offline.',
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
