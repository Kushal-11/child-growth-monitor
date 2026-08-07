import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/ar_scan_provider.dart';

class ArScanCard extends ConsumerStatefulWidget {
  const ArScanCard({
    super.key,
    required this.ownerUserId,
    required this.visitUuid,
  });
  final int ownerUserId;
  final String visitUuid;

  @override
  ConsumerState<ArScanCard> createState() => _ArScanCardState();
}

class _ArScanCardState extends ConsumerState<ArScanCard> {
  @override
  void initState() {
    super.initState();
    Future.microtask(() => ref.read(arScanProvider.notifier).check());
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(arScanProvider);
    if (state.loading) {
      return const Card(
        child: ListTile(
          leading: SizedBox.square(
            dimension: 22,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          title: Text('Checking efficient depth scan…'),
        ),
      );
    }
    if (state.useFallback) {
      return const Card(
        child: ListTile(
          leading: Icon(Icons.photo_camera_outlined),
          title: Text('Standard guided photos'),
          subtitle: Text(
            'Depth scanning is not supported on this phone. The lightweight '
            'capture remains available.',
          ),
        ),
      );
    }
    final result = state.result;
    return Card(
      color: Theme.of(context).colorScheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Efficient AR depth scan',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            const Text(
              'Captures up to eight sparse depth keyframes. No raw depth '
              'video or dense 3D mesh is retained.',
            ),
            if (result != null) ...[
              const SizedBox(height: 8),
              Text(
                'Experimental height '
                '${result.estimatedHeightCm.toStringAsFixed(1)} ± '
                '${result.uncertaintyCm.toStringAsFixed(1)} cm',
              ),
              const Text(
                'Research evidence only — continue with guided photos.',
              ),
            ],
            if (state.error != null) ...[
              const SizedBox(height: 8),
              Text(
                state.error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            const SizedBox(height: 10),
            FilledButton.icon(
              onPressed: state.scanning || result != null
                  ? null
                  : () => ref.read(arScanProvider.notifier).scanAndSave(
                        ownerUserId: widget.ownerUserId,
                        visitUuid: widget.visitUuid,
                      ),
              icon: const Icon(Icons.view_in_ar),
              label: Text(state.scanning ? 'Scanning…' : 'Start depth scan'),
            ),
          ],
        ),
      ),
    );
  }
}
