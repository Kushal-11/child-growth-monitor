import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/ar_scan_provider.dart';

class ArScanCard extends ConsumerWidget {
  const ArScanCard({
    super.key,
    required this.ownerUserId,
    required this.visitUuid,
  });

  final int ownerUserId;
  final String visitUuid;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(arScanProvider);
    if (state.loading) {
      return const Card(
        child: ListTile(
          leading: SizedBox.square(
            dimension: 22,
            child: Icon(Icons.manage_search_outlined),
          ),
          title: Text('Checking guided AR depth…'),
          subtitle: Text('Standard guided photos remain available.'),
        ),
      );
    }
    if (state.useFallback) {
      return const Card(
        child: ListTile(
          leading: Icon(Icons.photo_camera_outlined),
          title: Text('Standard guided photos'),
          subtitle: Text(
            'Full depth scanning is not supported on this phone. Continue '
            'with the standard front and side photos.',
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
              'Guided AR depth scan',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            const Text(
              'Optional contactless scan using raw depth confidence, floor '
              'stability, and front-to-side body geometry. No RGB, raw depth, '
              'point cloud, or mesh is retained.',
            ),
            const SizedBox(height: 8),
            const Text(
              'Use only when the child can stand safely without support. '
              'Otherwise skip this and continue with guided photos.',
            ),
            const SizedBox(height: 8),
            Text(
              'Google Play Services for AR (ARCore) is provided by Google LLC '
              'and governed by the Google Privacy Policy.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            if (result != null) ...[
              const SizedBox(height: 10),
              Text(
                'Estimated height '
                '${result.estimatedHeightCm.toStringAsFixed(1)} ± '
                '${result.uncertaintyCm.toStringAsFixed(1)} cm',
                style: Theme.of(context).textTheme.titleSmall,
              ),
              Text(
                '${result.acceptedKeyframes} frames • '
                '${result.scanCoverageDegrees.toStringAsFixed(0)}° coverage • '
                '${(result.qualityScore * 100).toStringAsFixed(0)}% quality',
              ),
              if (result.estimatedMuacCm != null)
                Text(
                  'Estimated MUAC '
                  '${result.estimatedMuacCm!.toStringAsFixed(1)} ± '
                  '${result.muacUncertaintyCm!.toStringAsFixed(1)} cm',
                ),
              Text(
                result.hasWeightGeometry
                    ? 'Body geometry captured for the contactless weight estimate.'
                    : 'Height saved. Weight/MUAC will use the best available guided-camera fallback.',
              ),
              const Text(
                'These values are estimates. Full height, weight, and MUAC '
                'results appear in the estimated report.',
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
                        ownerUserId: ownerUserId,
                        visitUuid: visitUuid,
                      ),
              icon: const Icon(Icons.view_in_ar),
              label: Text(
                state.scanning ? 'Scanning…' : 'Start guided depth scan',
              ),
            ),
          ],
        ),
      ),
    );
  }
}
