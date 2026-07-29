import 'package:flutter/material.dart';

import '../../guided_capture/domain/camera_screening_result.dart';

class EstimateProvenanceCard extends StatelessWidget {
  const EstimateProvenanceCard({
    super.key,
    required this.result,
  });

  final CameraScreeningResult result;

  @override
  Widget build(BuildContext context) {
    final confidence = result.classificationConfidence;
    final quality = result.captureQuality;
    final views = result.usedViews
        .map(
          (view) => view.isEmpty
              ? view
              : '${view[0].toUpperCase()}${view.substring(1).replaceAll('_', ' ')}',
        )
        .join(', ');

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'How this was estimated',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 10),
            _EvidenceRow(label: 'Method', value: result.method),
            _EvidenceRow(label: 'Model version', value: result.modelVersion),
            _EvidenceRow(
              label: 'Confidence',
              value: confidence == null
                  ? 'Not available'
                  : '${(confidence * 100).round()}%',
            ),
            _EvidenceRow(
              label: 'Capture quality',
              value: quality == null
                  ? 'Not available'
                  : '${(quality * 100).round()}%',
            ),
            _EvidenceRow(
              label: 'Used views',
              value: views.isEmpty ? 'Not available' : views,
            ),
            const SizedBox(height: 8),
            Text(
              'Research model: ${result.trainingDataLabel}. '
              'This output is non-clinical.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _EvidenceRow extends StatelessWidget {
  const _EvidenceRow({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 116,
            child: Text(
              label,
              style: Theme.of(context).textTheme.labelMedium,
            ),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }
}
