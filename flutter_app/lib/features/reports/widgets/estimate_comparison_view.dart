import 'package:flutter/material.dart';

import '../../guided_capture/domain/camera_screening_result.dart';
import '../providers/visit_report_provider.dart';

class EstimateComparisonView extends StatelessWidget {
  const EstimateComparisonView({
    super.key,
    required this.estimate,
    required this.measured,
    required this.authorized,
  });

  final CameraScreeningResult estimate;
  final MeasuredReportSnapshot measured;
  final bool authorized;

  @override
  Widget build(BuildContext context) {
    if (!authorized) return const SizedBox.shrink();

    final comparisons = <Widget>[
      if (estimate.reportableHeightCm != null && measured.heightCm != null)
        _NumericComparison(
          label: 'height',
          estimated: estimate.reportableHeightCm!,
          measured: measured.heightCm!,
          unit: 'cm',
        ),
      if (estimate.reportableWeightKg != null && measured.weightKg != null)
        _NumericComparison(
          label: 'weight',
          estimated: estimate.reportableWeightKg!,
          measured: measured.weightKg!,
          unit: 'kg',
        ),
      if (estimate.reportableStuntingStatus != null &&
          measured.hazStatus != null)
        Text(
          'Stunting classification agreement: '
          '${_agreement(
            estimate.reportableStuntingStatus!,
            measured.hazStatus!,
          )}',
        ),
      if (estimate.reportableWastingStatus != null &&
          measured.whzStatus != null)
        Text(
          'Wasting classification agreement: '
          '${_agreement(
            estimate.reportableWastingStatus!,
            measured.whzStatus!,
          )}',
        ),
    ];

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Compare with estimate',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 4),
              Text(
                'Camera model ${estimate.modelVersion}; '
                'result version ${estimate.version}',
              ),
              const SizedBox(height: 10),
              if (comparisons.isEmpty)
                const Text(
                  'No matching estimated and measured components are '
                  'available to compare.',
                )
              else
                ..._withSpacing(comparisons),
            ],
          ),
        ),
      ),
    );
  }

  static String _agreement(String estimated, String measured) =>
      estimated.trim().toLowerCase() == measured.trim().toLowerCase()
          ? 'Yes'
          : 'No';

  static List<Widget> _withSpacing(List<Widget> children) {
    return [
      for (var index = 0; index < children.length; index++) ...[
        if (index > 0) const SizedBox(height: 10),
        children[index],
      ],
    ];
  }
}

class _NumericComparison extends StatelessWidget {
  const _NumericComparison({
    required this.label,
    required this.estimated,
    required this.measured,
    required this.unit,
  });

  final String label;
  final double estimated;
  final double measured;
  final String unit;

  @override
  Widget build(BuildContext context) {
    final difference = measured - estimated;
    final signed =
        '${difference >= 0 ? '+' : ''}${difference.toStringAsFixed(1)}';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Estimated $label: ${estimated.toStringAsFixed(1)} $unit'),
        Text('Measured $label: ${measured.toStringAsFixed(1)} $unit'),
        Text('Signed difference: $signed $unit'),
        Text(
            'Absolute difference: ${difference.abs().toStringAsFixed(1)} $unit'),
      ],
    );
  }
}
