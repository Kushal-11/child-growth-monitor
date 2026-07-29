import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../guided_capture/domain/camera_screening_result.dart';
import 'estimate_provenance_card.dart';
import 'report_metric_card.dart';

const String estimatedReportNotice =
    'Results are estimated from photos and may change after measured '
    'details are added';

class EstimatedReportView extends StatelessWidget {
  const EstimatedReportView({
    super.key,
    required this.result,
    required this.visitDate,
    required this.onAddMeasuredDetails,
  });

  final CameraScreeningResult result;
  final DateTime visitDate;
  final VoidCallback onAddMeasuredDetails;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Estimated Growth Screening Report',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 4),
          Text('Visit date: ${DateFormat('dd MMM yyyy').format(visitDate)}'),
          const SizedBox(height: 12),
          Card(
            color: Theme.of(context).colorScheme.secondaryContainer,
            child: const Padding(
              padding: EdgeInsets.all(14),
              child: Text(estimatedReportNotice),
            ),
          ),
          const SizedBox(height: 8),
          if (result.estimatedHeightCm case final height?)
            ReportMetricCard(
              label: 'Estimated height',
              value: '${height.toStringAsFixed(1)} cm',
              detail: _sourceLabel(result.heightSource),
              icon: Icons.height,
            )
          else
            const _UnavailableMetric(
              message: 'Height could not be estimated from the captured views.',
            ),
          if (result.estimatedWeightKg case final weight?)
            ReportMetricCard(
              label: 'Estimated weight',
              value: '${weight.toStringAsFixed(1)} kg',
              detail: _sourceLabel(result.weightSource),
              icon: Icons.monitor_weight_outlined,
            )
          else
            const _UnavailableMetric(
              message: 'Weight could not be estimated from the captured views.',
            ),
          if (result.estimatedStuntingStatus case final status?)
            ReportMetricCard(
              label: 'Estimated stunting status',
              value: status,
              detail: result.estimatedHaz == null
                  ? null
                  : 'Estimated HAZ ${result.estimatedHaz!.toStringAsFixed(2)}',
              icon: Icons.show_chart,
            ),
          if (result.estimatedWastingStatus case final status?)
            ReportMetricCard(
              label: 'Estimated wasting status',
              value: status,
              detail: result.estimatedWhz == null
                  ? null
                  : 'Estimated WHZ ${result.estimatedWhz!.toStringAsFixed(2)}',
              icon: Icons.analytics_outlined,
            ),
          if (result.experimentalOverallCategory case final category?)
            ReportMetricCard(
              label: 'Experimental camera screening category',
              value: category,
              detail: 'Supplied by the active camera classifier',
              icon: Icons.science_outlined,
            ),
          const SizedBox(height: 8),
          EstimateProvenanceCard(result: result),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: onAddMeasuredDetails,
            icon: const Icon(Icons.add_chart),
            label: const Text('Add Measured Details'),
          ),
        ],
      ),
    );
  }

  static String? _sourceLabel(String? source) {
    return switch (source) {
      'who_height_for_age_median_v1' =>
        'WHO height-for-age statistical estimate',
      'ml_weight_estimator_v1' => 'On-device ML weight estimate',
      'who_weight_for_height_median_body_build_v1' =>
        'WHO median with body-build adjustment',
      _ => source,
    };
  }
}

class _UnavailableMetric extends StatelessWidget {
  const _UnavailableMetric({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: const Icon(Icons.info_outline),
        title: Text(message),
      ),
    );
  }
}
