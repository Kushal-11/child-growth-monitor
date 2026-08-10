import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../guided_capture/domain/camera_screening_result.dart';
import 'estimate_provenance_card.dart';
import 'report_metric_card.dart';

const String estimatedReportNotice =
    'These are research-only contactless estimates from AR depth, guided photos, and the '
    'on-device model. Review each source and estimated range below.';

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
          if (result.reportableHeightCm case final height?)
            ReportMetricCard(
              label: 'Estimated height',
              value: '${height.toStringAsFixed(1)} cm',
              detail: _detail(
                source: result.heightSource,
                lower: result.heightRangeLowerCm,
                upper: result.heightRangeUpperCm,
                unit: 'cm',
              ),
              icon: Icons.height,
            )
          else
            const _UnavailableMetric(
              message: 'Height estimate unavailable. Retry the guided scan.',
            ),
          if (result.reportableWeightKg case final weight?)
            ReportMetricCard(
              label: 'Estimated weight',
              value: '${weight.toStringAsFixed(1)} kg',
              detail: _detail(
                source: result.weightSource,
                lower: result.weightRangeLowerKg,
                upper: result.weightRangeUpperKg,
                unit: 'kg',
              ),
              icon: Icons.monitor_weight_outlined,
            )
          else
            const _UnavailableMetric(
              message: 'Weight estimate unavailable. Retry the guided scan.',
            ),
          if (result.reportableMuacCm case final muac?)
            ReportMetricCard(
              label: 'Estimated MUAC',
              value: '${muac.toStringAsFixed(1)} cm',
              detail: _detail(
                source: result.muacSource,
                lower: result.muacRangeLowerCm,
                upper: result.muacRangeUpperCm,
                unit: 'cm',
              ),
              icon: Icons.straighten,
            )
          else
            const _UnavailableMetric(
              message: 'MUAC estimate unavailable. Retry with the arm clear.',
            ),
          if (result.reportableStuntingStatus case final status?)
            ReportMetricCard(
              label: 'Estimated stunting status',
              value: status,
              detail: result.reportableHaz == null
                  ? null
                  : 'Estimated HAZ ${result.reportableHaz!.toStringAsFixed(2)}',
              icon: Icons.show_chart,
            ),
          if (result.reportableWastingStatus case final status?)
            ReportMetricCard(
              label: 'Estimated wasting status',
              value: status,
              detail: result.reportableWhz == null
                  ? null
                  : 'Estimated WHZ ${result.reportableWhz!.toStringAsFixed(2)}',
              icon: Icons.analytics_outlined,
            ),
          if (result.experimentalOverallCategory case final category?)
            ReportMetricCard(
              label: 'Experimental camera screening category',
              value: category,
              detail: 'Synthetic research output; not a WHO diagnosis',
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
      arcoreDepthHeightSourceV3 => 'ARCore depth height estimate',
      arcoreDepthHeightSourceV2 => 'ARCore depth height estimate (v2)',
      arcoreGeometryWeightSourceV3 =>
        'ARCore body geometry + on-device weight model',
      arcoreHeightPhotoGeometryWeightSourceV3 =>
        'ARCore height + guided-photo geometry + on-device weight model',
      arcoreArmMuacSourceV3 => 'ARCore upper-arm cross-section estimate',
      legacyWhoHeightSourceV1 => 'WHO height-for-age statistical estimate',
      experimentalMlWeightSourceV1 =>
        'Experimental on-device ML weight estimate',
      'ml_weight_estimator_v1' => 'Legacy on-device ML weight estimate',
      legacyWhoWeightSourceV1 => 'WHO median with body-build adjustment',
      _ => source,
    };
  }

  static String? _detail({
    required String? source,
    required double? lower,
    required double? upper,
    required String unit,
  }) {
    final sourceLabel = _sourceLabel(source);
    final range = lower != null && upper != null
        ? 'Estimated range ${lower.toStringAsFixed(1)}-'
            '${upper.toStringAsFixed(1)} $unit'
        : null;
    return [sourceLabel, range].whereType<String>().join(' · ').nullIfEmpty;
  }
}

extension on String {
  String? get nullIfEmpty => isEmpty ? null : this;
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
