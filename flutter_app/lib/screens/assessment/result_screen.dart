import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';
import '../../models/assessment_result.dart';
import '../../providers/assessment_provider.dart';
import '../shared/app_scaffold.dart';
import '../shared/status_badge.dart';

class ResultScreen extends ConsumerWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final result = ref.watch(assessmentResultProvider);

    if (result == null) {
      return AppScaffold(
        currentIndex: 0,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('No assessment result available.'),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () => context.go('/'),
                child: Text(t('run_assessment', ref)),
              ),
            ],
          ),
        ),
      );
    }

    return AppScaffold(
      currentIndex: 0,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _statusBanner(context, ref, result),
            if (_usesEstimatedEvidence(result)) ...[
              const SizedBox(height: 12),
              _estimateDisclosure(context, ref),
            ],
            const SizedBox(height: 16),
            _photoSection(context, ref, result),
            const SizedBox(height: 16),
            _metricCards(context, ref, result),
            if (result.mlPrediction != null) ...[
              const SizedBox(height: 16),
              _mlSection(context, ref, result.mlPrediction!),
            ],
            if (result.muac?.requiresConfirmation == true) ...[
              const SizedBox(height: 12),
              _muacNote(context, ref),
            ],
            const SizedBox(height: 24),
            _actionButtons(context, ref),
          ],
        ),
      ),
    );
  }

  Widget _statusBanner(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final haz = result.nutrition.hazStatus;

    final poshan = result.poshan.finalStatus;

    String title;
    String message;
    Color color;

    if (poshan == 'SAM') {
      title = t('banner_sam_title', ref);
      message = t('banner_sam_msg', ref);
      color = Colors.red;
    } else if (poshan == 'MAM') {
      title = t('banner_mam_title', ref);
      message = t('banner_mam_msg', ref);
      color = Colors.orange;
    } else if (haz != null && haz.toLowerCase().contains('stunted')) {
      title = haz;
      message = t('banner_stunted_msg', ref);
      color = Colors.amber.shade700;
    } else if (poshan == 'Normal') {
      title = t('banner_normal_title', ref);
      message = t('banner_normal_msg', ref);
      color = Colors.green;
    } else {
      title = t('banner_unknown_title', ref);
      message = t('banner_unknown_msg', ref);
      color = Colors.grey;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        border: Border(left: BorderSide(color: color, width: 4)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${result.childName} — ${result.ageMonths.toStringAsFixed(1)} ${t('months_unit', ref)}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: color,
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 4),
          Text(message),
          const SizedBox(height: 6),
          Text(
            'Poshan Setu status: $poshan',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          if (result.poshan.rationale.isNotEmpty) Text(result.poshan.rationale),
          if (result.mlPrediction == null) ...[
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.amber.shade100,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                t('fallback_used', ref),
                style: const TextStyle(fontSize: 11),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _photoSection(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final estimationMethod = result.measurement.estimationMethod;
    final confidence = result.measurement.confidenceScore;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (confidence != null) ...[
              Row(
                children: [
                  Text('${t('pose_confidence', ref)}: '),
                  Expanded(
                    child: LinearProgressIndicator(
                      value: confidence,
                      backgroundColor: Colors.grey.shade200,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text('${(confidence * 100).toStringAsFixed(0)}%'),
                ],
              ),
              const SizedBox(height: 4),
            ],
            if (estimationMethod != null)
              Text(
                estimationMethod == 'who_statistical'
                    ? t('analysis_pose_who', ref)
                    : 'Method: $estimationMethod',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }

  Widget _metricCards(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final heightIsDirect = {
      'manual',
      'reference_object',
    }.contains(result.measurement.heightMethod);
    final weightIsDirect = result.measurement.weightMethod == 'manual';

    return Column(
      children: [
        Card(
          child: ListTile(
            title: const Text('Poshan Setu v1 classification'),
            subtitle: Text(
              'Measured BMI: ${result.poshan.bmi?.toStringAsFixed(2) ?? '—'} '
              '(${result.poshan.bmiStatus}) · Measured MUAC: '
              '${result.poshan.muacStatus}\nEstimated WHO/MUAC screening: '
              '${result.combinedNutrition.status} · Stunting (HAZ): '
              '${result.nutrition.hazStatus ?? 'Insufficient data'}',
            ),
            trailing: StatusBadge(status: result.poshan.finalStatus),
          ),
        ),
        const SizedBox(height: 8),
        _metricCard(
          context,
          ref,
          title: t('metric_height', ref),
          value: result.measurement.effectiveHeightCm,
          unit: 'cm',
          source: _heightSource(ref, result.measurement),
          zscore: heightIsDirect ? result.nutrition.hazZscore : null,
          status: heightIsDirect ? result.nutrition.hazStatus : null,
        ),
        const SizedBox(height: 8),
        _metricCard(
          context,
          ref,
          title: t('metric_weight', ref),
          value: result.measurement.predictedWeightKg ??
              result.measurement.manualWeightKg,
          unit: 'kg',
          source: _weightSource(ref, result.measurement),
          zscore: heightIsDirect && weightIsDirect
              ? result.nutrition.whzZscore
              : null,
          status: heightIsDirect && weightIsDirect
              ? result.nutrition.whzStatus
              : null,
          extras: _weightExtras(context, ref, result.measurement),
        ),
        const SizedBox(height: 8),
        _muacCard(context, ref, result.muac),
      ],
    );
  }

  bool _usesEstimatedEvidence(AssessmentResult result) {
    final measurement = result.measurement;
    return measurement.heightMethod != 'manual' ||
        measurement.weightMethod != 'manual' ||
        result.muac?.isDirectMeasurement != true;
  }

  Widget _estimateDisclosure(BuildContext context, WidgetRef ref) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        border: Border.all(color: Colors.amber.shade700),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            t('screening_estimates_title', ref),
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  color: Colors.amber.shade900,
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 4),
          Text(t('screening_estimates_body', ref)),
        ],
      ),
    );
  }

  String _heightSource(WidgetRef ref, Measurement measurement) {
    switch (measurement.heightMethod) {
      case 'manual':
        return t('badge_manual', ref);
      case 'who_statistical':
      case 'who_median_estimated':
      case 'image_estimated':
        return t('badge_who_age_estimate', ref);
      case 'reference_object':
        return t('badge_image', ref);
      default:
        return t('badge_undetected', ref);
    }
  }

  String _weightSource(WidgetRef ref, Measurement measurement) {
    switch (measurement.weightMethod) {
      case 'manual':
        return t('badge_manual', ref);
      case 'ml_estimated':
        return t('badge_ml_estimate', ref);
      case 'who_statistical':
      case 'who_median_estimated':
        return t('badge_who_weight_estimate', ref);
      default:
        return t('badge_undetected', ref);
    }
  }

  Widget? _weightExtras(BuildContext context, WidgetRef ref, Measurement m) {
    if (!m.sideViewUsed) return null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('badge_side_view_ok', ref),
          style: const TextStyle(fontSize: 12, color: Colors.teal),
        ),
        if (m.chestDepthCm != null)
          Text(
            '${t('chest_depth', ref)} ${m.chestDepthCm!.toStringAsFixed(1)} cm',
            style: const TextStyle(fontSize: 11),
          ),
        if (m.abdDepthCm != null)
          Text(
            '${t('abd_depth', ref)} ${m.abdDepthCm!.toStringAsFixed(1)} cm',
            style: const TextStyle(fontSize: 11),
          ),
      ],
    );
  }

  Widget _metricCard(
    BuildContext context,
    WidgetRef ref, {
    required String title,
    required double? value,
    required String unit,
    required String source,
    required double? zscore,
    required String? status,
    Widget? extras,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        title,
                        style: Theme.of(context).textTheme.titleSmall,
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 1,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade200,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          source,
                          style: const TextStyle(fontSize: 11),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value != null ? '${value.toStringAsFixed(1)} $unit' : '—',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  if (zscore != null)
                    Text(
                      'Z-score: ${zscore.toStringAsFixed(2)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  if (extras != null) ...[const SizedBox(height: 4), extras],
                ],
              ),
            ),
            if (status != null) StatusBadge(status: status),
          ],
        ),
      ),
    );
  }

  Widget _muacCard(BuildContext context, WidgetRef ref, MuacDetail? muac) {
    if (muac == null) {
      return _metricCard(
        context,
        ref,
        title: t('metric_muac', ref),
        value: null,
        unit: 'cm',
        source: t('badge_na', ref),
        zscore: null,
        status: null,
      );
    }
    final source = muac.muacMethod == 'manual'
        ? t('badge_tape', ref)
        : t('badge_est', ref);
    return _metricCard(
      context,
      ref,
      title: t('metric_muac', ref),
      value: muac.muacCm,
      unit: 'cm',
      source: source,
      zscore: null,
      status: muac.muacStatus,
    );
  }

  Widget _mlSection(BuildContext context, WidgetRef ref, MlPrediction ml) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('ml_wasting_title', ref),
              style: Theme.of(context).textTheme.titleSmall,
            ),
            Text(
              t('ml_wasting_sub', ref),
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            _probabilityBar(
              t('sam_probability', ref),
              ml.samProbability,
              Colors.red,
            ),
            const SizedBox(height: 4),
            _probabilityBar(
              t('mam_probability', ref),
              ml.mamProbability,
              Colors.orange,
            ),
            const SizedBox(height: 4),
            _probabilityBar(
              t('normal_probability', ref),
              ml.normalProbability,
              Colors.green,
            ),
            if (ml.estimatedWeightKg != null) ...[
              const SizedBox(height: 8),
              Text(
                '${t('ml_estimated_weight', ref)} ${ml.estimatedWeightKg!.toStringAsFixed(2)} kg',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _probabilityBar(String label, double? value, Color color) {
    final pct = value ?? 0;
    return Row(
      children: [
        SizedBox(
          width: 120,
          child: Text(label, style: const TextStyle(fontSize: 12)),
        ),
        Expanded(
          child: LinearProgressIndicator(
            value: pct,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          '${(pct * 100).toStringAsFixed(0)}%',
          style: const TextStyle(fontSize: 12),
        ),
      ],
    );
  }

  Widget _muacNote(BuildContext context, WidgetRef ref) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber.shade200),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, color: Colors.amber.shade800, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: DefaultTextStyle.of(
                  context,
                ).style.copyWith(fontSize: 13),
                children: [
                  TextSpan(
                    text: '${t('muac_note_strong', ref)} ',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  TextSpan(text: t('muac_note_text', ref)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionButtons(BuildContext context, WidgetRef ref) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        FilledButton.icon(
          onPressed: () {
            ref.read(assessmentResultProvider.notifier).state = null;
            context.go('/');
          },
          icon: const Icon(Icons.refresh),
          label: Text(t('assess_another', ref)),
        ),
        OutlinedButton.icon(
          onPressed: () => context.go('/children'),
          icon: const Icon(Icons.people),
          label: Text(t('view_all_children', ref)),
        ),
      ],
    );
  }
}
