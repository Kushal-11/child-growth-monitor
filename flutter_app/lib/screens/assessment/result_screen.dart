import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../constants/config.dart';
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
            const SizedBox(height: 16),
            _photoSection(context, ref, result),
            const SizedBox(height: 16),
            _metricCards(context, ref, result),
            if (result.mlPrediction != null) ...[
              const SizedBox(height: 16),
              _mlSection(context, ref, result.mlPrediction!),
            ],
            if (result.muac?.muacMethod == 'estimated_from_whz') ...[
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
      BuildContext context, WidgetRef ref, AssessmentResult result) {
    final haz = result.nutrition.hazStatus;

    // WHO CMAM OR-rule: banner severity reflects WHZ, MUAC, AND the ML wasting
    // classifier together — not WHZ alone — so a tape-measured or ML-detected
    // SAM/MAM child is never shown the green "Normal" banner.
    final combined = result.combinedNutrition?.status ?? combineNutritionStatus(
      whzStatus: result.nutrition.whzStatus,
      muacStatus: result.muac?.muacStatus,
      mlStatus: result.mlPrediction?.wastingStatus,
    );

    String title;
    String message;
    Color color;

    if (combined == 'SAM') {
      title = t('banner_sam_title', ref);
      message = t('banner_sam_msg', ref);
      color = Colors.red;
    } else if (combined == 'MAM') {
      title = t('banner_mam_title', ref);
      message = t('banner_mam_msg', ref);
      color = Colors.orange;
    } else if (haz != null && haz.toLowerCase().contains('stunted')) {
      title = haz;
      message = t('banner_stunted_msg', ref);
      color = Colors.amber.shade700;
    } else if (combined == 'Normal') {
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
            style: Theme.of(context)
                .textTheme
                .titleMedium
                ?.copyWith(color: color, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Text(message),
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
      BuildContext context, WidgetRef ref, AssessmentResult result) {
    final annotatedImage = result.measurement.estimationMethod;
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
            if (annotatedImage != null)
              Text(
                'Method: $annotatedImage',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }

  Widget _metricCards(
      BuildContext context, WidgetRef ref, AssessmentResult result) {
    return Column(
      children: [
        _metricCard(
          context,
          ref,
          title: t('metric_height', ref),
          value: result.measurement.predictedHeightCm ??
              result.measurement.manualHeightCm,
          unit: 'cm',
          source: result.measurement.manualHeightCm != null
              ? t('badge_manual', ref)
              : result.measurement.predictedHeightCm != null
                  ? t('badge_image', ref)
                  : t('badge_undetected', ref),
          zscore: result.nutrition.hazZscore,
          status: result.nutrition.hazStatus,
        ),
        if (result.nutrition.bmi != null) ...[
          const SizedBox(height: 8),
          _metricCard(
            context,
            ref,
            title: 'BMI',
            value: result.nutrition.bmi,
            unit: 'kg/m²',
            source: 'BMI + MUAC protocol',
            status: result.nutrition.bmiStatus,
          ),
        ],
        const SizedBox(height: 8),
        _metricCard(
          context,
          ref,
          title: t('metric_weight', ref),
          value: result.measurement.predictedWeightKg ??
              result.measurement.manualWeightKg,
          unit: 'kg',
          source: result.measurement.manualWeightKg != null
              ? t('badge_manual', ref)
              : result.measurement.predictedWeightKg != null
                  ? t('badge_image', ref)
                  : t('badge_undetected', ref),
          zscore: result.nutrition.whzZscore,
          status: result.nutrition.whzStatus,
          extras: _weightExtras(context, ref, result.measurement),
        ),
        const SizedBox(height: 8),
        _muacCard(context, ref, result.muac),
      ],
    );
  }

  Widget? _weightExtras(
      BuildContext context, WidgetRef ref, Measurement m) {
    if (!m.sideViewUsed) return null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(t('badge_side_view_ok', ref),
            style: const TextStyle(fontSize: 12, color: Colors.teal)),
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
                      Text(title,
                          style: Theme.of(context).textTheme.titleSmall),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade200,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(source,
                            style: const TextStyle(fontSize: 11)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value != null
                        ? '${value.toStringAsFixed(1)} $unit'
                        : '—',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  if (zscore != null)
                    Text(
                      'Z-score: ${zscore.toStringAsFixed(2)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  if (extras != null) ...[
                    const SizedBox(height: 4),
                    extras,
                  ],
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

  Widget _mlSection(
      BuildContext context, WidgetRef ref, MlPrediction ml) {
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
                t('sam_probability', ref), ml.samProbability, Colors.red),
            const SizedBox(height: 4),
            _probabilityBar(
                t('mam_probability', ref), ml.mamProbability, Colors.orange),
            const SizedBox(height: 4),
            _probabilityBar(t('normal_probability', ref),
                ml.normalProbability, Colors.green),
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
        SizedBox(width: 120, child: Text(label, style: const TextStyle(fontSize: 12))),
        Expanded(
          child: LinearProgressIndicator(
            value: pct,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
        const SizedBox(width: 8),
        Text('${(pct * 100).toStringAsFixed(0)}%',
            style: const TextStyle(fontSize: 12)),
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
                style: DefaultTextStyle.of(context).style.copyWith(fontSize: 13),
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
