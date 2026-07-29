import 'package:flutter/material.dart';

import '../../../theme/app_colors.dart';
import '../../../theme/app_spacing.dart';

class AssessmentProgress extends StatelessWidget {
  const AssessmentProgress({
    super.key,
    required this.step,
    required this.label,
  });

  final int step;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: label,
      value: '${step + 1} of 3',
      child: Container(
        height: 62,
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.xl,
          vertical: AppSpacing.md,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    label,
                    key: Key('assessment_step_${step + 1}'),
                    style: Theme.of(context).textTheme.labelLarge?.copyWith(
                          color: AppColors.textSecondary,
                          fontWeight: FontWeight.w600,
                        ),
                  ),
                ),
                Text(
                  '${step + 1}/3',
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(
                        color: AppColors.primary,
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ],
            ),
            const SizedBox(height: AppSpacing.sm),
            ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                minHeight: 6,
                value: (step + 1) / 3,
                color: AppColors.primary,
                backgroundColor: AppColors.progressTrack,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
