import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../l10n/l10n_provider.dart';
import '../../../theme/app_colors.dart';
import '../../../theme/app_spacing.dart';

class AssessmentHeader extends ConsumerWidget {
  const AssessmentHeader({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      height: 112,
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.xl),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.progressTrack)),
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: const BoxDecoration(
              color: AppColors.primary,
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.monitor_heart_rounded,
              size: 21,
              color: Colors.white,
            ),
          ),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: Text(
              t('app_title', ref),
              maxLines: 2,
              style: Theme.of(context).textTheme.headlineSmall,
            ),
          ),
          TextButton(
            key: const Key('assessment_language'),
            onPressed: () => ref.read(localeProvider.notifier).toggle(),
            child: Text(
              ref.watch(localeProvider) == 'en'
                  ? t('lang_mr', ref)
                  : t('lang_en', ref),
            ),
          ),
        ],
      ),
    );
  }
}
