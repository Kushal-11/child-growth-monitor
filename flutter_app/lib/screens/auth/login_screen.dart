import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

    final theme = Theme.of(context);
            const _LoginBackdrop(),
              child: _LoginActions(ref: ref),
                  AppSpacing.lg,
                  84,
                  AppSpacing.lg,
                          style: theme.textTheme.headlineSmall,
                          style: theme.textTheme.bodyLarge?.copyWith(
                            color: AppColors.textSecondary,
                          ),
                        const _OfflineBadge(),
                        const SizedBox(height: AppSpacing.xxl),
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(AppSpacing.lg),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                                Text(
                                  t('sign_in', ref),
                                  style: theme.textTheme.titleLarge,
                                const SizedBox(height: AppSpacing.xs),
                                  t('sign_in_help', ref),
                                  style: theme.textTheme.bodyMedium,
                                ),
                                const SizedBox(height: AppSpacing.xl),
                                if (state.error != null) ...[
                                  _LoginError(message: state.error!),
                                  const SizedBox(height: AppSpacing.lg),
                                ],
                                TextFormField(
                                  key: const Key('login_username'),
                                  initialValue: state.username,
                                  enabled: !state.isSubmitting,
                                  autofillHints: const [
                                    AutofillHints.username,
                                    AutofillHints.email,
                                  ],
                                  keyboardType: TextInputType.emailAddress,
                                  textInputAction: TextInputAction.next,
                                  onChanged: notifier.updateUsername,
                                  decoration: InputDecoration(
                                    labelText: t('username_or_email', ref),
                                    prefixIcon: const Icon(
                                      Icons.person_outline_rounded,
                                    ),
                                    errorText: state.showValidationErrors &&
                                            state.username.trim().isEmpty
                                        ? t('required_field', ref)
                                        : null,
                                  ),
                                ),
                                const SizedBox(height: AppSpacing.lg),
                                TextFormField(
                                  key: const Key('login_password'),
                                  initialValue: state.password,
                                  enabled: !state.isSubmitting,
                                  obscureText: !state.passwordVisible,
                                  autofillHints: const [AutofillHints.password],
                                  textInputAction: TextInputAction.done,
                                  onChanged: notifier.updatePassword,
                                  onFieldSubmitted: (_) {
                                    if (!state.isSubmitting) notifier.submit();
                                  },
                                  decoration: InputDecoration(
                                    labelText: t('password', ref),
                                    prefixIcon: const Icon(
                                      Icons.lock_outline_rounded,
                                    ),
                                    suffixIcon: IconButton(
                                      key: const Key(
                                        'login_password_visibility',
                                      ),
                                      tooltip: state.passwordVisible
                                          ? t('hide_password', ref)
                                          : t('show_password', ref),
                                      onPressed:
                                          notifier.togglePasswordVisibility,
                                      icon: Icon(
                                        state.passwordVisible
                                            ? Icons.visibility_off_outlined
                                            : Icons.visibility_outlined,
                                    ),
                                    errorText: state.showValidationErrors &&
                                            state.password.isEmpty
                                        ? t('required_field', ref)
                                        : null,
                                  ),
                                ),
                                const SizedBox(height: AppSpacing.md),
                                _ServerHelpLink(ref: ref),
                                const SizedBox(height: AppSpacing.xxl),
                                FilledButton.icon(
                                  key: const Key('login_submit'),
                                  onPressed: state.isSubmitting
                                      ? null
                                      : notifier.submit,
                                  icon: state.isSubmitting
                                      ? const SizedBox(
                                          height: 18,
                                          width: 18,
                                          child: CircularProgressIndicator(
                                            strokeWidth: 2,
                                            color: Colors.white,
                                          ),
                                        )
                                      : const Icon(Icons.login_rounded),
                                  label: Text(
                                    state.isSubmitting
                                        ? t('signing_in', ref)
                                        : t('log_in', ref),
                                  ),
                        const _PrivacyNote(),
class _LoginBackdrop extends StatelessWidget {
  const _LoginBackdrop();

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned(
          top: -96,
          right: -96,
          child: Container(
            width: 240,
            height: 240,
            decoration: BoxDecoration(
              color: AppColors.primaryContainer.withValues(alpha: 0.8),
              shape: BoxShape.circle,
            ),
          ),
        ),
        Positioned(
          left: -72,
          bottom: 120,
          child: Container(
            width: 180,
            height: 180,
            decoration: BoxDecoration(
              color: AppColors.successSurface.withValues(alpha: 0.7),
              shape: BoxShape.circle,
            ),
          ),
        ),
      ],
    );
  }
}

class _LoginActions extends StatelessWidget {
  const _LoginActions({required this.ref});

  final WidgetRef ref;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        TextButton(
          key: const Key('login_language'),
          onPressed: () => ref.read(localeProvider.notifier).toggle(),
          child: Text(
            ref.watch(localeProvider) == 'en'
                ? t('lang_mr', ref)
                : t('lang_en', ref),
          ),
        ),
        IconButton.filledTonal(
          key: const Key('login_server_settings'),
          tooltip: t('server_settings', ref),
          onPressed: () => context.push('/settings'),
          icon: const Icon(Icons.settings_outlined),
        ),
      ],
    );
  }
}

class _OfflineBadge extends ConsumerWidget {
  const _OfflineBadge();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: 6,
        ),
        decoration: BoxDecoration(
          color: AppColors.primaryContainer,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.offline_bolt_outlined,
              size: 16,
              color: AppColors.primary,
            ),
            const SizedBox(width: 6),
            Text(
              t('offline_ready', ref),
              style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: AppColors.primary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ServerHelpLink extends StatelessWidget {
  const _ServerHelpLink({required this.ref});

  final WidgetRef ref;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerRight,
      child: TextButton.icon(
        key: const Key('login_configure_server'),
        onPressed: () => context.push('/settings'),
        icon: const Icon(Icons.dns_outlined, size: 18),
        label: Text(t('configure_server', ref)),
      ),
    );
  }
}

class _PrivacyNote extends ConsumerWidget {
  const _PrivacyNote();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(
          Icons.shield_outlined,
          size: 16,
          color: AppColors.textSecondary,
        ),
        const SizedBox(width: 6),
        Flexible(
          child: Text(
            t('login_privacy_note', ref),
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ),
      ],
    );
  }
}

    return Container(
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: BoxDecoration(
        color: AppColors.dangerSurface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.dangerBorder),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: AppColors.dangerText),
          const SizedBox(width: AppSpacing.sm),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: AppColors.dangerText,
          ),
        ],
