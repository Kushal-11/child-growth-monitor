import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../features/auth/providers/login_form_provider.dart';
import '../../l10n/l10n_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';

class LoginScreen extends ConsumerWidget {
  const LoginScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(loginFormProvider);
    final notifier = ref.read(loginFormProvider.notifier);
    final theme = Theme.of(context);

    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            const Positioned.fill(
              child: IgnorePointer(child: _LoginBackdrop()),
            ),
            Positioned(
              top: AppSpacing.sm,
              right: AppSpacing.md,
              child: _LoginActions(ref: ref),
            ),
            Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(
                  AppSpacing.xxl,
                  84,
                  AppSpacing.xxl,
                  AppSpacing.section,
                ),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 440),
                  child: AutofillGroup(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const _BrandMark(),
                        const SizedBox(height: AppSpacing.xl),
                        Text(
                          t('app_title', ref),
                          textAlign: TextAlign.center,
                          style: theme.textTheme.headlineSmall,
                        ),
                        const SizedBox(height: AppSpacing.sm),
                        Text(
                          t('login_welcome', ref),
                          textAlign: TextAlign.center,
                          style: theme.textTheme.bodyLarge?.copyWith(
                            color: AppColors.textSecondary,
                          ),
                        ),
                        const SizedBox(height: AppSpacing.md),
                        const _OfflineBadge(),
                        const SizedBox(height: AppSpacing.xxl),
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(AppSpacing.lg),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: [
                                Text(
                                  t('sign_in', ref),
                                  style: theme.textTheme.titleLarge,
                                ),
                                const SizedBox(height: AppSpacing.xs),
                                Text(
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
                                    if (!state.isSubmitting) {
                                      notifier.submit();
                                    }
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
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: AppSpacing.xl),
                        const _PrivacyNote(),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

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

class _BrandMark extends StatelessWidget {
  const _BrandMark();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          color: AppColors.primary,
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(
              color: AppColors.primary.withValues(alpha: 0.18),
              blurRadius: 24,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: const Icon(
          Icons.monitor_heart_rounded,
          color: Colors.white,
          size: 36,
        ),
      ),
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

class _LoginError extends StatelessWidget {
  const _LoginError({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      liveRegion: true,
      child: Container(
        padding: const EdgeInsets.all(AppSpacing.md),
        decoration: BoxDecoration(
          color: AppColors.dangerSurface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.dangerBorder),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(
              Icons.error_outline_rounded,
              color: AppColors.dangerText,
              size: 20,
            ),
            const SizedBox(width: AppSpacing.sm),
            Expanded(
              child: Text(
                message,
                style: const TextStyle(color: AppColors.dangerText),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
