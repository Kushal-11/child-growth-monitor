import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/api_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';
import '../../providers/sync_provider.dart';
import '../../services/image_storage_service.dart';
import '../shared/app_scaffold.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  final _urlController = TextEditingController();
  bool _loading = false;
  bool? _healthy;
  String? _error;
  bool _syncing = false;
  int? _bytesUsed;

  @override
  void initState() {
    super.initState();
    _loadUrl();
    _refreshStorage();
  }

  Future<void> _loadUrl() async {
    final saved = await loadSavedBaseUrl();
    if (!mounted) return;
    _urlController.text = saved;
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _saveAndTest() async {
    final url = _urlController.text.trim();
    if (!isAllowlistedHost(url)) {
      setState(() {
        _healthy = false;
        _error = t('invalid_url', ref);
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _healthy = null;
    });

    try {
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;
      final healthy = await ref.read(apiProvider).checkHealth();
      if (!mounted) return;
      setState(() => _healthy = healthy);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _healthy = false;
        _error = e.toString();
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _resetToDefault() {
    _urlController.text = 'http://10.0.2.2:8000';
    setState(() {
      _healthy = null;
      _error = null;
    });
  }

  Future<void> _refreshStorage() async {
    final used = await ImageStorageService().totalUsedBytes();
    if (!mounted) return;
    setState(() => _bytesUsed = used);
  }

  Future<void> _syncNow() async {
    setState(() => _syncing = true);
    try {
      await ref.read(syncServiceProvider).runOnce();
    } finally {
      if (mounted) setState(() => _syncing = false);
    }
  }

  Future<void> _clearImages() async {
    await ImageStorageService().clearAll();
    await _refreshStorage();
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AppScaffold(
      currentIndex: 2,
      child: ListView(
        padding: const EdgeInsets.all(AppSpacing.lg),
        children: [
          Text(
            t('settings_heading', ref),
            style: theme.textTheme.headlineSmall,
          ),
          const SizedBox(height: AppSpacing.xs),
          Text(
            t('settings_subtitle', ref),
            style: theme.textTheme.bodyMedium,
          ),
          const SizedBox(height: AppSpacing.xl),

          // Server Connection card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _SettingsSectionHeader(
                    icon: Icons.dns_outlined,
                    title: t('server_connection', ref),
                    subtitle: t('server_connection_help', ref),
                  ),
                  const SizedBox(height: AppSpacing.md),
                  TextFormField(
                    controller: _urlController,
                    decoration: InputDecoration(
                      labelText: t('api_base_url', ref),
                      border: const OutlineInputBorder(),
                    ),
                    keyboardType: TextInputType.url,
                  ),
                  const SizedBox(height: AppSpacing.md),

                  // Status indicator
                  if (_loading)
                    const Padding(
                      padding: EdgeInsets.only(bottom: 12),
                      child: LinearProgressIndicator(),
                    ),
                  if (_healthy != null && !_loading)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Row(
                        children: [
                          Icon(
                            _healthy! ? Icons.check_circle : Icons.error,
                            color: _healthy!
                                ? AppColors.successText
                                : AppColors.error,
                            size: 18,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _healthy!
                                  ? t('connected_ms', ref)
                                  : _error ?? t('connection_failed', ref),
                              style: TextStyle(
                                color: _healthy!
                                    ? AppColors.successText
                                    : AppColors.error,
                                fontSize: 13,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                  // Buttons
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton(
                          onPressed: _loading ? null : _saveAndTest,
                          child: Text(t('save_and_test', ref)),
                        ),
                      ),
                      const SizedBox(width: 8),
                      TextButton(
                        onPressed: _loading ? null : _resetToDefault,
                        child: Text(t('reset_default', ref)),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: AppSpacing.lg),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _SettingsSectionHeader(
                    icon: Icons.cloud_sync_outlined,
                    title: t('sync_status', ref),
                    subtitle: t('sync_status_help', ref),
                  ),
                  const SizedBox(height: AppSpacing.md),
                  Consumer(builder: (context, ref, _) {
                    final pending =
                        ref.watch(pendingSyncCountProvider).value ?? 0;
                    return Text(
                      pending == 0
                          ? t('sync_all_synced', ref)
                          : '${t('sync_pending', ref)}: $pending',
                    );
                  }),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: _syncing ? null : _syncNow,
                    icon: _syncing
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.sync),
                    label: Text(t('sync_now', ref)),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: AppSpacing.lg),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _SettingsSectionHeader(
                    icon: Icons.folder_copy_outlined,
                    title: t('storage_title', ref),
                    subtitle: t('storage_help', ref),
                  ),
                  const SizedBox(height: AppSpacing.sm),
                  Text(
                    _bytesUsed == null
                        ? '...'
                        : '${t('storage_used', ref)}: ${_formatBytes(_bytesUsed!)}',
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton.icon(
                    onPressed: _clearImages,
                    icon: const Icon(Icons.delete_outline),
                    label: Text(t('storage_clear', ref)),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SettingsSectionHeader extends StatelessWidget {
  const _SettingsSectionHeader({
    required this.icon,
    required this.title,
    required this.subtitle,
  });

  final IconData icon;
  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: AppColors.primaryContainer,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Icon(icon, color: AppColors.primary),
        ),
        const SizedBox(width: AppSpacing.md),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: theme.textTheme.titleMedium),
              const SizedBox(height: 2),
              Text(subtitle, style: theme.textTheme.bodyMedium),
            ],
          ),
        ),
      ],
    );
  }
}
