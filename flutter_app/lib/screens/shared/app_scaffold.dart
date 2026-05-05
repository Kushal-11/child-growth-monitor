import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/sync_provider.dart';

class AppScaffold extends ConsumerWidget {
  const AppScaffold({
    super.key,
    required this.child,
    required this.currentIndex,
  });

  final Widget child;
  final int currentIndex;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final pending = ref.watch(pendingSyncCountProvider).value ?? 0;
    return Scaffold(
      appBar: AppBar(
        title: Text(t('app_title', ref)),
        actions: [
          IconButton(
            tooltip: t('sync_status', ref),
            onPressed: () => context.go('/settings'),
            icon: Stack(
              clipBehavior: Clip.none,
              children: [
                Icon(pending == 0 ? Icons.cloud_done : Icons.cloud_upload),
                if (pending > 0)
                  Positioned(
                    right: -4,
                    top: -4,
                    child: Container(
                      padding: const EdgeInsets.all(2),
                      decoration: const BoxDecoration(
                        color: Colors.red,
                        shape: BoxShape.circle,
                      ),
                      constraints:
                          const BoxConstraints(minWidth: 16, minHeight: 16),
                      child: Text(
                        '$pending',
                        style:
                            const TextStyle(color: Colors.white, fontSize: 10),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          TextButton(
            onPressed: () => ref.read(localeProvider.notifier).toggle(),
            child: Text(
              ref.watch(localeProvider) == 'en'
                  ? t('lang_mr', ref)
                  : t('lang_en', ref),
              style: TextStyle(
                color: Theme.of(context).colorScheme.onPrimary,
              ),
            ),
          ),
        ],
      ),
      body: child,
      bottomNavigationBar: NavigationBar(
        selectedIndex: currentIndex,
        onDestinationSelected: (index) {
          switch (index) {
            case 0:
              context.go('/');
            case 1:
              context.go('/children');
            case 2:
              context.go('/settings');
          }
        },
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.assessment),
            label: t('nav_assess', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.people),
            label: t('nav_children', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.settings),
            label: t('nav_settings', ref),
          ),
        ],
      ),
    );
  }
}
