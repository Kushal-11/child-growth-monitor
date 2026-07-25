import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/api_provider.dart';
import 'providers/auth_provider.dart';
import 'providers/sync_provider.dart';
import 'router.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final savedUrl = await loadSavedBaseUrl();
  runApp(
    ProviderScope(
      overrides: [baseUrlProvider.overrideWith((ref) => savedUrl)],
      child: const ChildGrowthApp(),
    ),
  );
}

class ChildGrowthApp extends ConsumerStatefulWidget {
  const ChildGrowthApp({super.key});

  @override
  ConsumerState<ChildGrowthApp> createState() => _ChildGrowthAppState();
}

class _ChildGrowthAppState extends ConsumerState<ChildGrowthApp> {
  @override
  void initState() {
    super.initState();
    ref.listenManual<AuthState>(authProvider, (previous, next) {
      if (next.status == AuthStatus.authenticated &&
          previous?.status != AuthStatus.authenticated) {
        // Drain this owner's offline queue after cached-session restore or
        // interactive login. SyncService enforces owner scoping.
        unawaited(ref.read(syncServiceProvider).runOnce());
      }
    });
    // Restore any cached auth session so the router gate resolves immediately.
    ref.read(authProvider.notifier).restore();
    // Start the connectivity-triggered sync listener.
    ref.read(syncTriggerProvider);
  }

  @override
  Widget build(BuildContext context) {
    final router = ref.watch(routerProvider);
    return MaterialApp.router(
      title: 'SNEH Growth Monitor',
      theme: AppTheme.light(),
      routerConfig: router,
    );
  }
}
