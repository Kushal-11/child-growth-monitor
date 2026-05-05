import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/api_provider.dart';
import 'providers/sync_provider.dart';
import 'router.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final savedUrl = await loadSavedBaseUrl();
  runApp(
    ProviderScope(
      overrides: [
        baseUrlProvider.overrideWith((ref) => savedUrl),
      ],
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
    // Start the connectivity-triggered sync listener.
    ref.read(syncTriggerProvider);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'SNEH Growth Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      routerConfig: appRouter,
    );
  }
}
