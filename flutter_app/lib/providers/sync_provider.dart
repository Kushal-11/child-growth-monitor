import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/sync_service.dart';
import 'api_provider.dart';
import 'database_provider.dart';

final syncServiceProvider = Provider<SyncService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  return SyncService(
    db: ref.watch(databaseProvider),
    visitDao: ref.watch(visitDaoProvider),
    childDao: ref.watch(childDaoProvider),
    syncDao: ref.watch(syncQueueDaoProvider),
    baseUrl: effectiveBaseUrl(baseUrl),
  );
});

/// Live count of pending/failed visits awaiting sync.
final pendingSyncCountProvider = StreamProvider<int>((ref) {
  return ref.watch(syncQueueDaoProvider).watchPendingCount();
});

/// Long-lived listener: triggers sync on connectivity changes.
/// Started by main.dart via `ref.read(syncTriggerProvider)`.
final syncTriggerProvider = Provider<StreamSubscription>((ref) {
  final svc = ref.watch(syncServiceProvider);
  final sub = Connectivity().onConnectivityChanged.listen((results) {
    final online = results.any((r) =>
        r == ConnectivityResult.wifi ||
        r == ConnectivityResult.mobile ||
        r == ConnectivityResult.ethernet);
    if (online) {
      svc.runOnce();
    }
  });
  ref.onDispose(sub.cancel);
  return sub;
});
