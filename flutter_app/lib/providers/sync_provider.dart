import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/sync_service.dart';
import 'api_provider.dart';
import 'auth_provider.dart';
import 'database_provider.dart';

final syncServiceProvider = Provider<SyncService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  final auth = ref.watch(authProvider);
  final token = auth.token;
  return SyncService(
    db: ref.watch(databaseProvider),
    visitDao: ref.watch(visitDaoProvider),
    childDao: ref.watch(childDaoProvider),
    syncDao: ref.watch(syncQueueDaoProvider),
    baseUrl: effectiveBaseUrl(baseUrl),
    authToken: token,
    ownerUserId: auth.user?.id,
    onUnauthorized: () => ref.read(authProvider.notifier).onTokenRejected(),
  );
});

/// Live count of pending/failed visits awaiting sync.
final pendingSyncCountProvider = StreamProvider<int>((ref) {
  final ownerUserId = ref.watch(authProvider).user?.id;
  if (ownerUserId == null) return Stream.value(0);
  return ref.watch(syncQueueDaoProvider).watchPendingCount(ownerUserId);
});

/// Long-lived listener: triggers sync on connectivity changes.
/// Started by main.dart via `ref.read(syncTriggerProvider)`.
final syncTriggerProvider = Provider<StreamSubscription>((ref) {
  final sub = Connectivity().onConnectivityChanged.listen((results) {
    final online = results.any((r) =>
        r == ConnectivityResult.wifi ||
        r == ConnectivityResult.mobile ||
        r == ConnectivityResult.ethernet);
    if (online) {
      ref.read(syncServiceProvider).runOnce();
    }
  });
  ref.onDispose(sub.cancel);
  return sub;
});
