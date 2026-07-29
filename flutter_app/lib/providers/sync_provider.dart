import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/sync_service.dart';
import '../features/guided_capture/services/guided_sync_service.dart';
import 'api_provider.dart';
import 'assessment_service_provider.dart';
import 'auth_provider.dart';
import 'database_provider.dart';

final syncServiceProvider = Provider<SyncService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  final token = ref.watch(authProvider).token;
  return SyncService(
    db: ref.watch(databaseProvider),
    visitDao: ref.watch(visitDaoProvider),
    childDao: ref.watch(childDaoProvider),
    syncDao: ref.watch(syncQueueDaoProvider),
    baseUrl: effectiveBaseUrl(baseUrl),
    authToken: token,
    onUnauthorized: () => ref.read(authProvider.notifier).onTokenRejected(),
  );
});

final guidedSyncServiceProvider = Provider<GuidedSyncGateway>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  final token = ref.watch(authProvider).token;
  return GuidedSyncService(
    database: ref.watch(databaseProvider),
    outboxDao: ref.watch(syncOutboxDaoProvider),
    baseUrl: effectiveBaseUrl(baseUrl),
    authToken: token,
    imageStorage: ref.watch(imageStorageProvider),
    onUnauthorized: () => ref.read(authProvider.notifier).onTokenRejected(),
  );
});

class SyncCoordinator {
  const SyncCoordinator({
    required this.legacy,
    required this.guided,
    required this.ownerUserId,
  });

  final SyncService legacy;
  final GuidedSyncGateway guided;
  final int? ownerUserId;

  Future<void> runOnce() async {
    await legacy.runOnce();
    final owner = ownerUserId;
    if (owner != null) await guided.runOnce(owner);
  }
}

final syncCoordinatorProvider = Provider<SyncCoordinator>((ref) {
  return SyncCoordinator(
    legacy: ref.watch(syncServiceProvider),
    guided: ref.watch(guidedSyncServiceProvider),
    ownerUserId: ref.watch(authProvider).user?.id,
  );
});

/// Live count of pending/failed legacy and typed entities awaiting sync.
final pendingSyncCountProvider = StreamProvider<int>((ref) {
  final database = ref.watch(databaseProvider);
  return database
      .customSelect(
        "SELECT "
        "(SELECT COUNT(*) FROM sync_queue "
        "WHERE status IN ('pending', 'failed')) + "
        "(SELECT COUNT(*) FROM sync_outbox "
        "WHERE status IN ('pending', 'failed')) AS pending_count",
        readsFrom: {
          database.syncQueue,
          database.syncOutbox,
        },
      )
      .watchSingle()
      .map((row) => row.read<int>('pending_count'));
});

final guidedMediaStatusProvider =
    FutureProvider<GuidedMediaStatus>((ref) async {
  final ownerUserId = ref.watch(authProvider).user?.id;
  if (ownerUserId == null) {
    return const GuidedMediaStatus(
      acknowledged: 0,
      pending: 0,
      failed: 0,
      deletionRequested: 0,
    );
  }
  return ref.watch(guidedSyncServiceProvider).mediaStatus(ownerUserId);
});

/// Long-lived listener: triggers sync on connectivity changes.
/// Started by main.dart via `ref.read(syncTriggerProvider)`.
final syncTriggerProvider = Provider<StreamSubscription>((ref) {
  final svc = ref.watch(syncCoordinatorProvider);
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
