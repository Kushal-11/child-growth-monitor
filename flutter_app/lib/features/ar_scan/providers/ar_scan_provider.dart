import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../providers/database_provider.dart';
import '../domain/ar_scan_models.dart';
import '../repositories/ar_scan_repository.dart';
import '../services/ar_scan_platform.dart';

class ArScanState {
  const ArScanState({
    this.loading = true,
    this.scanning = false,
    this.capability,
    this.result,
    this.error,
  });
  final bool loading;
  final bool scanning;
  final ArScanCapability? capability;
  final FullArScanResult? result;
  final String? error;

  bool get useFallback => !loading && capability?.shouldOfferFullScan != true;
}

class ArScanNotifier extends StateNotifier<ArScanState> {
  ArScanNotifier({
    required ArScanPlatform platform,
    required ArScanRepository repository,
  })  : _platform = platform,
        _repository = repository,
        super(const ArScanState());
  final ArScanPlatform _platform;
  final ArScanRepository _repository;

  Future<void> check() async {
    state = const ArScanState();
    final capability = await _platform.checkCapability();
    if (!mounted) return;
    state = ArScanState(loading: false, capability: capability);
  }

  Future<void> scanAndSave({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    if (state.capability?.shouldOfferFullScan != true || state.scanning) {
      return;
    }
    state = ArScanState(
      loading: false,
      scanning: true,
      capability: state.capability,
    );
    try {
      final visit = await _repository.getVisitContext(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
      );
      final result = await _platform.startFullScan(
        ageMonths: visit.ageMonths,
        sex: visit.sex,
      );
      if (!mounted) return;
      if (result == null) {
        state = ArScanState(loading: false, capability: state.capability);
        return;
      }
      await _repository.saveExperimentalResult(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        result: result,
      );
      if (!mounted) return;
      state = ArScanState(
        loading: false,
        capability: state.capability,
        result: result,
      );
    } catch (error) {
      if (!mounted) return;
      state = ArScanState(
        loading: false,
        capability: state.capability,
        error: 'Depth scan unavailable. Continue with guided photos. ($error)',
      );
    }
  }
}

final arScanPlatformProvider = Provider<ArScanPlatform>(
  (_) => const MethodChannelArScanPlatform(),
);
final arScanRepositoryProvider = Provider<ArScanRepository>(
  (ref) => DriftArScanRepository(ref.watch(databaseProvider)),
);
final arScanProvider =
    StateNotifierProvider.autoDispose<ArScanNotifier, ArScanState>(
  (ref) {
    final notifier = ArScanNotifier(
      platform: ref.watch(arScanPlatformProvider),
      repository: ref.watch(arScanRepositoryProvider),
    );
    unawaited(notifier.check());
    return notifier;
  },
);
