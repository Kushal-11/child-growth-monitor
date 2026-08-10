import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../providers/database_provider.dart';
import '../../../providers/assessment_service_provider.dart';
import '../domain/ar_scan_models.dart';
import '../repositories/ar_scan_repository.dart';
import '../services/ar_scan_platform.dart';

class ArScanProcessedResult {
  const ArScanProcessedResult({
    this.estimatedWeightKg,
    this.weightRangeLowerKg,
    this.weightRangeUpperKg,
    this.weightSource,
  });

  final double? estimatedWeightKg;
  final double? weightRangeLowerKg;
  final double? weightRangeUpperKg;
  final String? weightSource;
}

typedef ArScanPostProcessor =
    Future<ArScanProcessedResult?> Function({
      required int ownerUserId,
      required String visitUuid,
    });

class ArScanState {
  const ArScanState({
    this.loading = true,
    this.scanning = false,
    this.capability,
    this.result,
    this.processedResult,
    this.error,
  });
  final bool loading;
  final bool scanning;
  final ArScanCapability? capability;
  final FullArScanResult? result;
  final ArScanProcessedResult? processedResult;
  final String? error;

  bool get useFallback => !loading && capability?.shouldOfferFullScan != true;
}

class ArScanNotifier extends StateNotifier<ArScanState> {
  ArScanNotifier({
    required ArScanPlatform platform,
    required ArScanRepository repository,
    required ArScanPostProcessor postProcessor,
  }) : _platform = platform,
       _repository = repository,
       _postProcessor = postProcessor,
       super(const ArScanState());
  final ArScanPlatform _platform;
  final ArScanRepository _repository;
  final ArScanPostProcessor _postProcessor;

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
      ArScanProcessedResult? processedResult;
      String? processingError;
      if (visit.entryMethod == 'assessment') {
        try {
          processedResult = await _postProcessor(
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
          );
        } catch (error) {
          processingError =
              'Depth height and MUAC were saved, but the AR weight estimate '
              'could not be completed. ($error)';
        }
      }
      if (!mounted) return;
      state = ArScanState(
        loading: false,
        capability: state.capability,
        result: result,
        processedResult: processedResult,
        error: processingError,
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
final arScanPostProcessorProvider = Provider<ArScanPostProcessor>((ref) {
  return ({required int ownerUserId, required String visitUuid}) async {
    final workflow = await ref.read(cameraScreeningWorkflowProvider.future);
    final result = await workflow.processAssessment(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    return ArScanProcessedResult(
      estimatedWeightKg: result.estimatedWeightKg,
      weightRangeLowerKg: result.weightRangeLowerKg,
      weightRangeUpperKg: result.weightRangeUpperKg,
      weightSource: result.weightSource,
    );
  };
});
final arScanProvider =
    StateNotifierProvider.autoDispose<ArScanNotifier, ArScanState>((ref) {
      final notifier = ArScanNotifier(
        platform: ref.watch(arScanPlatformProvider),
        repository: ref.watch(arScanRepositoryProvider),
        postProcessor: ref.watch(arScanPostProcessorProvider),
      );
      unawaited(notifier.check());
      return notifier;
    });
