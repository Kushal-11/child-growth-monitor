import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../providers/auth_provider.dart';
import '../../guided_capture/domain/capture_models.dart';
import '../providers/visit_report_provider.dart';
import '../widgets/estimate_comparison_view.dart';
import '../widgets/estimated_report_view.dart';
import '../widgets/measured_report_view.dart';

class VisitReportScreen extends ConsumerStatefulWidget {
  const VisitReportScreen({
    super.key,
    required this.visitUuid,
    this.ownerUserId,
  });

  final String visitUuid;
  final int? ownerUserId;

  @override
  ConsumerState<VisitReportScreen> createState() => _VisitReportScreenState();
}

class _VisitReportScreenState extends ConsumerState<VisitReportScreen> {
  bool _processing = false;
  bool _automaticProcessingRequested = false;
  String? _localError;

  int? get _ownerUserId =>
      widget.ownerUserId ?? ref.read(authProvider).user?.id;

  VisitReportRequest? get _request {
    final ownerUserId = _ownerUserId;
    return ownerUserId == null
        ? null
        : VisitReportRequest(
            visitUuid: widget.visitUuid,
            ownerUserId: ownerUserId,
          );
  }

  Future<void> _process() async {
    final request = _request;
    if (request == null || _processing) return;
    setState(() {
      _processing = true;
      _localError = null;
    });
    try {
      await ref.read(cameraScreeningProcessorProvider).process(
            ownerUserId: request.ownerUserId,
            visitUuid: request.visitUuid,
          );
      ref.invalidate(visitReportProvider(request));
      await ref.read(visitReportProvider(request).future);
    } catch (error) {
      ref.invalidate(visitReportProvider(request));
      if (mounted) setState(() => _localError = error.toString());
    } finally {
      if (mounted) setState(() => _processing = false);
    }
  }

  void _scheduleAutomaticProcessing() {
    if (_automaticProcessingRequested) return;
    _automaticProcessingRequested = true;
    WidgetsBinding.instance.addPostFrameCallback((_) => _process());
  }

  void _addMeasuredDetails(DateTime visitDate) {
    final date = visitDate.toIso8601String().substring(0, 10);
    context.go(
      '/visits/${widget.visitUuid}/measured-details?visitDate=$date',
    );
  }

  @override
  Widget build(BuildContext context) {
    final request = _request;
    if (request == null) {
      return const Scaffold(
        body: Center(child: Text('An authenticated operator is required')),
      );
    }
    final report = ref.watch(visitReportProvider(request));
    return Scaffold(
      appBar: AppBar(title: const Text('Visit report')),
      body: report.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => _ErrorState(
          message: _localError ?? error.toString(),
          onRetry: () => ref.invalidate(visitReportProvider(request)),
        ),
        data: (snapshot) {
          if (snapshot.captureState == CaptureState.draftCapture) {
            _scheduleAutomaticProcessing();
            return const _ProcessingState();
          }
          if (snapshot.captureState == CaptureState.processing) {
            return const _ProcessingState();
          }
          if (snapshot.captureState == CaptureState.processingFailed) {
            return _FailureState(
              acceptedAssetCount: snapshot.acceptedAssetCount,
              retrying: _processing,
              error: _localError,
              onRetry: _process,
            );
          }
          final measured = snapshot.measuredReport;
          if (snapshot.captureState == CaptureState.measuredReport) {
            if (measured == null) {
              return _ErrorState(
                message:
                    'No saved measurement-based report was found for this visit.',
                onRetry: () => ref.invalidate(visitReportProvider(request)),
              );
            }
            return SingleChildScrollView(
              child: Column(
                children: [
                  MeasuredReportView(
                    report: measured,
                    visitDate: snapshot.visitDate,
                    onEditMeasuredDetails: () =>
                        _addMeasuredDetails(snapshot.visitDate),
                  ),
                  if (snapshot.latestCameraResult case final estimate?)
                    EstimateComparisonView(
                      estimate: estimate,
                      measured: measured,
                      authorized: true,
                    ),
                ],
              ),
            );
          }
          final result = snapshot.latestCameraResult;
          if (result == null) {
            return _ErrorState(
              message: 'No saved camera result was found for this visit.',
              onRetry: () => ref.invalidate(visitReportProvider(request)),
            );
          }
          return EstimatedReportView(
            result: result,
            visitDate: snapshot.visitDate,
            onAddMeasuredDetails: () => _addMeasuredDetails(snapshot.visitDate),
          );
        },
      ),
    );
  }
}

class _ProcessingState extends StatelessWidget {
  const _ProcessingState();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 16),
          Text('Processing estimate'),
        ],
      ),
    );
  }
}

class _FailureState extends StatelessWidget {
  const _FailureState({
    required this.acceptedAssetCount,
    required this.retrying,
    required this.onRetry,
    this.error,
  });

  final int acceptedAssetCount;
  final bool retrying;
  final VoidCallback onRetry;
  final String? error;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, size: 48),
            const SizedBox(height: 12),
            Text(
              'Estimate failed — retry',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Text(
              '$acceptedAssetCount accepted photos remain saved. '
              'Retrying does not delete or replace them.',
              textAlign: TextAlign.center,
            ),
            if (error != null) ...[
              const SizedBox(height: 8),
              Text(error!, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: retrying ? null : onRetry,
              icon: const Icon(Icons.refresh),
              label: Text(retrying ? 'Retrying…' : 'Retry estimate'),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  const _ErrorState({
    required this.message,
    required this.onRetry,
  });

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 12),
            OutlinedButton(
              onPressed: onRetry,
              child: const Text('Reload report'),
            ),
          ],
        ),
      ),
    );
  }
}
