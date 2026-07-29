import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../providers/assessment_service_provider.dart';
import '../../../providers/database_provider.dart';
import '../../reports/providers/visit_report_provider.dart';
import '../domain/measured_details.dart';
import '../services/measured_report_service.dart';

class MeasuredVisitRequest {
  const MeasuredVisitRequest({
    required this.visitUuid,
    required this.ownerUserId,
  });

  final String visitUuid;
  final int ownerUserId;

  @override
  bool operator ==(Object other) =>
      other is MeasuredVisitRequest &&
      other.visitUuid == visitUuid &&
      other.ownerUserId == ownerUserId;

  @override
  int get hashCode => Object.hash(visitUuid, ownerUserId);
}

class MeasuredDetailsState {
  const MeasuredDetailsState({
    this.saving = false,
    this.saved = false,
    this.errorMessage,
  });

  final bool saving;
  final bool saved;
  final String? errorMessage;
}

class MeasuredDetailsNotifier extends StateNotifier<MeasuredDetailsState> {
  MeasuredDetailsNotifier(this._ref, this._request)
      : super(const MeasuredDetailsState());

  final Ref _ref;
  final MeasuredVisitRequest _request;

  Future<void> save({
    required int editorUserId,
    required MeasuredDetails details,
  }) async {
    state = const MeasuredDetailsState(saving: true);
    try {
      final gateway = await _ref.read(measuredReportGatewayProvider.future);
      await gateway.save(
        ownerUserId: _request.ownerUserId,
        visitUuid: _request.visitUuid,
        editorUserId: editorUserId,
        details: details,
      );
      _ref.invalidate(
        visitReportProvider(
          VisitReportRequest(
            visitUuid: _request.visitUuid,
            ownerUserId: _request.ownerUserId,
          ),
        ),
      );
      state = const MeasuredDetailsState(saved: true);
    } catch (error) {
      state = MeasuredDetailsState(errorMessage: error.toString());
      rethrow;
    }
  }
}

final measuredReportGatewayProvider =
    FutureProvider<MeasuredReportGateway>((ref) async {
  final who = await ref.watch(whoDataServiceProvider.future);
  return MeasuredReportService(
    database: ref.watch(databaseProvider),
    revisionDao: ref.watch(measuredDetailRevisionDaoProvider),
    who: who,
  );
});

final measuredVisitContextProvider =
    FutureProvider.family<MeasuredVisitContext, MeasuredVisitRequest>(
  (ref, request) async {
    final gateway = await ref.watch(measuredReportGatewayProvider.future);
    return gateway.loadContext(
      ownerUserId: request.ownerUserId,
      visitUuid: request.visitUuid,
    );
  },
);

final measuredDetailsProvider = StateNotifierProvider.autoDispose.family<
    MeasuredDetailsNotifier, MeasuredDetailsState, MeasuredVisitRequest>(
  (ref, request) => MeasuredDetailsNotifier(ref, request),
);
