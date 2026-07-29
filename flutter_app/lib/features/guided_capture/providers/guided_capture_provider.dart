import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';

import '../../../providers/database_provider.dart';
import '../domain/capture_models.dart';
import '../repositories/guided_capture_repository.dart';
import '../services/guided_camera_controller.dart';

const String guidedCaptureConsentVersion = 'guided_capture_consent_v1';
const int maxRequiredRoleFailures = 3;
const Object _notSet = Object();

enum GuidedCapturePhase {
  idle,
  loading,
  consent,
  capture,
  review,
  incomplete,
  error,
}

class GuidedCaptureState {
  const GuidedCaptureState({
    this.phase = GuidedCapturePhase.idle,
    this.child,
    this.ownerUserId,
    this.visitUuid,
    this.captureState,
    this.currentRole,
    this.acceptedFrames = const {},
    this.skippedRoles = const {},
    this.roleFailures = const {},
    this.consentRecorded = false,
    this.saving = false,
    this.errorMessage,
  });

  final GuidedCapturePhase phase;
  final GuidedCaptureChild? child;
  final int? ownerUserId;
  final String? visitUuid;
  final CaptureState? captureState;
  final CaptureAssetRole? currentRole;
  final Map<CaptureAssetRole, List<GuidedRetainedFrame>> acceptedFrames;
  final Set<CaptureAssetRole> skippedRoles;
  final Map<CaptureAssetRole, int> roleFailures;
  final bool consentRecorded;
  final bool saving;
  final String? errorMessage;

  bool get requiredRolesComplete => CaptureAssetRole.requiredRoles.every(
        (role) => acceptedFrames[role]?.isNotEmpty ?? false,
      );

  bool get canReviewRequired =>
      phase == GuidedCapturePhase.capture && requiredRolesComplete;

  GuidedCaptureState copyWith({
    GuidedCapturePhase? phase,
    Object? child = _notSet,
    Object? ownerUserId = _notSet,
    Object? visitUuid = _notSet,
    Object? captureState = _notSet,
    Object? currentRole = _notSet,
    Map<CaptureAssetRole, List<GuidedRetainedFrame>>? acceptedFrames,
    Set<CaptureAssetRole>? skippedRoles,
    Map<CaptureAssetRole, int>? roleFailures,
    bool? consentRecorded,
    bool? saving,
    Object? errorMessage = _notSet,
  }) {
    return GuidedCaptureState(
      phase: phase ?? this.phase,
      child:
          identical(child, _notSet) ? this.child : child as GuidedCaptureChild?,
      ownerUserId: identical(ownerUserId, _notSet)
          ? this.ownerUserId
          : ownerUserId as int?,
      visitUuid:
          identical(visitUuid, _notSet) ? this.visitUuid : visitUuid as String?,
      captureState: identical(captureState, _notSet)
          ? this.captureState
          : captureState as CaptureState?,
      currentRole: identical(currentRole, _notSet)
          ? this.currentRole
          : currentRole as CaptureAssetRole?,
      acceptedFrames: acceptedFrames ?? this.acceptedFrames,
      skippedRoles: skippedRoles ?? this.skippedRoles,
      roleFailures: roleFailures ?? this.roleFailures,
      consentRecorded: consentRecorded ?? this.consentRecorded,
      saving: saving ?? this.saving,
      errorMessage: identical(errorMessage, _notSet)
          ? this.errorMessage
          : errorMessage as String?,
    );
  }

  Map<String, Object?> toDebugJson() => {
        'phase': phase.name,
        'child_id': child?.id,
        'owner_user_id': ownerUserId,
        'visit_uuid': visitUuid,
        'capture_state': captureState?.wireValue,
        'current_role': currentRole?.wireValue,
        'accepted_roles': acceptedFrames.keys
            .map((role) => role.wireValue)
            .toList(growable: false),
        'skipped_roles':
            skippedRoles.map((role) => role.wireValue).toList(growable: false),
        'consent_recorded': consentRecorded,
      };
}

class GuidedCaptureNotifier extends StateNotifier<GuidedCaptureState> {
  GuidedCaptureNotifier({
    required GuidedCaptureRepository repository,
    String Function()? newUuid,
    DateTime Function()? now,
  })  : _repository = repository,
        _newUuid = newUuid ?? const Uuid().v4,
        _now = now ?? DateTime.now,
        super(const GuidedCaptureState());

  final GuidedCaptureRepository _repository;
  final String Function() _newUuid;
  final DateTime Function() _now;

  Future<void> initializeNew({
    required int childId,
    required int ownerUserId,
  }) async {
    if (state.phase == GuidedCapturePhase.consent &&
        state.child?.id == childId &&
        state.ownerUserId == ownerUserId) {
      return;
    }
    state = const GuidedCaptureState(phase: GuidedCapturePhase.loading);
    try {
      final child = await _repository.getOwnerChild(
        childId: childId,
        ownerUserId: ownerUserId,
      );
      if (child == null) {
        throw StateError('Owner-scoped child was not found');
      }
      state = GuidedCaptureState(
        phase: GuidedCapturePhase.consent,
        child: child,
        ownerUserId: ownerUserId,
      );
    } catch (error) {
      state = GuidedCaptureState(
        phase: GuidedCapturePhase.error,
        ownerUserId: ownerUserId,
        errorMessage: error.toString(),
      );
      rethrow;
    }
  }

  Future<String> acceptConsent({
    required String operatorIdentifier,
    required String deviceMetadataJson,
  }) async {
    final child = state.child;
    if (state.phase != GuidedCapturePhase.consent ||
        child == null ||
        state.ownerUserId == null) {
      throw StateError('Consent requires an initialized owner-scoped child');
    }
    state = state.copyWith(saving: true, errorMessage: null);
    final visitUuid = _newUuid();
    final timestamp = _now();
    try {
      final snapshot = await _repository.createDraft(
        child: child,
        visitUuid: visitUuid,
        visitDate: DateTime(timestamp.year, timestamp.month, timestamp.day),
        deviceMetadataJson: deviceMetadataJson,
        consentVersion: guidedCaptureConsentVersion,
        consentTimestamp: timestamp,
        consentOperatorIdentifier: operatorIdentifier,
      );
      state = GuidedCaptureState(
        phase: GuidedCapturePhase.capture,
        child: snapshot.child,
        ownerUserId: snapshot.child.ownerUserId,
        visitUuid: snapshot.visitUuid,
        captureState: snapshot.captureState,
        currentRole: CaptureAssetRole.front,
        acceptedFrames: snapshot.acceptedFrames,
        consentRecorded: true,
      );
      return visitUuid;
    } catch (error) {
      state = state.copyWith(
        saving: false,
        errorMessage: error.toString(),
      );
      rethrow;
    }
  }

  Future<void> resume({
    required String visitUuid,
    required int ownerUserId,
  }) async {
    if (state.visitUuid == visitUuid &&
        state.ownerUserId == ownerUserId &&
        state.phase != GuidedCapturePhase.idle) {
      return;
    }
    state = const GuidedCaptureState(phase: GuidedCapturePhase.loading);
    try {
      final snapshot = await _repository.loadDraft(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
      );
      if (snapshot == null) {
        throw StateError('Owner-scoped guided visit was not found');
      }
      final missingRequired = CaptureAssetRole.requiredRoles
          .where(
            (role) => snapshot.acceptedFrames[role]?.isNotEmpty != true,
          )
          .firstOrNull;
      state = GuidedCaptureState(
        phase: missingRequired == null
            ? GuidedCapturePhase.review
            : GuidedCapturePhase.capture,
        child: snapshot.child,
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        captureState: snapshot.captureState,
        currentRole: missingRequired,
        acceptedFrames: snapshot.acceptedFrames,
        consentRecorded: true,
      );
    } catch (error) {
      state = GuidedCaptureState(
        phase: GuidedCapturePhase.error,
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        errorMessage: error.toString(),
      );
      rethrow;
    }
  }

  Future<void> acceptFrames(List<GuidedRetainedFrame> frames) async {
    final ownerUserId = state.ownerUserId;
    final visitUuid = state.visitUuid;
    final currentRole = state.currentRole;
    if (state.phase != GuidedCapturePhase.capture ||
        ownerUserId == null ||
        visitUuid == null ||
        currentRole == null ||
        frames.isEmpty ||
        frames.any((frame) => frame.role != currentRole)) {
      throw StateError('Frames do not match the active capture role');
    }

    state = state.copyWith(saving: true, errorMessage: null);
    try {
      await _repository.saveAcceptedFrames(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        frames: frames,
      );
      final accepted = {
        ...state.acceptedFrames,
        currentRole: List<GuidedRetainedFrame>.unmodifiable(frames),
      };
      final nextRole = _nextRole(currentRole);
      state = state.copyWith(
        phase: nextRole == null
            ? GuidedCapturePhase.review
            : GuidedCapturePhase.capture,
        currentRole: nextRole,
        acceptedFrames: accepted,
        saving: false,
      );
    } catch (error) {
      state = state.copyWith(
        saving: false,
        errorMessage: error.toString(),
      );
      rethrow;
    }
  }

  Future<void> skipCurrentRole() async {
    final role = state.currentRole;
    if (state.phase != GuidedCapturePhase.capture || role == null) {
      throw StateError('No capture role is active');
    }
    if (CaptureAssetRole.requiredRoles.contains(role)) {
      throw StateError('${role.wireValue} is required');
    }
    final nextRole = _nextRole(role);
    state = state.copyWith(
      phase: nextRole == null
          ? GuidedCapturePhase.review
          : GuidedCapturePhase.capture,
      currentRole: nextRole,
      skippedRoles: {...state.skippedRoles, role},
    );
  }

  void reviewRequiredPhotos() {
    if (!state.requiredRolesComplete) {
      throw StateError('Front and side captures are required');
    }
    state = state.copyWith(
      phase: GuidedCapturePhase.review,
      currentRole: null,
    );
  }

  Future<void> recordRoleFailure(CaptureAssetRole role) async {
    if (state.phase != GuidedCapturePhase.capture ||
        state.currentRole != role) {
      throw StateError('Failure does not match the active capture role');
    }
    final failures = {
      ...state.roleFailures,
      role: (state.roleFailures[role] ?? 0) + 1,
    };
    state = state.copyWith(roleFailures: failures);
    if (CaptureAssetRole.requiredRoles.contains(role) &&
        failures[role]! >= maxRequiredRoleFailures) {
      final ownerUserId = state.ownerUserId!;
      final visitUuid = state.visitUuid!;
      await _repository.markIncomplete(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
      );
      state = state.copyWith(
        phase: GuidedCapturePhase.incomplete,
        captureState: CaptureState.incompleteCapture,
        currentRole: null,
      );
    }
  }

  void declineConsent() {
    if (state.phase != GuidedCapturePhase.consent) return;
    state = const GuidedCaptureState();
  }

  CaptureAssetRole? _nextRole(CaptureAssetRole role) {
    const ordered = [
      CaptureAssetRole.front,
      CaptureAssetRole.side,
      CaptureAssetRole.back,
      CaptureAssetRole.armFront,
      CaptureAssetRole.armSide,
    ];
    final index = ordered.indexOf(role);
    return index < 0 || index == ordered.length - 1 ? null : ordered[index + 1];
  }
}

final guidedCaptureRepositoryProvider =
    Provider<GuidedCaptureRepository>((ref) {
  return DriftGuidedCaptureRepository(
    database: ref.watch(databaseProvider),
    visitDao: ref.watch(guidedVisitDaoProvider),
    captureAssetDao: ref.watch(captureAssetDaoProvider),
  );
});

final guidedCaptureProvider =
    StateNotifierProvider<GuidedCaptureNotifier, GuidedCaptureState>((ref) {
  return GuidedCaptureNotifier(
    repository: ref.watch(guidedCaptureRepositoryProvider),
  );
});
