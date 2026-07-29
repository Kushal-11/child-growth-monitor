/// Canonical guided-capture values shared with the backend contract.
library;

enum CaptureState {
  draftCapture('draft_capture'),
  incompleteCapture('incomplete_capture'),
  processing('processing'),
  estimatedReport('estimated_report'),
  processingFailed('processing_failed'),
  measuredReport('measured_report');

  const CaptureState(this.wireValue);
  final String wireValue;

  static CaptureState fromWire(String value) => values.firstWhere(
        (candidate) => candidate.wireValue == value,
        orElse: () => throw FormatException('Unknown capture state: $value'),
      );
}

enum CaptureAssetRole {
  front('front'),
  side('side'),
  back('back'),
  armFront('arm_front'),
  armSide('arm_side');

  const CaptureAssetRole(this.wireValue);
  final String wireValue;

  static const List<CaptureAssetRole> requiredRoles = [front, side];

  static CaptureAssetRole fromWire(String value) => values.firstWhere(
        (candidate) => candidate.wireValue == value,
        orElse: () => throw FormatException('Unknown capture role: $value'),
      );
}

enum MeasurementMode {
  standingHeight('standing_height'),
  recumbentLength('recumbent_length');

  const MeasurementMode(this.wireValue);
  final String wireValue;

  static MeasurementMode fromWire(String value) => values.firstWhere(
        (candidate) => candidate.wireValue == value,
        orElse: () => throw FormatException('Unknown measurement mode: $value'),
      );
}

enum OedemaStatus {
  yes('yes'),
  no('no'),
  notChecked('not_checked');

  const OedemaStatus(this.wireValue);
  final String wireValue;

  static OedemaStatus fromWire(String value) => values.firstWhere(
        (candidate) => candidate.wireValue == value,
        orElse: () => throw FormatException('Unknown oedema value: $value'),
      );
}

const Map<CaptureState, Set<CaptureState>> allowedCaptureStateTransitions = {
  CaptureState.draftCapture: {
    CaptureState.incompleteCapture,
    CaptureState.processing,
  },
  CaptureState.incompleteCapture: {CaptureState.draftCapture},
  CaptureState.processing: {
    CaptureState.estimatedReport,
    CaptureState.processingFailed,
  },
  CaptureState.estimatedReport: {
    CaptureState.processing,
    CaptureState.measuredReport,
  },
  CaptureState.processingFailed: {CaptureState.processing},
  CaptureState.measuredReport: {CaptureState.measuredReport},
};

bool canTransitionCaptureState(CaptureState current, CaptureState target) =>
    allowedCaptureStateTransitions[current]?.contains(target) ?? false;

void requireCaptureStateTransition(CaptureState current, CaptureState target) {
  if (!canTransitionCaptureState(current, target)) {
    throw StateError(
      'Invalid capture-state transition: '
      '${current.wireValue} -> ${target.wireValue}',
    );
  }
}
