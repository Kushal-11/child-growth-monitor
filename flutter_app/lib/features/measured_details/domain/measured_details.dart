import '../../guided_capture/domain/capture_models.dart';
import '../../../services/poshan_setu_service.dart';

const double measuredHeightMinCm = 30;
const double measuredHeightMaxCm = 150;
const double measuredWeightMinKg = 0.5;
const double measuredWeightMaxKg = 50;
const double measuredMuacMinCm = 5;
const double measuredMuacMaxCm = 30;

class MeasuredDetails {
  MeasuredDetails({
    required this.measurementDate,
    required this.measuredAt,
    required this.measurementMode,
    required this.oedema,
    this.heightCm,
    this.weightKg,
    this.muacCm,
    this.notes,
    this.reason,
  }) {
    MeasuredDetailsValidators.requirePlausible(
      name: 'height_cm',
      value: heightCm,
      minimum: measuredHeightMinCm,
      maximum: measuredHeightMaxCm,
    );
    MeasuredDetailsValidators.requirePlausible(
      name: 'weight_kg',
      value: weightKg,
      minimum: measuredWeightMinKg,
      maximum: measuredWeightMaxKg,
    );
    MeasuredDetailsValidators.requirePlausible(
      name: 'muac_cm',
      value: muacCm,
      minimum: measuredMuacMinCm,
      maximum: measuredMuacMaxCm,
    );
    if (heightCm == null &&
        weightKg == null &&
        muacCm == null &&
        oedema == OedemaStatus.notChecked) {
      throw ArgumentError(
        'Enter at least one measured detail or record an oedema check',
      );
    }
    if ((notes?.length ?? 0) > 2000) {
      throw ArgumentError.value(
          notes, 'notes', 'must be 2000 characters or fewer');
    }
    if ((reason?.length ?? 0) > 500) {
      throw ArgumentError.value(
        reason,
        'reason',
        'must be 500 characters or fewer',
      );
    }
  }

  final DateTime measurementDate;
  final DateTime measuredAt;
  final MeasurementMode measurementMode;
  final OedemaStatus oedema;
  final double? heightCm;
  final double? weightKg;
  final double? muacCm;
  final String? notes;
  final String? reason;
}

abstract final class MeasuredDetailsValidators {
  static void requirePlausible({
    required String name,
    required double? value,
    required double minimum,
    required double maximum,
  }) {
    if (value != null &&
        (!value.isFinite || value < minimum || value > maximum)) {
      throw ArgumentError.value(
        value,
        name,
        'must be between $minimum and $maximum',
      );
    }
  }

  static String? optionalText(
    String? raw, {
    required String label,
    required double minimum,
    required double maximum,
  }) {
    final trimmed = raw?.trim() ?? '';
    if (trimmed.isEmpty) return null;
    final value = double.tryParse(trimmed);
    if (value == null || !value.isFinite) {
      return 'Enter a valid $label';
    }
    if (value < minimum || value > maximum) {
      return '$label must be between $minimum and $maximum';
    }
    return null;
  }

  static String? requiredText(
    String? raw, {
    required String label,
    required double minimum,
    required double maximum,
  }) {
    if (raw == null || raw.trim().isEmpty) return 'Required';
    return optionalText(
      raw,
      label: label,
      minimum: minimum,
      maximum: maximum,
    );
  }
}

class MeasuredVisitContext {
  const MeasuredVisitContext({
    required this.visitUuid,
    required this.ownerUserId,
    required this.childId,
    required this.visitDate,
    required this.ageMonths,
    required this.completedAgeMonths,
    required this.sex,
  });

  final String visitUuid;
  final int ownerUserId;
  final int childId;
  final DateTime visitDate;
  final double ageMonths;
  final int completedAgeMonths;
  final String sex;
}

class MeasuredReport {
  const MeasuredReport({
    required this.heightCm,
    required this.weightKg,
    required this.muacCm,
    required this.measurementMode,
    required this.oedema,
    required this.hazZscore,
    required this.hazStatus,
    required this.whzZscore,
    required this.whzStatus,
    required this.muacStatus,
    required this.muacEligible,
    required this.whoAcuteStatus,
    required this.whoAcuteTriggeredBy,
    required this.whoAcuteRationale,
    required this.poshan,
  });

  final double? heightCm;
  final double? weightKg;
  final double? muacCm;
  final MeasurementMode measurementMode;
  final OedemaStatus oedema;
  final double? hazZscore;
  final String? hazStatus;
  final double? whzZscore;
  final String? whzStatus;
  final String? muacStatus;
  final bool muacEligible;
  final String whoAcuteStatus;
  final List<String> whoAcuteTriggeredBy;
  final String whoAcuteRationale;
  final PoshanSetuResult poshan;
}
