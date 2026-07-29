import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';

import '../../../constants/config.dart';
import '../../../database/daos/measured_detail_revision_dao.dart';
import '../../../database/database.dart';
import '../../../services/nutrition_service.dart';
import '../../../services/poshan_setu_service.dart';
import '../../../services/who_data_service.dart';
import '../../guided_capture/domain/capture_models.dart';
import '../domain/measured_details.dart';

abstract interface class MeasuredReportGateway {
  Future<MeasuredVisitContext> loadContext({
    required int ownerUserId,
    required String visitUuid,
  });

  Future<void> save({
    required int ownerUserId,
    required String visitUuid,
    required int editorUserId,
    required MeasuredDetails details,
  });
}

class MeasuredReportService implements MeasuredReportGateway {
  MeasuredReportService({
    required AppDatabase database,
    required MeasuredDetailRevisionDao revisionDao,
    required WhoDataService who,
    String Function()? newUuid,
    DateTime Function()? now,
  })  : _database = database,
        _revisionDao = revisionDao,
        _nutrition = NutritionService(who),
        _newUuid = newUuid ?? const Uuid().v4,
        _now = now ?? DateTime.now;

  final AppDatabase _database;
  final MeasuredDetailRevisionDao _revisionDao;
  final NutritionService _nutrition;
  final String Function() _newUuid;
  final DateTime Function() _now;

  @override
  Future<MeasuredVisitContext> loadContext({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    final visit = await (_database.select(_database.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId) &
                row.entryMethod.equals('guided_capture'),
          ))
        .getSingleOrNull();
    if (visit == null) {
      throw StateError('Owner-scoped guided visit was not found');
    }
    final child = await (_database.select(_database.children)
          ..where(
            (row) =>
                row.id.equals(visit.childId) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingleOrNull();
    if (child == null) {
      throw StateError('Owner-scoped child was not found');
    }
    final dateOfBirth = DateTime.parse(child.dateOfBirth);
    return MeasuredVisitContext(
      visitUuid: visitUuid,
      ownerUserId: ownerUserId,
      childId: child.id,
      visitDate: _dateOnly(visit.visitDate),
      ageMonths: visit.ageMonths,
      completedAgeMonths: _completedMonths(
        dateOfBirth,
        visit.visitDate,
      ),
      sex: child.sex,
    );
  }

  MeasuredReport calculate({
    required MeasuredVisitContext context,
    required MeasuredDetails details,
  }) {
    final height = details.heightCm;
    final weight = details.weightKg;
    final muac = details.muacCm;
    final haz = height == null
        ? null
        : _finiteOrNull(
            _nutrition.computeHaz(
              context.sex,
              context.completedAgeMonths,
              height,
            ),
          );
    final whz = height == null || weight == null
        ? null
        : _finiteOrNull(
            _nutrition.computeWhz(
              context.sex,
              context.ageMonths,
              height,
              weight,
            ),
          );
    final hazStatus = haz == null ? null : classifyHaz(haz);
    final whzStatus = whz == null ? null : classifyWhz(whz);
    final muacEligible = context.ageMonths >= poshanMuacMinAgeMonths &&
        context.ageMonths < poshanMuacMaxAgeMonths;
    final muacStatus = muac == null ? null : classifyMuac(muac, muacEligible);

    final acuteComponents = <(String, String)>[
      if (whzStatus != null)
        (
          'whz',
          whzStatus == 'SAM' || whzStatus == 'MAM' ? whzStatus : 'NORMAL',
        ),
      if (muacStatus != null) ('muac', muacStatus),
      if (details.oedema == OedemaStatus.yes) ('oedema', 'SAM'),
    ];
    final samTriggers = acuteComponents
        .where((component) => component.$2 == 'SAM')
        .map((component) => component.$1)
        .toList(growable: false);
    final mamTriggers = acuteComponents
        .where((component) => component.$2 == 'MAM')
        .map((component) => component.$1)
        .toList(growable: false);
    final String acuteStatus;
    final List<String> acuteTriggers;
    if (samTriggers.isNotEmpty) {
      acuteStatus = 'SAM';
      acuteTriggers = samTriggers;
    } else if (mamTriggers.isNotEmpty) {
      acuteStatus = 'MAM';
      acuteTriggers = mamTriggers;
    } else if (acuteComponents.isNotEmpty) {
      acuteStatus = 'NORMAL';
      acuteTriggers = acuteComponents
          .map((component) => component.$1)
          .toList(growable: false);
    } else {
      acuteStatus = 'UNKNOWN';
      acuteTriggers = const [];
    }
    final missing = <String>[
      if (height == null) 'height or length not measured',
      if (weight == null) 'weight not measured',
      if (height != null && weight != null && whz == null)
        'WHZ unavailable for the measured length or height',
      if (muac == null) 'tape MUAC not measured',
      if (muac != null && !muacEligible)
        'tape MUAC ineligible outside 6-59 months',
      if (details.oedema == OedemaStatus.notChecked) 'oedema not checked',
    ];
    final poshan = const PoshanSetuService().classify(
      sex: context.sex,
      ageMonths: context.ageMonths,
      heightCm: height,
      heightSource: height == null ? 'unavailable' : 'manual',
      weightKg: weight,
      weightSource: weight == null ? 'unavailable' : 'manual',
      muacCm: muac,
      muacSource: muac == null ? 'unavailable' : 'tape',
    );

    return MeasuredReport(
      heightCm: height,
      weightKg: weight,
      muacCm: muac,
      measurementMode: details.measurementMode,
      oedema: details.oedema,
      hazZscore: haz,
      hazStatus: hazStatus,
      whzZscore: whz,
      whzStatus: whzStatus,
      muacStatus: muacStatus,
      muacEligible: muacEligible,
      whoAcuteStatus: acuteStatus,
      whoAcuteTriggeredBy: acuteTriggers,
      whoAcuteRationale:
          'WHO acute malnutrition uses eligible measured WHZ, tape MUAC, '
          'and oedema; status $acuteStatus; triggers $acuteTriggers; '
          'missing $missing.',
      poshan: poshan,
    );
  }

  @override
  Future<void> save({
    required int ownerUserId,
    required String visitUuid,
    required int editorUserId,
    required MeasuredDetails details,
  }) async {
    if (editorUserId != ownerUserId) {
      throw ArgumentError(
        'Measured-detail editor must match the visit owner',
      );
    }
    final context = await loadContext(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    if (!_sameDate(context.visitDate, details.measurementDate)) {
      throw ArgumentError(
        'The measurement date must match the locked visit date',
      );
    }
    final visit = await (_database.select(_database.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId),
          ))
        .getSingle();
    final existing = await (_database.select(_database.measurements)
          ..where((row) => row.visitId.equals(visit.id)))
        .getSingleOrNull();
    final merged = MeasuredDetails(
      measurementDate: context.visitDate,
      measuredAt: details.measuredAt,
      measurementMode: details.measurementMode,
      oedema: details.oedema,
      heightCm: details.heightCm ?? existing?.manualHeightCm,
      weightKg: details.weightKg ?? existing?.manualWeightKg,
      muacCm: details.muacCm ?? existing?.muacCm,
      notes: details.notes ?? existing?.measuredNotes,
      reason: details.reason,
    );
    final report = calculate(context: context, details: merged);
    final before = _snapshot(existing);
    final after = _reportSnapshot(
      report,
      measuredAt: merged.measuredAt,
      notes: merged.notes,
    );
    final revisionUuid = _newUuid();
    final createdAt = _now();
    final payload = {
      'revision_uuid': revisionUuid,
      'visit_uuid': visitUuid,
      'before': before,
      'after': after,
      'editor_user_id': editorUserId,
      'created_at': createdAt.toIso8601String(),
      'reason': merged.reason,
    };

    await _revisionDao.saveMeasuredReport(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      revisionUuid: revisionUuid,
      beforeJson: jsonEncode(before),
      afterJson: jsonEncode(after),
      reason: merged.reason,
      payloadJson: jsonEncode(payload),
      measurement: MeasurementsCompanion(
        manualHeightCm: Value(report.heightCm),
        manualWeightKg: Value(report.weightKg),
        effectiveHeightCm: Value(report.heightCm),
        effectiveWeightKg: Value(report.weightKg),
        heightMethod: Value(report.heightCm == null ? 'unavailable' : 'manual'),
        weightMethod: Value(report.weightKg == null ? 'unavailable' : 'manual'),
        estimationMethod: const Value('manual'),
        bmi: Value(report.poshan.bmi),
        bmiStatus: Value(report.poshan.bmiStatus),
        hazZscore: Value(report.hazZscore),
        whzZscore: Value(report.whzZscore),
        hazStatus: Value(report.hazStatus),
        whzStatus: Value(report.whzStatus),
        muacCm: Value(report.muacCm),
        muacStatus: Value(report.muacStatus),
        muacMethod: Value(report.muacCm == null ? 'unavailable' : 'tape'),
        muacAgeInRange: Value(report.muacEligible),
        muacIsDirectMeasurement: Value(report.muacCm != null),
        combinedStatus: Value(report.whoAcuteStatus),
        combinedTriggeredBy: Value(jsonEncode(report.whoAcuteTriggeredBy)),
        combinedRationale: Value(report.whoAcuteRationale),
        combinedMethod: const Value('who_measured_whz_muac_oedema_v1'),
        poshanStatus: Value(report.poshan.finalStatus),
        poshanTriggeredBy: Value(jsonEncode(report.poshan.triggeredBy)),
        classificationMethod: Value(report.poshan.classificationMethod),
        classificationRationale: Value(report.poshan.rationale),
        poshanComplete: Value(report.poshan.complete),
        measurementMode: Value(report.measurementMode.wireValue),
        oedema: Value(report.oedema.wireValue),
        measuredAt: Value(merged.measuredAt),
        editorUserId: Value(editorUserId),
        measuredNotes: Value(merged.notes),
        whoAcuteStatus: Value(report.whoAcuteStatus),
        whoAcuteTriggeredBy: Value(jsonEncode(report.whoAcuteTriggeredBy)),
        whoAcuteRationale: Value(report.whoAcuteRationale),
      ),
    );
  }

  Map<String, Object?> _snapshot(Measurement? measurement) {
    if (measurement == null) return const {};
    return {
      'height_cm': measurement.manualHeightCm,
      'weight_kg': measurement.manualWeightKg,
      'muac_cm': measurement.muacCm,
      'measurement_mode': measurement.measurementMode,
      'oedema': measurement.oedema,
      'measured_at': measurement.measuredAt?.toIso8601String(),
      'notes': measurement.measuredNotes,
      'haz_zscore': measurement.hazZscore,
      'whz_zscore': measurement.whzZscore,
      'haz_status': measurement.hazStatus,
      'whz_status': measurement.whzStatus,
      'who_acute_status': measurement.whoAcuteStatus,
      'poshan_status': measurement.poshanStatus,
    };
  }

  Map<String, Object?> _reportSnapshot(
    MeasuredReport report, {
    required DateTime measuredAt,
    required String? notes,
  }) {
    return {
      'height_cm': report.heightCm,
      'weight_kg': report.weightKg,
      'muac_cm': report.muacCm,
      'measurement_mode': report.measurementMode.wireValue,
      'oedema': report.oedema.wireValue,
      'measured_at': measuredAt.toIso8601String(),
      'notes': notes,
      'haz_zscore': report.hazZscore,
      'whz_zscore': report.whzZscore,
      'haz_status': report.hazStatus,
      'whz_status': report.whzStatus,
      'who_acute_status': report.whoAcuteStatus,
      'poshan_status': report.poshan.finalStatus,
    };
  }

  static DateTime _dateOnly(DateTime value) =>
      DateTime(value.year, value.month, value.day);

  static bool _sameDate(DateTime left, DateTime right) =>
      left.year == right.year &&
      left.month == right.month &&
      left.day == right.day;

  static int _completedMonths(DateTime dateOfBirth, DateTime asOf) {
    var months =
        (asOf.year - dateOfBirth.year) * 12 + asOf.month - dateOfBirth.month;
    final anniversary = _addMonths(dateOfBirth, months);
    if (asOf.isBefore(anniversary)) months -= 1;
    return months;
  }

  static DateTime _addMonths(DateTime value, int months) {
    final monthIndex = value.year * 12 + value.month - 1 + months;
    final year = monthIndex ~/ 12;
    final month = monthIndex % 12 + 1;
    final lastDay = DateTime(year, month + 1, 0).day;
    return DateTime(year, month, value.day > lastDay ? lastDay : value.day);
  }

  static double? _finiteOrNull(double? value) =>
      value != null && value.isFinite ? value : null;
}
