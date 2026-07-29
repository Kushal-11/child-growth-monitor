import 'package:child_growth_monitor_app/database/daos/guided_visit_dao.dart';
import 'package:child_growth_monitor_app/database/daos/measured_detail_revision_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/measured_details/domain/measured_details.dart';
import 'package:child_growth_monitor_app/features/measured_details/services/measured_report_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import '../../fixtures/who_test_data.dart';

void main() {
  late WhoDataService who;

  setUpAll(() async {
    who = WhoDataService();
    await loadWhoForTests(who);
  });

  group('MeasuredDetails validation', () {
    test('height and weight are independently optional and plausible', () {
      final heightOnly = MeasuredDetails(
        measurementDate: DateTime(2026, 7, 29),
        measuredAt: DateTime.utc(2026, 7, 29, 10),
        measurementMode: MeasurementMode.standingHeight,
        oedema: OedemaStatus.notChecked,
        heightCm: 83.58,
      );
      final weightOnly = MeasuredDetails(
        measurementDate: DateTime(2026, 7, 29),
        measuredAt: DateTime.utc(2026, 7, 29, 10),
        measurementMode: MeasurementMode.standingHeight,
        oedema: OedemaStatus.notChecked,
        weightKg: 12,
      );

      expect(heightOnly.weightKg, isNull);
      expect(weightOnly.heightCm, isNull);
      expect(
        () => MeasuredDetails(
          measurementDate: DateTime(2026, 7, 29),
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.notChecked,
          heightCm: double.nan,
        ),
        throwsArgumentError,
      );
      expect(
        () => MeasuredDetails(
          measurementDate: DateTime(2026, 7, 29),
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.notChecked,
          muacCm: 31,
        ),
        throwsArgumentError,
      );
    });

    test('all-empty submission is rejected', () {
      expect(
        () => MeasuredDetails(
          measurementDate: DateTime(2026, 7, 29),
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.notChecked,
        ),
        throwsArgumentError,
      );
    });
  });

  group('Measured report compatibility', () {
    late AppDatabase db;
    late MeasuredReportService service;

    setUp(() {
      db = AppDatabase.forTesting(NativeDatabase.memory());
      service = MeasuredReportService(
        database: db,
        revisionDao: MeasuredDetailRevisionDao(db),
        who: who,
        newUuid: () => '40000000-0000-0000-0000-000000000001',
        now: () => DateTime.utc(2026, 7, 29, 10),
      );
    });

    tearDown(() => db.close());

    final context = MeasuredVisitContext(
      visitUuid: '10000000-0000-0000-0000-000000000001',
      ownerUserId: 7,
      childId: 11,
      visitDate: _visitDate,
      ageMonths: 30,
      completedAgeMonths: 30,
      sex: 'F',
    );

    test('height-only real WHO result mirrors backend Task 4 case', () {
      final report = service.calculate(
        context: context,
        details: MeasuredDetails(
          measurementDate: _visitDate,
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.notChecked,
          heightCm: 83.58,
        ),
      );

      expect(report.hazZscore, closeTo(-2.01, 0.02));
      expect(report.whzZscore, isNull);
      expect(report.muacStatus, isNull);
      expect(report.whoAcuteStatus, 'UNKNOWN');
      expect(report.poshan.finalStatus, 'Indeterminate');
    });

    test('oedema independently triggers WHO SAM but not Poshan', () {
      final report = service.calculate(
        context: context,
        details: MeasuredDetails(
          measurementDate: _visitDate,
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.yes,
        ),
      );

      expect(report.whoAcuteStatus, 'SAM');
      expect(report.whoAcuteTriggeredBy, ['oedema']);
      expect(report.poshan.finalStatus, 'Indeterminate');
      expect(report.poshan.classificationMethod, 'poshan_setu_v1');
    });

    test('MUAC outside 6-59 months is stored but classification-ineligible',
        () {
      final youngContext = MeasuredVisitContext(
        visitUuid: '10000000-0000-0000-0000-000000000001',
        ownerUserId: 7,
        childId: 11,
        visitDate: _visitDate,
        ageMonths: 5.9,
        completedAgeMonths: 5,
        sex: 'F',
      );
      final report = service.calculate(
        context: youngContext,
        details: MeasuredDetails(
          measurementDate: _visitDate,
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.recumbentLength,
          oedema: OedemaStatus.notChecked,
          muacCm: 10,
        ),
      );

      expect(report.muacEligible, isFalse);
      expect(report.muacStatus, isNull);
      expect(report.whoAcuteStatus, 'UNKNOWN');
    });
  });

  group('MeasuredReportService persistence', () {
    late AppDatabase db;
    late GuidedVisitDao visitDao;
    late MeasuredReportService service;
    late int childId;
    const visitUuid = '10000000-0000-0000-0000-000000000001';

    setUp(() async {
      db = AppDatabase.forTesting(NativeDatabase.memory());
      visitDao = GuidedVisitDao(db);
      childId = await db.into(db.children).insert(
            ChildrenCompanion.insert(
              name: 'Child 001',
              dateOfBirth: '2024-01-29',
              sex: 'F',
              ownerUserId: const Value(7),
            ),
          );
      final visit = await visitDao.createDraft(
        childId: childId,
        ownerUserId: 7,
        localUuid: visitUuid,
        visitDate: _visitDate,
        ageMonths: 30,
        deviceMetadataJson: '{}',
        consentVersion: 'guided_capture_consent_v1',
        consentTimestamp: _visitDate,
        consentOperatorIdentifier: 'worker-7',
      );
      await (db.update(db.visits)..where((row) => row.id.equals(visit.id)))
          .write(
        const VisitsCompanion(captureState: Value('estimated_report')),
      );
      service = MeasuredReportService(
        database: db,
        revisionDao: MeasuredDetailRevisionDao(db),
        who: who,
        newUuid: _RevisionUuids().next,
        now: () => DateTime.utc(2026, 7, 29, 10),
      );
    });

    tearDown(() => db.close());

    test('partial saves merge and append immutable before-after revisions',
        () async {
      await service.save(
        ownerUserId: 7,
        visitUuid: visitUuid,
        editorUserId: 7,
        details: MeasuredDetails(
          measurementDate: _visitDate,
          measuredAt: DateTime.utc(2026, 7, 29, 10),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.notChecked,
          heightCm: 83.58,
        ),
      );
      await service.save(
        ownerUserId: 7,
        visitUuid: visitUuid,
        editorUserId: 7,
        details: MeasuredDetails(
          measurementDate: _visitDate,
          measuredAt: DateTime.utc(2026, 7, 29, 11),
          measurementMode: MeasurementMode.standingHeight,
          oedema: OedemaStatus.no,
          weightKg: 11,
        ),
      );

      final measurement = await db.select(db.measurements).getSingle();
      final revisions = await db.select(db.measuredDetailRevisions).get();
      final outbox = await db.select(db.syncOutbox).get();
      final visit = await visitDao.getByUuid(
        ownerUserId: 7,
        visitUuid: visitUuid,
      );

      expect(measurement.manualHeightCm, 83.58);
      expect(measurement.manualWeightKg, 11);
      expect(measurement.heightMethod, 'manual');
      expect(measurement.weightMethod, 'manual');
      expect(revisions, hasLength(2));
      expect(revisions[0].beforeJson, '{}');
      expect(revisions[1].beforeJson, contains('"height_cm":83.58'));
      expect(revisions[0].afterJson, isNot(revisions[1].afterJson));
      expect(visit!.captureState, 'measured_report');
      expect(
        outbox.singleWhere((entry) => entry.entityType == 'visit').payloadJson,
        contains('"capture_state":"measured_report"'),
      );
      expect(
        outbox.where((entry) => entry.entityType == 'measured_revision'),
        hasLength(2),
      );
    });

    test('visit-date mismatch leaves report and revision history unchanged',
        () async {
      await expectLater(
        service.save(
          ownerUserId: 7,
          visitUuid: visitUuid,
          editorUserId: 7,
          details: MeasuredDetails(
            measurementDate: DateTime(2026, 7, 28),
            measuredAt: DateTime.utc(2026, 7, 29, 10),
            measurementMode: MeasurementMode.standingHeight,
            oedema: OedemaStatus.notChecked,
            heightCm: 83.58,
          ),
        ),
        throwsA(
          isA<ArgumentError>().having(
            (error) => error.message,
            'message',
            contains('visit date'),
          ),
        ),
      );

      expect(await db.select(db.measurements).get(), isEmpty);
      expect(await db.select(db.measuredDetailRevisions).get(), isEmpty);
    });
  });
}

final DateTime _visitDate = DateTime(2026, 7, 29);

class _RevisionUuids {
  int _value = 0;

  String next() {
    _value += 1;
    return '40000000-0000-0000-0000-${_value.toString().padLeft(12, '0')}';
  }
}
