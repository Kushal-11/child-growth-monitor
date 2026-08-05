import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/reports/repositories/clinical_csv_export_repository.dart';
import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late AppDatabase database;
  late DriftClinicalCsvExportRepository repository;

  setUp(() {
    database = AppDatabase.forTesting(NativeDatabase.memory());
    repository = DriftClinicalCsvExportRepository(database);
  });

  tearDown(() => database.close());

  test(
    'exports owner-scoped assessment and camera reports, excluding drafts',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Child 001',
              dateOfBirth: '2022-01-28',
              sex: 'm',
              ownerUserId: const Value(7),
              location: const Value('Nehrunagar'),
            ),
          );
      final legacyAssessmentId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '10000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 6, 12)),
              ageMonths: 52.4,
              // Legacy assessment rows did not persist owner_user_id. The child
              // owner is the safe fallback for exporting those existing rows.
              ownerUserId: const Value(null),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: legacyAssessmentId,
              manualHeightCm: const Value(100),
              predictedHeightCm: const Value(100),
              heightMethod: const Value('manual'),
              manualWeightKg: const Value(12.7),
              predictedWeightKg: const Value(12.7),
              mlEstimatedWeightKg: const Value(20.61),
              weightMethod: const Value('manual'),
              muacCm: const Value(13.5),
              muacStatus: const Value('MAM'),
              muacMethod: const Value('manual'),
              muacAgeInRange: const Value(true),
              muacConfidence: const Value(1),
              muacUncertaintyLowerCm: const Value(13.5),
              muacUncertaintyUpperCm: const Value(13.5),
              muacCalibrationVersion: const Value('direct-tape'),
              muacIsDirectMeasurement: const Value(true),
              muacRequiresConfirmation: const Value(false),
              hazZscore: const Value(-1.25),
              whzZscore: const Value(-2.1),
              hazStatus: const Value('Normal'),
              whzStatus: const Value('MAM'),
              poshanStatus: const Value('MAM'),
              poshanComplete: const Value(true),
            ),
          );

      final guidedVisitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '10000000-0000-0000-0000-000000000002',
              visitDate: Value(DateTime(2026, 7, 1)),
              ageMonths: 53,
              ownerUserId: const Value(7),
              entryMethod: const Value('guided_capture'),
              captureState: const Value('estimated_report'),
            ),
          );
      await database.into(database.cameraResults).insert(
            CameraResultsCompanion.insert(
              resultUuid: '20000000-0000-0000-0000-000000000001',
              visitId: guidedVisitId,
              version: 1,
              estimatedHeightCm: const Value(105.6),
              estimatedWeightKg: const Value(20.61),
              heightSource: const Value(legacyWhoHeightSourceV1),
              weightSource: const Value(legacyWhoWeightSourceV1),
              estimatedStuntingStatus: const Value('Normal'),
              estimatedWastingStatus: const Value('Overweight'),
              experimentalOverallCategory: const Value('Normal'),
              method: 'camera_screening_v1',
              modelVersion: 'field-model-v1',
              manifestChecksum:
                  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
              trainingDataLabel: 'field-validation-pending',
            ),
          );
      // A draft has no result or measurement, so it is not an assessment/report.
      await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '10000000-0000-0000-0000-000000000003',
              ageMonths: 53,
              ownerUserId: const Value(7),
              entryMethod: const Value('guided_capture'),
              captureState: const Value('draft_capture'),
            ),
          );

      final otherChildId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Other worker child',
              dateOfBirth: '2023-01-01',
              sex: 'F',
              ownerUserId: const Value(8),
            ),
          );
      final otherVisitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: otherChildId,
              localUuid: '10000000-0000-0000-0000-000000000004',
              ageMonths: 40,
              ownerUserId: const Value(8),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: otherVisitId,
              manualHeightCm: const Value(90),
            ),
          );

      final records = await repository.loadSavedRecords(ownerUserId: 7);

      expect(records, hasLength(2));
      final measured = records.first;
      expect(measured.childId, childId);
      expect(measured.childName, 'Child 001');
      expect(measured.area, 'Nehrunagar');
      expect(measured.sex, 'M');
      expect(measured.measurementDate, '2026-06-12');
      expect(measured.actualHeightCm, 100);
      expect(measured.calculatedHeightCm, isNull);
      expect(measured.actualWeightKg, 12.7);
      expect(measured.calculatedWeightKg, 20.61);
      expect(measured.calculatedWeightMethod, experimentalMlWeightSourceV1);
      expect(measured.muacCm, 13.5);
      expect(measured.calculatedMuacCm, isNull);
      expect(measured.muacStatus, 'MAM');
      expect(measured.muacMethod, 'manual');
      expect(measured.muacAgeInRange, isTrue);
      expect(measured.muacConfidence, 1);
      expect(measured.muacUncertaintyLowerCm, 13.5);
      expect(measured.muacUncertaintyUpperCm, 13.5);
      expect(measured.muacCalibrationVersion, 'direct-tape');
      expect(measured.muacIsDirectMeasurement, isTrue);
      expect(measured.muacRequiresConfirmation, isFalse);
      expect(measured.hazZscore, -1.25);
      expect(measured.whzZscore, -2.1);
      expect(measured.fieldCategory, isNull);
      expect(measured.predictedFieldCategory, 'MAM');
      expect(measured.notes, contains('visit_uuid='));

      final estimated = records.last;
      expect(estimated.calculatedHeightCm, isNull);
      expect(estimated.calculatedWeightKg, isNull);
      expect(estimated.predictedFieldCategory, 'Normal');
      expect(estimated.stuntingPrediction, isNull);
      expect(estimated.wastingPrediction, isNull);
      expect(estimated.notes, contains('non_clinical=true'));
    },
  );

  test(
    'exports eligible calibrated camera estimates and their z-scores',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Calibrated Child',
              dateOfBirth: '2023-04-10',
              sex: 'F',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '30000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 8, 5)),
              ageMonths: 39.8,
              ownerUserId: const Value(7),
              entryMethod: const Value('guided_capture'),
              captureState: const Value('estimated_report'),
            ),
          );
      await database.into(database.cameraResults).insert(
            CameraResultsCompanion.insert(
              resultUuid: '40000000-0000-0000-0000-000000000001',
              visitId: visitId,
              version: 1,
              estimatedHeightCm: const Value(96.4),
              estimatedWeightKg: const Value(13.2),
              heightSource: const Value('reference_object'),
              weightSource: const Value(experimentalMlWeightSourceV1),
              estimatedHaz: const Value(-1.4),
              estimatedWhz: const Value(-1.8),
              estimatedStuntingStatus: const Value('Normal'),
              estimatedWastingStatus: const Value('Normal'),
              experimentalOverallCategory: const Value('Normal'),
              method: cameraScreeningMethodV1,
              modelVersion: 'field-model-v1',
              manifestChecksum:
                  'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
              trainingDataLabel: 'field-validation-pending',
            ),
          );

      final records = await repository.loadSavedRecords(ownerUserId: 7);

      expect(records, hasLength(1));
      final record = records.single;
      expect(record.childName, 'Calibrated Child');
      expect(record.calculatedHeightCm, 96.4);
      expect(record.calculatedHeightMethod, 'reference_object');
      expect(record.calculatedWeightKg, 13.2);
      expect(record.calculatedWeightMethod, experimentalMlWeightSourceV1);
      expect(record.hazZscore, -1.4);
      expect(record.whzZscore, -1.8);
      expect(record.stuntingPrediction, 'Normal');
      expect(record.wastingPrediction, 'Normal');
    },
  );

  test(
    'exports estimated MUAC evidence and confirmation guidance',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'MUAC Estimate Child',
              dateOfBirth: '2024-01-12',
              sex: 'M',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '50000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 8, 5)),
              ageMonths: 30.8,
              ownerUserId: const Value(7),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              predictedHeightCm: const Value(91.2),
              heightMethod: const Value('reference_object'),
              predictedWeightKg: const Value(11.4),
              mlEstimatedWeightKg: const Value(99),
              weightMethod: const Value('who_statistical'),
              muacCm: const Value(12.9),
              muacMethod: const Value('landmark_estimated'),
              muacAgeInRange: const Value(true),
              muacConfidence: const Value(0.82),
              muacUncertaintyLowerCm: const Value(12.1),
              muacUncertaintyUpperCm: const Value(13.7),
              muacModelVersion: const Value('landmark-ratio-v1'),
              muacCalibrationVersion: const Value('unvalidated-paired-tape-v0'),
              muacIsDirectMeasurement: const Value(false),
              muacRequiresConfirmation: const Value(true),
              muacReferralGuidance:
                  const Value('Confirm with a tape before clinical use.'),
            ),
          );

      final records = await repository.loadSavedRecords(ownerUserId: 7);

      expect(records, hasLength(1));
      final record = records.single;
      expect(record.actualHeightCm, isNull);
      expect(record.calculatedHeightCm, 91.2);
      expect(record.calculatedHeightMethod, 'reference_object');
      expect(record.actualWeightKg, isNull);
      expect(record.calculatedWeightKg, 11.4);
      expect(record.calculatedWeightMethod, legacyWhoWeightSourceV1);
      expect(record.muacCm, isNull);
      expect(record.calculatedMuacCm, 12.9);
      expect(record.muacMethod, 'landmark_estimated');
      expect(record.muacAgeInRange, isTrue);
      expect(record.muacConfidence, 0.82);
      expect(record.muacUncertaintyLowerCm, 12.1);
      expect(record.muacUncertaintyUpperCm, 13.7);
      expect(record.muacModelVersion, 'landmark-ratio-v1');
      expect(record.muacCalibrationVersion, 'unvalidated-paired-tape-v0');
      expect(record.muacIsDirectMeasurement, isFalse);
      expect(record.muacRequiresConfirmation, isTrue);
      expect(
        record.muacReferralGuidance,
        'Confirm with a tape before clinical use.',
      );
    },
  );
}
