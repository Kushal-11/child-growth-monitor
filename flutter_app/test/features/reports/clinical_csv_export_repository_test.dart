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

  test('exports owner-scoped assessment and camera reports, excluding drafts',
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
            muacMethod: const Value('manual'),
            muacIsDirectMeasurement: const Value(true),
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
    expect(measured.area, 'Nehrunagar');
    expect(measured.sex, 'M');
    expect(measured.measurementDate, '2026-06-12');
    expect(measured.actualHeightCm, 100);
    expect(measured.calculatedHeightCm, isNull);
    expect(measured.actualWeightKg, 12.7);
    expect(measured.calculatedWeightKg, isNull);
    expect(measured.muacCm, 13.5);
    expect(measured.calculatedMuacCm, isNull);
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
  });
}
