import 'dart:convert';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/reports/repositories/clinical_csv_export_repository.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';
import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late AppDatabase database;
  late DriftClinicalCsvExportRepository repository;
  late WhoDataService who;

  setUpAll(() async {
    who = WhoDataService();
    await who.loadFromFiles(
      manifestPath: 'assets/who_data/who_reference_manifest.json',
      wflBoysPath: 'assets/who_data/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'assets/who_data/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'assets/who_data/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'assets/who_data/who_wfh_girls_2_5.xlsx',
      lfaBoysPath: 'assets/who_data/who_lhfa_boys_0_2.xlsx',
      lfaGirlsPath: 'assets/who_data/who_lhfa_girls_0_2.xlsx',
      hfaBoysPath: 'assets/who_data/who_lhfa_boys_2_5.xlsx',
      hfaGirlsPath: 'assets/who_data/who_lhfa_girls_2_5.xlsx',
      wfaBoysPath: 'assets/who_data/who_wfa_boys_0_5.xlsx',
      wfaGirlsPath: 'assets/who_data/who_wfa_girls_0_5.xlsx',
      bfaBoys0To2Path: 'assets/who_data/who_bfa_boys_0_2.xlsx',
      bfaBoys2To5Path: 'assets/who_data/who_bfa_boys_2_5.xlsx',
      bfaGirls0To2Path: 'assets/who_data/who_bfa_girls_0_2.xlsx',
      bfaGirls2To5Path: 'assets/who_data/who_bfa_girls_2_5.xlsx',
      acfaBoysPath: 'assets/who_data/who_acfa_boys_3_5.xlsx',
      acfaGirlsPath: 'assets/who_data/who_acfa_girls_3_5.xlsx',
    );
  });

  setUp(() {
    database = AppDatabase.forTesting(NativeDatabase.memory());
    repository = DriftClinicalCsvExportRepository(database, whoData: who);
  });

  tearDown(() => database.close());

  test(
    'exports identifying inputs and actual/calculated values side by side',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Anaya Patil',
              dateOfBirth: '2022-01-28',
              sex: 'f',
              guardianName: const Value('Meera Patil'),
              location: const Value('Nehrunagar'),
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '10000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 6, 12)),
              ageMonths: 52.4,
              ownerUserId: const Value(null),
              entryMethod: const Value('manual'),
              captureState: const Value('measured_report'),
              notes: const Value('Visit note'),
              consentVersion: const Value('photo-consent-v1'),
              consentTimestamp: Value(DateTime(2026, 6, 12, 9, 45)),
              consentOperatorIdentifier: const Value('field.worker'),
            ),
          );
      await database.into(database.measuredDetailRevisions).insert(
            MeasuredDetailRevisionsCompanion.insert(
              revisionUuid: '70000000-0000-0000-0000-000000000001',
              visitId: visitId,
              revisionNumber: 1,
              beforeJson: '{}',
              afterJson: '{}',
              reason: const Value('Tape measurement added'),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              manualHeightCm: const Value(100),
              predictedHeightCm: const Value(98.4),
              heightMethod: const Value('manual'),
              heightConfidence: const Value(0.91),
              manualWeightKg: const Value(12.7),
              predictedWeightKg: const Value(12.7),
              mlEstimatedWeightKg: const Value(13.2),
              weightMethod: const Value('manual'),
              weightConfidence: const Value(0.86),
              estimationMethod: const Value('reference_object'),
              muacCm: const Value(13.5),
              muacStatus: const Value('MAM'),
              muacMethod: const Value('tape'),
              muacAgeInRange: const Value(true),
              muacIsDirectMeasurement: const Value(true),
              hazZscore: const Value(-1.25),
              whzZscore: const Value(-2.1),
              // Deliberately mixed raw labels. Z-scores are authoritative for
              // the typed stunting and wasting export columns.
              hazStatus: const Value('SAM'),
              whzStatus: const Value('MAM'),
              bmi: const Value(12.7),
              bmiStatus: const Value('MAM'),
              poshanStatus: const Value('MAM'),
              poshanTriggeredBy: const Value('["muac"]'),
              classificationMethod: const Value('poshan_setu_v1'),
              classificationRationale: const Value('MAM flagged by MUAC'),
              classificationConfidence: const Value(0.9),
              poshanComplete: const Value(true),
              measurementMode: const Value('standing_height'),
              oedema: const Value('no'),
              measuredAt: Value(DateTime(2026, 6, 12, 10, 30)),
              measuredNotes: const Value('Measurement note'),
              bodyBuild: const Value('average'),
              sideViewUsed: const Value(true),
              samProbability: const Value(0.1),
              mamProbability: const Value(0.7),
              normalProbability: const Value(0.15),
              riskOverweightProbability: const Value(0.03),
              overweightProbability: const Value(0.02),
            ),
          );

      // Drafts and another owner's visits must not leak into this export.
      await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '10000000-0000-0000-0000-000000000002',
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
              sex: 'M',
              ownerUserId: const Value(8),
            ),
          );
      final otherVisitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: otherChildId,
              localUuid: '10000000-0000-0000-0000-000000000003',
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

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.childName, 'Anaya Patil');
      expect(record.childId, childId);
      expect(record.guardianName, 'Meera Patil');
      expect(record.area, 'Nehrunagar');
      expect(record.sex, 'F');
      expect(record.recordedAgeMonths, 52.4);
      expect(record.ageDays, isNotNull);
      expect(record.whoAgeMonths, isNotNull);
      expect(record.visitUuid, '10000000-0000-0000-0000-000000000001');
      expect(record.entryMethod, 'manual');
      expect(record.consentVersion, 'photo-consent-v1');
      expect(record.consentOperatorIdentifier, 'field.worker');
      expect(record.measurementMode, 'standing_height');
      expect(record.oedema, 'No');
      expect(record.measuredAt, '2026-06-12T10:30:00.000');
      expect(record.measurementUpdateReason, 'Tape measurement added');
      expect(record.actualHeightCm, 100);
      expect(record.actualWhoAdjustedHeightCm, 100);
      expect(record.calculatedHeightCm, 98.4);
      expect(record.calculatedHeightMethod, 'reference_object');
      expect(record.actualWeightKg, 12.7);
      expect(record.calculatedWeightKg, 13.2);
      expect(record.calculatedWeightMethod, experimentalMlWeightSourceV1);
      expect(record.actualMuacCm, 13.5);
      // Stored MAM is deliberately wrong for 13.5 cm; export recomputes it.
      expect(record.actualMuacStatus, 'Normal');
      expect(record.actualMuacMethod, 'tape');
      // A direct tape MUAC must not be duplicated with a synthetic value
      // derived from WHZ. Calculated MUAC is exported only when independently
      // stored by an estimation pathway.
      expect(record.calculatedMuacCm, isNull);
      expect(record.calculatedMuacMethod, isNull);
      expect(record.actualHazZscore, isNotNull);
      expect(record.actualWhzZscore, isNotNull);
      expect(record.actualWazZscore, isNotNull);
      expect(record.actualBazZscore, isNotNull);
      expect(record.actualHazQualityFlag, 'OK');
      expect(record.actualWhzQualityFlag, 'OK');
      expect(record.actualStuntingClassification, isNot(anyOf('SAM', 'MAM')));
      expect(record.actualWastingClassification, isNot(anyOf('SAM', 'MAM')));
      expect(record.calculatedHazZscore, isNotNull);
      expect(record.calculatedWhzZscore, isNotNull);
      expect(record.calculatedWazZscore, isNotNull);
      expect(record.calculatedBazZscore, isNotNull);
      expect(record.poshanSetuBmiStatus, 'SAM');
      expect(record.poshanSetuMuacStatus, 'Normal');
      expect(record.poshanSetuFinalStatus, 'SAM');
      expect(record.storedOverallNutritionPrediction, 'MAM');
      expect(record.visitNotes, 'Visit note');
      expect(record.measurementNotes, 'Measurement note');
      expect(
        record.provenanceNotes,
        isNot(contains('calculated_muac_generated_from_calculated_whz=true')),
      );
    },
  );

  test(
    'uses strict stunting and wasting vocabularies for camera predictions',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Camera Child',
              dateOfBirth: '2023-04-10',
              sex: 'F',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '20000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 8, 5)),
              ageMonths: 39.8,
              ownerUserId: const Value(7),
              entryMethod: const Value('guided_capture'),
              captureState: const Value('estimated_report'),
            ),
          );
      await database.into(database.cameraResults).insert(
            CameraResultsCompanion.insert(
              resultUuid: '30000000-0000-0000-0000-000000000001',
              visitId: visitId,
              version: 1,
              estimatedHeightCm: const Value(96.4),
              estimatedWeightKg: const Value(13.2),
              heightSource: const Value('reference_object'),
              weightSource: const Value(experimentalMlWeightSourceV1),
              estimatedHaz: const Value(-3.4),
              estimatedWhz: const Value(-2.5),
              estimatedStuntingStatus: const Value('SAM'),
              estimatedWastingStatus: const Value('Stunted'),
              experimentalOverallCategory: const Value('SAM'),
              method: cameraScreeningMethodV1,
              modelVersion: 'field-model-v1',
              manifestChecksum:
                  'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
              trainingDataLabel: 'field-validation-pending',
            ),
          );

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.calculatedHeightCm, 96.4);
      expect(record.calculatedWeightKg, 13.2);
      expect(record.calculatedStuntingPrediction, 'Not Stunted');
      expect(record.calculatedWastingPrediction, isNotNull);
      expect(record.storedOverallNutritionPrediction, 'SAM');
      expect(record.calculatedStuntingPrediction, isNot(anyOf('SAM', 'MAM')));
      expect(record.calculatedWastingPrediction, isNot(anyOf('SAM', 'MAM')));
      expect(record.calculatedWazZscore, isNotNull);
      expect(record.calculatedBazZscore, isNotNull);
      expect(record.provenanceNotes, contains('camera_non_clinical=true'));
      expect(
        record.provenanceNotes,
        contains('camera_zscores_recomputed_from_same_basis_values=true'),
      );
    },
  );

  test('exports ARCore height, geometry weight, MUAC, and quality evidence',
      () async {
    const scan = FullArScanResult(
      estimatedHeightCm: 91.2,
      uncertaintyCm: 0.6,
      acceptedKeyframes: 20,
      validDepthFraction: 0.58,
      meanDepthConfidence: 0.84,
      scanCoverageDegrees: 92,
      cameraTravelMeters: 0.8,
      floorStabilityCm: 1.1,
      capturedBodyPoints: 6400,
      durationMs: 15000,
      qualityScore: 0.92,
      depthMode: 'raw_depth_with_confidence',
      shoulderWidthCm: 23,
      hipWidthCm: 21,
      torsoLengthCm: 30,
      upperArmLengthCm: 16,
      chestDepthCm: 13,
      abdomenDepthCm: 12,
      estimatedMuacCm: 13.1,
      muacUncertaintyCm: 0.4,
      poseQualityScore: 0.9,
      geometryQualityScore: 0.88,
    );
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'ARCore Child',
            dateOfBirth: '2023-10-10',
            sex: 'M',
            ownerUserId: const Value(7),
          ),
        );
    final visitId = await database.into(database.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: '35000000-0000-0000-0000-000000000001',
            visitDate: Value(DateTime(2026, 8, 5)),
            ageMonths: 33.8,
            ownerUserId: const Value(7),
            entryMethod: const Value('assessment'),
            deviceMetadataJson: Value(
              jsonEncode({'arcore_depth_scan': scan.toJson()}),
            ),
          ),
        );
    await database.into(database.measurements).insert(
          MeasurementsCompanion.insert(
            visitId: visitId,
            manualHeightCm: const Value(90),
            manualWeightKg: const Value(10.5),
            muacCm: const Value(12.8),
            muacMethod: const Value('tape'),
            muacIsDirectMeasurement: const Value(true),
          ),
        );
    await database.into(database.cameraResults).insert(
          CameraResultsCompanion.insert(
            resultUuid: '36000000-0000-0000-0000-000000000001',
            visitId: visitId,
            version: 1,
            estimatedHeightCm: const Value(91.2),
            estimatedWeightKg: const Value(11.7),
            estimatedMuacCm: const Value(13.1),
            heightSource: const Value(arcoreDepthHeightSourceV3),
            weightSource: const Value(arcoreGeometryWeightSourceV3),
            muacSource: const Value(arcoreArmMuacSourceV3),
            heightRangeLowerCm: const Value(90.6),
            heightRangeUpperCm: const Value(91.8),
            weightRangeLowerKg: const Value(11.1),
            weightRangeUpperKg: const Value(12.3),
            muacRangeLowerCm: const Value(12.7),
            muacRangeUpperCm: const Value(13.5),
            method: cameraScreeningContactlessMethodV2,
            modelVersion: 'field-model-v1',
            manifestChecksum:
                'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
            trainingDataLabel: 'field-validation-pending',
          ),
        );

    final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

    expect(record.exportSchemaVersion, 'clinical_csv_v5_arcore_recovery');
    expect(record.calculatedHeightCm, 91.2);
    expect(record.calculatedHeightMethod, arcoreDepthHeightSourceV3);
    expect(record.calculatedWeightKg, 11.7);
    expect(record.calculatedWeightMethod, arcoreGeometryWeightSourceV3);
    expect(record.calculatedMuacCm, 13.1);
    expect(record.calculatedMuacMethod, arcoreArmMuacSourceV3);
    expect(record.arcoreScanAvailable, isTrue);
    expect(record.arcoreDepthHeightCm, 91.2);
    expect(record.arcoreGeometryMlWeightKg, 11.7);
    expect(record.arcoreArmMuacCm, 13.1);
    expect(record.arcoreQualityScore, 0.92);
    expect(record.arcoreShoulderWidthCm, 23);
    expect(record.provenanceNotes, contains('arcore_non_clinical=true'));
  });

  test(
    'leaves cross-domain labels blank when no z-score can correct them',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Mixed Label Child',
              dateOfBirth: '2023-01-01',
              sex: 'M',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '40000000-0000-0000-0000-000000000001',
              ageMonths: 43,
              ownerUserId: const Value(7),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              hazStatus: const Value('SAM'),
              whzStatus: const Value('Stunted'),
              poshanStatus: const Value('MAM'),
              poshanComplete: const Value(true),
            ),
          );

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.calculatedStuntingPrediction, isNull);
      expect(record.calculatedWastingPrediction, isNull);
      expect(record.storedOverallNutritionPrediction, 'MAM');
      expect(record.poshanSetuFinalStatus, 'Indeterminate');
    },
  );

  test(
    'exports estimated MUAC evidence and suppresses implausible ML weight',
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
              muacStatus: const Value('MAM'),
              muacMethod: const Value('landmark_estimated'),
              muacAgeInRange: const Value(true),
              muacConfidence: const Value(0.82),
              muacUncertaintyLowerCm: const Value(12.1),
              muacUncertaintyUpperCm: const Value(13.7),
              muacModelVersion: const Value('landmark-ratio-v1'),
              muacCalibrationVersion: const Value('unvalidated-paired-tape-v0'),
              muacIsDirectMeasurement: const Value(false),
              muacRequiresConfirmation: const Value(true),
              muacReferralGuidance: const Value(
                'Confirm with a tape before clinical use.',
              ),
            ),
          );

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.actualHeightCm, isNull);
      expect(record.calculatedHeightCm, 91.2);
      expect(record.actualWeightKg, isNull);
      expect(record.calculatedWeightKg, 11.4);
      expect(record.calculatedWeightMethod, legacyWhoWeightSourceV1);
      expect(record.actualMuacCm, isNull);
      expect(record.calculatedMuacCm, 12.9);
      // Stored MAM is deliberately inconsistent with 12.9 cm.
      expect(record.calculatedMuacStatus, 'Normal');
      expect(record.calculatedMuacMethod, 'landmark_estimated');
      expect(record.calculatedMuacConfidence, 0.82);
      expect(record.calculatedMuacUncertaintyLowerCm, 12.1);
      expect(record.calculatedMuacUncertaintyUpperCm, 13.7);
      expect(record.calculatedMuacModelVersion, 'landmark-ratio-v1');
      expect(
        record.calculatedMuacCalibrationVersion,
        'unvalidated-paired-tape-v0',
      );
      expect(record.calculatedMuacRequiresConfirmation, isTrue);
      expect(
        record.calculatedMuacReferralGuidance,
        'Confirm with a tape before clinical use.',
      );
      expect(
        record.provenanceNotes,
        contains('implausible_ml_weight_suppressed=true'),
      );
    },
  );

  test(
    'does not invent WHO median predictions for manual-only records',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Manual Only Child',
              dateOfBirth: '2023-01-01',
              sex: 'F',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '60000000-0000-0000-0000-000000000001',
              ageMonths: 43.1,
              ownerUserId: const Value(7),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              manualHeightCm: const Value(90),
              heightMethod: const Value('manual'),
              manualWeightKg: const Value(10),
              weightMethod: const Value('manual'),
              muacCm: const Value(13),
              muacMethod: const Value('manual'),
              muacIsDirectMeasurement: const Value(true),
            ),
          );

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.actualHeightCm, 90);
      expect(record.calculatedHeightCm, isNull);
      expect(record.actualWeightKg, 10);
      expect(record.calculatedWeightKg, isNull);
      expect(record.actualMuacCm, 13);
      expect(record.calculatedMuacCm, isNull);
    },
  );

  test(
    'recovers stored ML outputs without duplicating manual evidence',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Historical Assessment Child',
              dateOfBirth: '2023-01-01',
              sex: 'F',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '61000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 1, 1)),
              ageMonths: 36,
              ownerUserId: const Value(7),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              manualHeightCm: const Value(90),
              predictedHeightCm: const Value(90),
              effectiveHeightCm: const Value(90),
              heightMethod: const Value('manual'),
              heightConfidence: const Value(1),
              manualWeightKg: const Value(10),
              predictedWeightKg: const Value(10),
              effectiveWeightKg: const Value(10),
              weightMethod: const Value('manual'),
              weightConfidence: const Value(1),
              mlEstimatedWeightKg: const Value(12),
              wastingStatus: const Value('MAM'),
              wastingMethod: const Value('ml_classifier'),
              samProbability: const Value(0.1),
              mamProbability: const Value(0.7),
              normalProbability: const Value(0.15),
              riskOverweightProbability: const Value(0.03),
              overweightProbability: const Value(0.02),
              muacCm: const Value(13),
              muacMethod: const Value('manual'),
              muacIsDirectMeasurement: const Value(true),
              measurementMode: const Value('standing_height'),
              oedema: const Value('no'),
            ),
          );

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.actualHeightCm, 90);
      expect(record.calculatedHeightCm, isNull);
      expect(record.calculatedHeightAvailability, 'not_independently_recorded');
      expect(record.calculatedWeightKg, 12);
      expect(record.calculatedWeightConfidence, isNull);
      expect(record.calculatedWeightAvailability, 'available');
      expect(record.weightErrorKg, 2);
      expect(record.calculatedWhzZscore, isNull);
      expect(record.calculatedWazZscore, isNotNull);
      expect(record.actualMuacCm, 13);
      expect(record.calculatedMuacCm, isNull);
      expect(record.calculatedMuacAvailability, 'not_independently_recorded');
      expect(record.mlEstimatedWeightKg, 12);
      expect(record.mlWeightAcceptedForCalculation, isTrue);
      expect(record.mlWastingPrediction, 'MAM');
      expect(record.mlWastingMethod, 'ml_classifier');
      expect(record.samProbability, 0.1);
      expect(record.mamProbability, 0.7);
      expect(
        record.provenanceNotes,
        contains('manual_height_duplicate_suppressed=true'),
      );
      expect(
        record.provenanceNotes,
        contains('ml_weight_confidence_not_recorded=true'),
      );
      expect(
        record.provenanceNotes,
        contains('calculated_muac_not_independently_recorded=true'),
      );
    },
  );

  test(
    'suppresses stored WHO population height from child estimates',
    () async {
      final childId = await database.into(database.children).insert(
            ChildrenCompanion.insert(
              name: 'Historical Reference Child',
              dateOfBirth: '2023-01-01',
              sex: 'M',
              ownerUserId: const Value(7),
            ),
          );
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: '62000000-0000-0000-0000-000000000001',
              visitDate: Value(DateTime(2026, 1, 1)),
              ageMonths: 36,
              ownerUserId: const Value(7),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              predictedHeightCm: const Value(95),
              effectiveHeightCm: const Value(95),
              heightMethod: const Value('who_statistical'),
              mlEstimatedWeightKg: const Value(13),
              weightMethod: const Value('ml_estimated'),
            ),
          );

      final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

      expect(record.calculatedHeightCm, isNull);
      expect(
        record.calculatedHeightAvailability,
        'population_reference_suppressed',
      );
      expect(record.mlEstimatedWeightKg, 13);
      expect(record.calculatedWeightKg, 13);
      expect(record.calculatedWazZscore, isNotNull);
      expect(record.calculatedWhzZscore, isNull);
      expect(
        record.provenanceNotes,
        contains('stored_population_height_suppressed=true'),
      );
    },
  );

  test('never mixes measured height with calculated weight', () async {
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'No Mixed Inputs Child',
            dateOfBirth: '2023-01-01',
            sex: 'M',
            ownerUserId: const Value(7),
          ),
        );
    final visitId = await database.into(database.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: '80000000-0000-0000-0000-000000000001',
            visitDate: Value(DateTime(2026, 1, 1)),
            ageMonths: 36,
            ownerUserId: const Value(7),
          ),
        );
    await database.into(database.measurements).insert(
          MeasurementsCompanion.insert(
            visitId: visitId,
            manualHeightCm: const Value(90),
            heightMethod: const Value('manual'),
            predictedWeightKg: const Value(11),
            weightMethod: const Value('ml_estimated'),
            measurementMode: const Value('standing_height'),
            oedema: const Value('no'),
          ),
        );

    final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

    expect(record.actualHazZscore, isNotNull);
    expect(record.actualWhzZscore, isNull);
    expect(record.actualWhzQualityFlag, 'UNAVAILABLE_MISSING_INPUT');
    expect(record.calculatedWeightKg, 11);
    expect(record.calculatedHeightCm, isNull);
    expect(record.calculatedWazZscore, isNotNull);
    expect(record.calculatedWhzZscore, isNull);
    expect(record.calculatedBazZscore, isNull);
    expect(record.calculatedAcuteNutritionPrediction, 'Indeterminate');
  });

  test('oedema triggers measured SAM and suppresses weight scores', () async {
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'Oedema Child',
            dateOfBirth: '2023-01-01',
            sex: 'F',
            ownerUserId: const Value(7),
          ),
        );
    final visitId = await database.into(database.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: '81000000-0000-0000-0000-000000000001',
            visitDate: Value(DateTime(2026, 1, 1)),
            ageMonths: 36,
            ownerUserId: const Value(7),
          ),
        );
    await database.into(database.measurements).insert(
          MeasurementsCompanion.insert(
            visitId: visitId,
            manualHeightCm: const Value(95),
            manualWeightKg: const Value(12),
            heightMethod: const Value('manual'),
            weightMethod: const Value('manual'),
            muacCm: const Value(13),
            muacMethod: const Value('tape'),
            muacIsDirectMeasurement: const Value(true),
            measurementMode: const Value('standing_height'),
            oedema: const Value('yes'),
          ),
        );

    final record = (await repository.loadSavedRecords(ownerUserId: 7)).single;

    expect(record.actualHazZscore, isNotNull);
    expect(record.actualWhzZscore, isNull);
    expect(record.actualWazZscore, isNull);
    expect(record.actualBazZscore, isNull);
    expect(record.actualWhzQualityFlag, 'NOT_INTERPRETABLE_OEDEMA');
    expect(record.actualWazQualityFlag, 'NOT_INTERPRETABLE_OEDEMA');
    expect(record.actualBazQualityFlag, 'NOT_INTERPRETABLE_OEDEMA');
    expect(record.actualAcuteNutritionClassification, 'SAM');
    expect(record.actualAcuteTriggeredBy, '["oedema"]');
    expect(record.oedemaGrade, isNull);
    expect(record.provenanceNotes, contains('oedema_grade_not_collected=true'));
  });

  test('exports measured change from the previous completed visit', () async {
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'Longitudinal Child',
            dateOfBirth: '2023-01-01',
            sex: 'M',
            ownerUserId: const Value(7),
          ),
        );
    Future<void> addVisit({
      required String uuid,
      required DateTime date,
      required double height,
      required double weight,
      required double muac,
    }) async {
      final visitId = await database.into(database.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: uuid,
              visitDate: Value(date),
              ageMonths: 36,
              ownerUserId: const Value(7),
            ),
          );
      await database.into(database.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              manualHeightCm: Value(height),
              manualWeightKg: Value(weight),
              heightMethod: const Value('manual'),
              weightMethod: const Value('manual'),
              muacCm: Value(muac),
              muacMethod: const Value('tape'),
              muacIsDirectMeasurement: const Value(true),
              measurementMode: const Value('standing_height'),
              oedema: const Value('no'),
            ),
          );
    }

    await addVisit(
      uuid: '82000000-0000-0000-0000-000000000001',
      date: DateTime(2026, 1, 1),
      height: 90,
      weight: 11,
      muac: 12,
    );
    await addVisit(
      uuid: '82000000-0000-0000-0000-000000000002',
      date: DateTime(2026, 2, 1),
      height: 91,
      weight: 11.5,
      muac: 12.3,
    );

    final records = await repository.loadSavedRecords(ownerUserId: 7);
    expect(records, hasLength(2));
    expect(records.first.previousMeasurementDate, isNull);
    expect(records.last.previousMeasurementDate, '2026-01-01');
    expect(records.last.daysSincePreviousMeasurement, 31);
    expect(records.last.actualHeightChangeCm, 1);
    expect(records.last.actualWeightChangeKg, 0.5);
    expect(records.last.actualMuacChangeCm, closeTo(0.3, 0.0001));
  });
}
