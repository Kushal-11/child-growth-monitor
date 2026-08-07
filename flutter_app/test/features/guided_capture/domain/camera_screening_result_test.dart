import 'dart:convert';

import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('manifest metadata checksum is derived from the exact manifest bytes',
      () {
    final bytes = utf8.encode(jsonEncode({
      'schema_version': 1,
      'model_version': 'synthetic-who-v1',
      'training_data_label': 'synthetic_who_research_only',
      'non_clinical': true,
      'files': <String, Object?>{},
    }));

    final metadata = CameraModelMetadata.fromManifestBytes(bytes);

    expect(metadata.modelVersion, 'synthetic-who-v1');
    expect(metadata.trainingDataLabel, 'synthetic_who_research_only');
    expect(metadata.manifestChecksum, sha256.convert(bytes).toString());
  });

  test('camera result serializes only explicitly estimated provenance', () {
    final result = CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000001',
      version: 1,
      estimatedHeightCm: 88,
      estimatedWeightKg: 11,
      estimatedMuacCm: 12.4,
      heightSource: 'who_height_for_age_median_v1',
      weightSource: 'ml_weight_estimator_v1',
      muacSource: 'photo_landmark',
      estimatedHaz: -0.2,
      estimatedWhz: -1.1,
      estimatedStuntingStatus: 'Normal',
      estimatedWastingStatus: 'NORMAL',
      experimentalOverallCategory: 'MAM',
      componentProbabilities: const {
        'SAM': 0.1,
        'MAM': 0.6,
        'Normal': 0.2,
        'Risk_Overweight': 0.05,
        'Overweight': 0.05,
      },
      bodyProportionFeatures: const {'shoulder_height_ratio': 0.2},
      captureQualitySummary: const {
        'overall': 0.91,
        'used_views': ['front', 'side'],
      },
      method: cameraScreeningMethodV1,
      modelVersion: 'synthetic-who-v1',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'synthetic_who_research_only',
      createdAt: DateTime.utc(2026, 7, 29),
    );

    final json = result.toJson();
    final keys = json.keys.join(' ');

    expect(json['non_clinical'], isTrue);
    expect(json['estimated_haz'], -0.2);
    expect(json['estimated_whz'], -1.1);
    expect(keys, isNot(contains('manual')));
    expect(keys, isNot(contains('poshan')));
    expect(json['estimated_muac_cm'], 12.4);
    expect(json['muac_source'], 'photo_landmark');
    expect(keys, isNot(contains('oedema')));
    expect(keys, isNot(contains('haz_zscore')));
    expect(keys, isNot(contains('whz_zscore')));
    expect(CameraScreeningResult.fromJson(json).toJson(), json);
  });

  test('camera category is rejected without valid component probabilities', () {
    expect(
      () => CameraScreeningResult(
        resultUuid: '30000000-0000-0000-0000-000000000002',
        version: 1,
        experimentalOverallCategory: 'Normal',
        componentProbabilities: const {'Normal': 1.2},
        captureQualitySummary: const {},
        method: cameraScreeningMethodV1,
        modelVersion: 'synthetic-who-v1',
        manifestChecksum: 'b' * 64,
        trainingDataLabel: 'synthetic_who_research_only',
        createdAt: DateTime.utc(2026, 7, 29),
      ),
      throwsArgumentError,
    );
  });

  test('population estimates remain visible with explicit provenance', () {
    final result = CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000003',
      version: 1,
      estimatedHeightCm: 90,
      estimatedWeightKg: 12,
      heightSource: legacyWhoHeightSourceV1,
      weightSource: legacyWhoWeightSourceV1,
      estimatedHaz: 0,
      estimatedWhz: 0,
      estimatedStuntingStatus: 'Normal',
      estimatedWastingStatus: 'NORMAL',
      captureQualitySummary: const {},
      method: cameraScreeningMethodV1,
      modelVersion: 'synthetic-who-v1',
      manifestChecksum: 'c' * 64,
      trainingDataLabel: 'synthetic_who_research_only',
      createdAt: DateTime.utc(2026, 7, 29),
    );

    expect(result.reportableHeightCm, 90);
    expect(result.reportableWeightKg, 12);
    expect(result.reportableHaz, isNull);
    expect(result.reportableWhz, isNull);
    expect(result.reportableStuntingStatus, isNull);
    expect(result.reportableWastingStatus, isNull);
  });
}
