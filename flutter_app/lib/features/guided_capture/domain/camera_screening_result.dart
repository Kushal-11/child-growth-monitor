import 'dart:convert';

import 'package:crypto/crypto.dart';

import '../../ar_scan/domain/ar_scan_models.dart';
import 'capture_models.dart';

const String cameraScreeningMethodV1 = 'camera_screening_v1';
const String cameraScreeningContactlessMethodV2 =
    'camera_screening_contactless_v2';
const String experimentalMlWeightSourceV1 =
    'experimental_ml_weight_estimator_v1';
const String arcoreDepthHeightSourceV3 = 'arcore_depth_height_v3';
const String arcoreDepthHeightSourceV2 = 'arcore_depth_height_v2';
const String arcoreGeometryWeightSourceV3 = 'arcore_geometry_ml_weight_v3';
const String arcoreHeightPhotoGeometryWeightSourceV3 =
    'arcore_height_photo_geometry_ml_weight_v3';
const String arcoreArmMuacSourceV3 = 'arcore_arm_cross_section_muac_v3';
const String legacyWhoHeightSourceV1 = 'who_height_for_age_median_v1';
const String legacyWhoWeightSourceV1 =
    'who_weight_for_height_median_body_build_v1';
const String whoReferenceFeatureScalingV1 =
    'who_population_reference_for_feature_scaling_v1';

const Set<String> cameraClassifierCategories = {
  'SAM',
  'MAM',
  'Normal',
  'Risk_Overweight',
  'Overweight',
};

class CameraModelMetadata {
  const CameraModelMetadata({
    required this.modelVersion,
    required this.manifestChecksum,
    required this.trainingDataLabel,
  });

  final String modelVersion;
  final String manifestChecksum;
  final String trainingDataLabel;

  factory CameraModelMetadata.fromManifestBytes(List<int> bytes) {
    final decoded = jsonDecode(utf8.decode(bytes));
    if (decoded is! Map<String, dynamic> ||
        decoded['schema_version'] != 1 ||
        decoded['model_version'] is! String ||
        decoded['training_data_label'] is! String ||
        decoded['non_clinical'] != true ||
        decoded['files'] is! Map<String, dynamic>) {
      throw const FormatException('Unsupported camera model manifest');
    }
    return CameraModelMetadata(
      modelVersion: decoded['model_version'] as String,
      manifestChecksum: sha256.convert(bytes).toString(),
      trainingDataLabel: decoded['training_data_label'] as String,
    );
  }
}

class CameraScreeningVisit {
  const CameraScreeningVisit({
    required this.visitUuid,
    required this.ownerUserId,
    required this.ageMonths,
    required this.sex,
    this.arScan,
  });

  final String visitUuid;
  final int ownerUserId;
  final double ageMonths;
  final String sex;
  final FullArScanResult? arScan;
}

class CameraScreeningAsset {
  const CameraScreeningAsset({
    required this.role,
    required this.localPath,
    this.poseScore,
    this.coverageScore,
    this.orientationScore,
    this.sharpnessScore,
    this.lightingScore,
    this.overallScore,
  });

  final CaptureAssetRole role;
  final String localPath;
  final double? poseScore;
  final double? coverageScore;
  final double? orientationScore;
  final double? sharpnessScore;
  final double? lightingScore;
  final double? overallScore;
}

class CameraScreeningResult {
  CameraScreeningResult({
    required this.resultUuid,
    required this.version,
    this.supersedesResultUuid,
    this.estimatedHeightCm,
    this.estimatedWeightKg,
    this.estimatedMuacCm,
    this.heightSource,
    this.weightSource,
    this.muacSource,
    this.heightRangeLowerCm,
    this.heightRangeUpperCm,
    this.weightRangeLowerKg,
    this.weightRangeUpperKg,
    this.muacRangeLowerCm,
    this.muacRangeUpperCm,
    this.estimatedHaz,
    this.estimatedWhz,
    this.estimatedStuntingStatus,
    this.estimatedWastingStatus,
    this.experimentalOverallCategory,
    Map<String, double> componentProbabilities = const {},
    Map<String, Object?> bodyProportionFeatures = const {},
    required Map<String, Object?> captureQualitySummary,
    required this.method,
    required this.modelVersion,
    required this.manifestChecksum,
    required this.trainingDataLabel,
    required this.createdAt,
  })  : componentProbabilities =
            Map<String, double>.unmodifiable(componentProbabilities),
        bodyProportionFeatures =
            Map<String, Object?>.unmodifiable(bodyProportionFeatures),
        captureQualitySummary =
            Map<String, Object?>.unmodifiable(captureQualitySummary) {
    if (resultUuid.isEmpty || version < 1) {
      throw ArgumentError('A result UUID and positive version are required');
    }
    if (method != cameraScreeningMethodV1 &&
        method != cameraScreeningContactlessMethodV2) {
      throw ArgumentError.value(method, 'method', 'unsupported method');
    }
    if (!RegExp(r'^[a-f0-9]{64}$').hasMatch(manifestChecksum)) {
      throw ArgumentError.value(
        manifestChecksum,
        'manifestChecksum',
        'must be a lowercase SHA-256 checksum',
      );
    }
    for (final entry in {
      'estimatedHeightCm': estimatedHeightCm,
      'estimatedWeightKg': estimatedWeightKg,
      'estimatedMuacCm': estimatedMuacCm,
      'heightRangeLowerCm': heightRangeLowerCm,
      'heightRangeUpperCm': heightRangeUpperCm,
      'weightRangeLowerKg': weightRangeLowerKg,
      'weightRangeUpperKg': weightRangeUpperKg,
      'muacRangeLowerCm': muacRangeLowerCm,
      'muacRangeUpperCm': muacRangeUpperCm,
      'estimatedHaz': estimatedHaz,
      'estimatedWhz': estimatedWhz,
    }.entries) {
      if (entry.value != null && !entry.value!.isFinite) {
        throw ArgumentError.value(entry.value, entry.key, 'must be finite');
      }
    }
    _validateRange(
      estimate: estimatedHeightCm,
      lower: heightRangeLowerCm,
      upper: heightRangeUpperCm,
      name: 'heightRangeCm',
    );
    _validateRange(
      estimate: estimatedWeightKg,
      lower: weightRangeLowerKg,
      upper: weightRangeUpperKg,
      name: 'weightRangeKg',
    );
    _validateRange(
      estimate: estimatedMuacCm,
      lower: muacRangeLowerCm,
      upper: muacRangeUpperCm,
      name: 'muacRangeCm',
    );
    _validateClassifierOutput(
      experimentalOverallCategory,
      this.componentProbabilities,
    );
  }

  final String resultUuid;
  final int version;
  final String? supersedesResultUuid;
  final double? estimatedHeightCm;
  final double? estimatedWeightKg;
  final double? estimatedMuacCm;
  final String? heightSource;
  final String? weightSource;
  final String? muacSource;
  final double? heightRangeLowerCm;
  final double? heightRangeUpperCm;
  final double? weightRangeLowerKg;
  final double? weightRangeUpperKg;
  final double? muacRangeLowerCm;
  final double? muacRangeUpperCm;
  final double? estimatedHaz;
  final double? estimatedWhz;
  final String? estimatedStuntingStatus;
  final String? estimatedWastingStatus;
  final String? experimentalOverallCategory;
  final Map<String, double> componentProbabilities;
  final Map<String, Object?> bodyProportionFeatures;
  final Map<String, Object?> captureQualitySummary;
  final String method;
  final String modelVersion;
  final String manifestChecksum;
  final String trainingDataLabel;
  final DateTime createdAt;

  bool get nonClinical => true;

  bool get usesLegacyPopulationHeight =>
      heightSource == legacyWhoHeightSourceV1;

  bool get usesLegacyPopulationWeight =>
      weightSource == legacyWhoWeightSourceV1;

  double? get reportableHeightCm => estimatedHeightCm;

  double? get reportableWeightKg => estimatedWeightKg;

  double? get reportableMuacCm => estimatedMuacCm;

  double? get reportableHaz => usesLegacyPopulationHeight ? null : estimatedHaz;

  double? get reportableWhz =>
      usesLegacyPopulationHeight || usesLegacyPopulationWeight
          ? null
          : estimatedWhz;

  String? get reportableStuntingStatus =>
      usesLegacyPopulationHeight ? null : estimatedStuntingStatus;

  String? get reportableWastingStatus =>
      usesLegacyPopulationHeight || usesLegacyPopulationWeight
          ? null
          : estimatedWastingStatus;

  double? get classificationConfidence {
    final category = experimentalOverallCategory;
    return category == null ? null : componentProbabilities[category];
  }

  double? get captureQuality {
    final value = captureQualitySummary['overall'];
    return value is num && value.isFinite ? value.toDouble() : null;
  }

  List<String> get usedViews {
    final value = captureQualitySummary['used_views'];
    return value is List
        ? value.whereType<String>().toList(growable: false)
        : const [];
  }

  Map<String, Object?> toJson() => {
        'result_uuid': resultUuid,
        'version': version,
        'supersedes_result_uuid': supersedesResultUuid,
        'estimated_height_cm': estimatedHeightCm,
        'estimated_weight_kg': estimatedWeightKg,
        'estimated_muac_cm': estimatedMuacCm,
        'height_source': heightSource,
        'weight_source': weightSource,
        'muac_source': muacSource,
        'height_range_lower_cm': heightRangeLowerCm,
        'height_range_upper_cm': heightRangeUpperCm,
        'weight_range_lower_kg': weightRangeLowerKg,
        'weight_range_upper_kg': weightRangeUpperKg,
        'muac_range_lower_cm': muacRangeLowerCm,
        'muac_range_upper_cm': muacRangeUpperCm,
        'estimated_haz': estimatedHaz,
        'estimated_whz': estimatedWhz,
        'estimated_stunting_status': estimatedStuntingStatus,
        'estimated_wasting_status': estimatedWastingStatus,
        'experimental_overall_category': experimentalOverallCategory,
        'component_probabilities': componentProbabilities,
        'body_proportion_features': bodyProportionFeatures,
        'capture_quality_summary': captureQualitySummary,
        'method': method,
        'model_version': modelVersion,
        'manifest_checksum': manifestChecksum,
        'training_data_label': trainingDataLabel,
        'non_clinical': true,
        'created_at': createdAt.toIso8601String(),
      };

  factory CameraScreeningResult.fromJson(Map<String, Object?> json) {
    return CameraScreeningResult(
      resultUuid: json['result_uuid'] as String,
      version: json['version'] as int,
      supersedesResultUuid: json['supersedes_result_uuid'] as String?,
      estimatedHeightCm: (json['estimated_height_cm'] as num?)?.toDouble(),
      estimatedWeightKg: (json['estimated_weight_kg'] as num?)?.toDouble(),
      estimatedMuacCm: (json['estimated_muac_cm'] as num?)?.toDouble(),
      heightSource: json['height_source'] as String?,
      weightSource: json['weight_source'] as String?,
      muacSource: json['muac_source'] as String?,
      heightRangeLowerCm: (json['height_range_lower_cm'] as num?)?.toDouble(),
      heightRangeUpperCm: (json['height_range_upper_cm'] as num?)?.toDouble(),
      weightRangeLowerKg: (json['weight_range_lower_kg'] as num?)?.toDouble(),
      weightRangeUpperKg: (json['weight_range_upper_kg'] as num?)?.toDouble(),
      muacRangeLowerCm: (json['muac_range_lower_cm'] as num?)?.toDouble(),
      muacRangeUpperCm: (json['muac_range_upper_cm'] as num?)?.toDouble(),
      estimatedHaz: (json['estimated_haz'] as num?)?.toDouble(),
      estimatedWhz: (json['estimated_whz'] as num?)?.toDouble(),
      estimatedStuntingStatus: json['estimated_stunting_status'] as String?,
      estimatedWastingStatus: json['estimated_wasting_status'] as String?,
      experimentalOverallCategory:
          json['experimental_overall_category'] as String?,
      componentProbabilities: _doubleMap(json['component_probabilities']),
      bodyProportionFeatures: _objectMap(json['body_proportion_features']),
      captureQualitySummary: _objectMap(json['capture_quality_summary']),
      method: json['method'] as String,
      modelVersion: json['model_version'] as String,
      manifestChecksum: json['manifest_checksum'] as String,
      trainingDataLabel: json['training_data_label'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }

  static void _validateClassifierOutput(
    String? category,
    Map<String, double> probabilities,
  ) {
    if (category == null && probabilities.isEmpty) return;
    if (category == null ||
        !cameraClassifierCategories.contains(category) ||
        probabilities.keys.toSet().length !=
            cameraClassifierCategories.length ||
        !probabilities.keys.toSet().containsAll(cameraClassifierCategories)) {
      throw ArgumentError('Classifier category and all probabilities required');
    }
    final values = probabilities.values;
    if (values.any((value) => !value.isFinite || value < 0 || value > 1)) {
      throw ArgumentError.value(
        probabilities,
        'componentProbabilities',
        'probabilities must be finite values from 0 to 1',
      );
    }
    final sum = values.fold<double>(0, (total, value) => total + value);
    if ((sum - 1).abs() > 0.02) {
      throw ArgumentError.value(
        probabilities,
        'componentProbabilities',
        'probabilities must sum to 1',
      );
    }
    final highest = probabilities.entries.reduce(
      (left, right) => left.value >= right.value ? left : right,
    );
    if (highest.key != category) {
      throw ArgumentError('Classifier category must match probability argmax');
    }
  }

  static void _validateRange({
    required double? estimate,
    required double? lower,
    required double? upper,
    required String name,
  }) {
    if (lower == null && upper == null) return;
    if (estimate == null ||
        lower == null ||
        upper == null ||
        !lower.isFinite ||
        !upper.isFinite ||
        lower > estimate ||
        estimate > upper) {
      throw ArgumentError('$name must contain its finite estimate');
    }
  }

  static Map<String, double> _doubleMap(Object? value) {
    if (value is! Map) return const {};
    return {
      for (final entry in value.entries)
        entry.key.toString(): (entry.value as num).toDouble(),
    };
  }

  static Map<String, Object?> _objectMap(Object? value) {
    if (value is! Map) return const {};
    return {
      for (final entry in value.entries) entry.key.toString(): entry.value,
    };
  }
}
