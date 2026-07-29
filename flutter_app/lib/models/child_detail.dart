class ChildDetail {
  ChildDetail({
    required this.id,
    required this.name,
    required this.dateOfBirth,
    required this.sex,
    this.guardianName,
    this.location,
    required this.visits,
  });

  final int id;
  final String name;
  final String dateOfBirth;
  final String sex;
  final String? guardianName;
  final String? location;
  final List<ChildVisit> visits;

  factory ChildDetail.fromJson(Map<String, dynamic> json) {
    return ChildDetail(
      id: json['id'] as int,
      name: json['name'] as String,
      dateOfBirth: json['date_of_birth'] as String,
      sex: json['sex'] as String,
      guardianName: json['guardian_name'] as String?,
      location: json['location'] as String?,
      visits: (json['visits'] as List<dynamic>? ?? const [])
          .map((v) => ChildVisit.fromJson(v as Map<String, dynamic>))
          .toList(),
    );
  }
}

class ChildVisit {
  ChildVisit({
    required this.visitId,
    this.localUuid,
    this.visitDate,
    this.ageMonths,
    this.entryMethod,
    this.captureState,
    this.cameraResultSummary,
    this.hasMeasuredReport = false,
    this.requiredAssetAcknowledgement = const {},
    this.requiredAssetsAcknowledged = false,
    this.mediaDeletedAt,
    this.measurement,
  });

  final int visitId;
  final String? localUuid;
  final String? visitDate;
  final double? ageMonths;
  final String? entryMethod;
  final String? captureState;
  final CameraResultSummary? cameraResultSummary;
  final bool hasMeasuredReport;
  final Map<String, String> requiredAssetAcknowledgement;
  final bool requiredAssetsAcknowledged;
  final String? mediaDeletedAt;
  final ChildVisitMeasurement? measurement;

  factory ChildVisit.fromJson(Map<String, dynamic> json) {
    final acknowledgement = json['required_asset_acknowledgement'];
    return ChildVisit(
      visitId: json['visit_id'] as int,
      localUuid: json['local_uuid'] as String?,
      visitDate: json['visit_date'] as String?,
      ageMonths: (json['age_months'] as num?)?.toDouble(),
      entryMethod: json['entry_method'] as String?,
      captureState: json['capture_state'] as String?,
      cameraResultSummary: json['camera_result_summary'] == null
          ? null
          : CameraResultSummary.fromJson(
              json['camera_result_summary'] as Map<String, dynamic>,
            ),
      hasMeasuredReport: json['has_measured_report'] as bool? ?? false,
      requiredAssetAcknowledgement: acknowledgement is Map
          ? {
              for (final entry in acknowledgement.entries)
                entry.key.toString(): entry.value.toString(),
            }
          : const {},
      requiredAssetsAcknowledged:
          json['required_assets_acknowledged'] as bool? ?? false,
      mediaDeletedAt: json['media_deleted_at'] as String?,
      measurement: json['measurement'] == null
          ? null
          : ChildVisitMeasurement.fromJson(
              json['measurement'] as Map<String, dynamic>,
            ),
    );
  }
}

class CameraResultSummary {
  const CameraResultSummary({
    required this.resultUuid,
    required this.version,
    this.estimatedHeightCm,
    this.estimatedWeightKg,
    this.estimatedStuntingStatus,
    this.estimatedWastingStatus,
    this.experimentalOverallCategory,
    required this.method,
    required this.modelVersion,
    required this.nonClinical,
  });

  final String resultUuid;
  final int version;
  final double? estimatedHeightCm;
  final double? estimatedWeightKg;
  final String? estimatedStuntingStatus;
  final String? estimatedWastingStatus;
  final String? experimentalOverallCategory;
  final String method;
  final String modelVersion;
  final bool nonClinical;

  factory CameraResultSummary.fromJson(Map<String, dynamic> json) {
    return CameraResultSummary(
      resultUuid: json['result_uuid'] as String,
      version: json['version'] as int,
      estimatedHeightCm: (json['estimated_height_cm'] as num?)?.toDouble(),
      estimatedWeightKg: (json['estimated_weight_kg'] as num?)?.toDouble(),
      estimatedStuntingStatus: json['estimated_stunting_status'] as String?,
      estimatedWastingStatus: json['estimated_wasting_status'] as String?,
      experimentalOverallCategory:
          json['experimental_overall_category'] as String?,
      method: json['method'] as String,
      modelVersion: json['model_version'] as String,
      nonClinical: json['non_clinical'] as bool? ?? true,
    );
  }
}

class ChildVisitMeasurement {
  ChildVisitMeasurement({
    this.predictedHeightCm,
    this.predictedWeightKg,
    this.heightMethod,
    this.weightMethod,
    this.muacCm,
    this.muacMethod,
    this.hazZscore,
    this.whzZscore,
    this.hazStatus,
    this.whzStatus,
    this.confidenceScore,
  });

  final double? predictedHeightCm;
  final double? predictedWeightKg;
  final String? heightMethod;
  final String? weightMethod;
  final double? muacCm;
  final String? muacMethod;
  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? confidenceScore;

  factory ChildVisitMeasurement.fromJson(Map<String, dynamic> json) {
    return ChildVisitMeasurement(
      predictedHeightCm: (json['manual_height_cm'] as num?)?.toDouble() ??
          (json['effective_height_cm'] as num?)?.toDouble() ??
          (json['predicted_height_cm'] as num?)?.toDouble(),
      predictedWeightKg: (json['manual_weight_kg'] as num?)?.toDouble() ??
          (json['effective_weight_kg'] as num?)?.toDouble() ??
          (json['predicted_weight_kg'] as num?)?.toDouble(),
      heightMethod: json['height_method'] as String?,
      weightMethod: json['weight_method'] as String?,
      muacCm: (json['muac_cm'] as num?)?.toDouble(),
      muacMethod: json['muac_method'] as String?,
      hazZscore: (json['haz_zscore'] as num?)?.toDouble(),
      whzZscore: (json['whz_zscore'] as num?)?.toDouble(),
      hazStatus: json['haz_status'] as String?,
      whzStatus: json['whz_status'] as String?,
      confidenceScore: (json['confidence_score'] as num?)?.toDouble(),
    );
  }
}
