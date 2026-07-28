class AssessmentResult {
  AssessmentResult({
    required this.childName,
    required this.sex,
    required this.ageMonths,
    required this.summary,
    required this.measurement,
    required this.nutrition,
    this.mlPrediction,
    this.muac,
    required this.combinedNutrition,
  });

  final String childName;
  final String sex;
  final double ageMonths;
  final String summary;
  final Measurement measurement;
  final Nutrition nutrition;
  final MlPrediction? mlPrediction;
  final MuacDetail? muac;
  final CombinedNutritionDetail combinedNutrition;

  factory AssessmentResult.fromJson(Map<String, dynamic> json) {
    return AssessmentResult(
      childName: json['child_name'] as String,
      sex: json['sex'] as String,
      ageMonths: (json['age_months'] as num).toDouble(),
      summary: json['summary'] as String,
      measurement: Measurement.fromJson(
        json['measurement'] as Map<String, dynamic>? ?? {},
      ),
      nutrition: Nutrition.fromJson(
        json['nutrition'] as Map<String, dynamic>? ?? {},
      ),
      mlPrediction: json['ml_prediction'] == null
          ? null
          : MlPrediction.fromJson(
              json['ml_prediction'] as Map<String, dynamic>,
            ),
      muac: json['muac'] == null
          ? null
          : MuacDetail.fromJson(json['muac'] as Map<String, dynamic>),
      combinedNutrition: CombinedNutritionDetail.fromJson(
        json['combined_nutrition'] as Map<String, dynamic>? ?? const {},
      ),
    );
  }
}

class CombinedNutritionDetail {
  const CombinedNutritionDetail({
    required this.status,
    this.triggeredBy = const [],
    this.rationale = '',
    this.method = 'who_muac_whz_or_rule',
    this.confidenceScore,
  });

  final String status;
  final List<String> triggeredBy;
  final String rationale;
  final String method;
  final double? confidenceScore;

  factory CombinedNutritionDetail.fromJson(Map<String, dynamic> json) {
    return CombinedNutritionDetail(
      status: json['status'] as String? ?? 'UNKNOWN',
      triggeredBy: (json['triggered_by'] as List<dynamic>? ?? const [])
          .map((value) => value.toString())
          .toList(),
      rationale: json['rationale'] as String? ?? '',
      method: json['method'] as String? ?? 'who_muac_whz_or_rule',
      confidenceScore: (json['confidence_score'] as num?)?.toDouble(),
    );
  }
}

class Measurement {
  Measurement({
    this.effectiveHeightCm,
    this.heightMethod = 'unavailable',
    this.predictedHeightCm,
    this.predictedWeightKg,
    this.manualHeightCm,
    this.manualWeightKg,
    this.confidenceScore,
    this.bodyBuild,
    this.estimationMethod,
    this.sideViewUsed = false,
    this.chestDepthCm,
    this.abdDepthCm,
  });

  final double? effectiveHeightCm;
  final String heightMethod;
  final double? predictedHeightCm;
  final double? predictedWeightKg;
  final double? manualHeightCm;
  final double? manualWeightKg;
  final double? confidenceScore;
  final String? bodyBuild;
  final String? estimationMethod;
  final bool sideViewUsed;
  final double? chestDepthCm;
  final double? abdDepthCm;

  factory Measurement.fromJson(Map<String, dynamic> json) {
    return Measurement(
      effectiveHeightCm: (json['effective_height_cm'] as num?)?.toDouble(),
      heightMethod: json['height_method'] as String? ?? 'unavailable',
      predictedHeightCm: (json['predicted_height_cm'] as num?)?.toDouble(),
      predictedWeightKg: (json['predicted_weight_kg'] as num?)?.toDouble(),
      manualHeightCm: (json['manual_height_cm'] as num?)?.toDouble(),
      manualWeightKg: (json['manual_weight_kg'] as num?)?.toDouble(),
      confidenceScore: (json['confidence_score'] as num?)?.toDouble(),
      bodyBuild: json['body_build'] as String?,
      estimationMethod: json['estimation_method'] as String?,
      sideViewUsed: json['side_view_used'] as bool? ?? false,
      chestDepthCm: (json['chest_depth_cm'] as num?)?.toDouble(),
      abdDepthCm: (json['abd_depth_cm'] as num?)?.toDouble(),
    );
  }
}

class Nutrition {
  Nutrition({
    this.hazZscore,
    this.whzZscore,
    this.hazStatus,
    this.whzStatus,
    this.ageMonths,
  });

  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? ageMonths;

  factory Nutrition.fromJson(Map<String, dynamic> json) {
    return Nutrition(
      hazZscore: (json['haz_zscore'] as num?)?.toDouble(),
      whzZscore: (json['whz_zscore'] as num?)?.toDouble(),
      hazStatus: json['haz_status'] as String?,
      whzStatus: json['whz_status'] as String?,
      ageMonths: (json['age_months'] as num?)?.toDouble(),
    );
  }
}

class MlPrediction {
  MlPrediction({
    this.estimatedWeightKg,
    this.samProbability,
    this.mamProbability,
    this.normalProbability,
    this.riskProbability,
    this.overweightProbability,
    this.wastingStatus,
  });

  final double? estimatedWeightKg;
  final double? samProbability;
  final double? mamProbability;
  final double? normalProbability;
  final double? riskProbability;
  final double? overweightProbability;
  final String? wastingStatus;

  factory MlPrediction.fromJson(Map<String, dynamic> json) {
    return MlPrediction(
      estimatedWeightKg: (json['estimated_weight_kg'] as num?)?.toDouble(),
      samProbability: (json['sam_probability'] as num?)?.toDouble(),
      mamProbability: (json['mam_probability'] as num?)?.toDouble(),
      normalProbability: (json['normal_probability'] as num?)?.toDouble(),
      riskProbability: (json['risk_probability'] as num?)?.toDouble(),
      overweightProbability: (json['overweight_probability'] as num?)
          ?.toDouble(),
      wastingStatus: json['wasting_status'] as String?,
    );
  }
}

class MuacDetail {
  MuacDetail({
    this.muacCm,
    this.muacStatus,
    this.muacMethod,
    this.ageInRange,
    this.confidence,
    this.uncertaintyLowerCm,
    this.uncertaintyUpperCm,
    this.modelVersion,
    this.calibrationVersion,
    this.isDirectMeasurement = false,
    this.requiresConfirmation = false,
    this.referralGuidance,
  });

  final double? muacCm;
  final String? muacStatus;
  final String? muacMethod;
  final bool? ageInRange;
  final double? confidence;
  final double? uncertaintyLowerCm;
  final double? uncertaintyUpperCm;
  final String? modelVersion;
  final String? calibrationVersion;
  final bool isDirectMeasurement;
  final bool requiresConfirmation;
  final String? referralGuidance;

  factory MuacDetail.fromJson(Map<String, dynamic> json) {
    return MuacDetail(
      muacCm: (json['muac_cm'] as num?)?.toDouble(),
      muacStatus: json['muac_status'] as String?,
      muacMethod: json['muac_method'] as String?,
      ageInRange: json['age_in_range'] as bool?,
      confidence: (json['confidence'] as num?)?.toDouble(),
      uncertaintyLowerCm: (json['uncertainty_lower_cm'] as num?)?.toDouble(),
      uncertaintyUpperCm: (json['uncertainty_upper_cm'] as num?)?.toDouble(),
      modelVersion: json['model_version'] as String?,
      calibrationVersion: json['calibration_version'] as String?,
      isDirectMeasurement: json['is_direct_measurement'] as bool? ?? false,
      requiresConfirmation: json['requires_confirmation'] as bool? ?? false,
      referralGuidance: json['referral_guidance'] as String?,
    );
  }
}
