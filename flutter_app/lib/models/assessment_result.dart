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
  });

  final String childName;
  final String sex;
  final double ageMonths;
  final String summary;
  final Measurement measurement;
  final Nutrition nutrition;
  final MlPrediction? mlPrediction;
  final MuacDetail? muac;

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
          : MlPrediction.fromJson(json['ml_prediction'] as Map<String, dynamic>),
      muac: json['muac'] == null
          ? null
          : MuacDetail.fromJson(json['muac'] as Map<String, dynamic>),
    );
  }
}

class Measurement {
  Measurement({
    this.predictedHeightCm,
    this.predictedWeightKg,
    this.manualHeightCm,
    this.manualWeightKg,
    this.confidenceScore,
    this.bodyBuild,
    this.estimationMethod,
    this.effectiveHeightCm,
    this.effectiveWeightKg,
    this.heightMethod,
    this.weightMethod,
    this.heightConfidence,
    this.weightConfidence,
    this.sideViewUsed = false,
    this.chestDepthCm,
    this.abdDepthCm,
  });

  final double? predictedHeightCm;
  final double? predictedWeightKg;
  final double? manualHeightCm;
  final double? manualWeightKg;
  final double? confidenceScore;
  final String? bodyBuild;
  final String? estimationMethod;
  final double? effectiveHeightCm;
  final double? effectiveWeightKg;
  final String? heightMethod;
  final String? weightMethod;
  final double? heightConfidence;
  final double? weightConfidence;
  final bool sideViewUsed;
  final double? chestDepthCm;
  final double? abdDepthCm;

  factory Measurement.fromJson(Map<String, dynamic> json) {
    return Measurement(
      predictedHeightCm: (json['predicted_height_cm'] as num?)?.toDouble(),
      predictedWeightKg: (json['predicted_weight_kg'] as num?)?.toDouble(),
      manualHeightCm: (json['manual_height_cm'] as num?)?.toDouble(),
      manualWeightKg: (json['manual_weight_kg'] as num?)?.toDouble(),
      confidenceScore: (json['confidence_score'] as num?)?.toDouble(),
      bodyBuild: json['body_build'] as String?,
      estimationMethod: json['estimation_method'] as String?,
      effectiveHeightCm: (json['effective_height_cm'] as num?)?.toDouble(),
      effectiveWeightKg: (json['effective_weight_kg'] as num?)?.toDouble(),
      heightMethod: json['height_method'] as String?,
      weightMethod: json['weight_method'] as String?,
      heightConfidence: (json['height_confidence'] as num?)?.toDouble(),
      weightConfidence: (json['weight_confidence'] as num?)?.toDouble(),
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
    this.bmi,
    this.bmiStatus,
    this.combinedStatus,
    this.triggeringIndicators = const [],
    this.rationale,
    this.protocolVersion,
    this.classificationConfidence,
  });

  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? ageMonths;
  final double? bmi;
  final String? bmiStatus;
  final String? combinedStatus;
  final List<String> triggeringIndicators;
  final String? rationale;
  final String? protocolVersion;
  final double? classificationConfidence;

  factory Nutrition.fromJson(Map<String, dynamic> json) {
    return Nutrition(
      hazZscore: (json['haz_zscore'] as num?)?.toDouble(),
      whzZscore: (json['whz_zscore'] as num?)?.toDouble(),
      hazStatus: json['haz_status'] as String?,
      whzStatus: json['whz_status'] as String?,
      ageMonths: (json['age_months'] as num?)?.toDouble(),
      bmi: (json['bmi'] as num?)?.toDouble(),
      bmiStatus: json['bmi_status'] as String?,
      combinedStatus: json['combined_status'] as String?,
      triggeringIndicators: (json['triggering_indicators'] as List<dynamic>? ?? const []).cast<String>(),
      rationale: json['rationale'] as String?,
      protocolVersion: json['protocol_version'] as String?,
      classificationConfidence: (json['classification_confidence'] as num?)?.toDouble(),
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
    this.wastingMethod,
  });

  final double? estimatedWeightKg;
  final double? samProbability;
  final double? mamProbability;
  final double? normalProbability;
  final double? riskProbability;
  final double? overweightProbability;
  final String? wastingStatus;
  final String? wastingMethod;

  factory MlPrediction.fromJson(Map<String, dynamic> json) {
    return MlPrediction(
      estimatedWeightKg: (json['estimated_weight_kg'] as num?)?.toDouble(),
      samProbability: (json['sam_probability'] as num?)?.toDouble(),
      mamProbability: (json['mam_probability'] as num?)?.toDouble(),
      normalProbability: (json['normal_probability'] as num?)?.toDouble(),
      riskProbability: (json['risk_probability'] as num?)?.toDouble(),
      overweightProbability: (json['overweight_probability'] as num?)?.toDouble(),
      wastingStatus: json['wasting_status'] as String?,
      wastingMethod: json['wasting_method'] as String?,
    );
  }
}

class MuacDetail {
  MuacDetail({
    this.muacCm,
    this.muacStatus,
    this.muacMethod,
    this.ageInRange,
  });

  final double? muacCm;
  final String? muacStatus;
  final String? muacMethod;
  final bool? ageInRange;

  factory MuacDetail.fromJson(Map<String, dynamic> json) {
    return MuacDetail(
      muacCm: (json['muac_cm'] as num?)?.toDouble(),
      muacStatus: json['muac_status'] as String?,
      muacMethod: json['muac_method'] as String?,
      ageInRange: json['age_in_range'] as bool?,
    );
  }
}
