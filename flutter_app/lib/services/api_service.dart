import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/assessment_result.dart';
import '../models/child.dart';
import '../models/child_detail.dart';

class ApiException implements Exception {
  ApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

class ApiService {
  ApiService({required this.baseUrl});

  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<bool> checkHealth() async {
    final response = await http.get(_uri('/api/v1/health'));
    return response.statusCode == 200;
  }

  Future<List<ChildSummary>> getChildren() async {
    final response = await http.get(_uri('/api/v1/children'));
    if (response.statusCode != 200) {
      throw _apiError(response, fallback: 'Failed to load children');
    }

    final decoded = jsonDecode(response.body) as List<dynamic>;
    return decoded
        .map((item) => ChildSummary.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<ChildDetail> getChildDetail(int childId) async {
    final response = await http.get(_uri('/api/v1/children/$childId'));
    if (response.statusCode != 200) {
      throw _apiError(response, fallback: 'Failed to load child detail');
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return ChildDetail.fromJson(decoded);
  }

  Future<AssessmentResult> submitAssessment({
    required String frontImagePath,
    String? sideImagePath,
    String? backImagePath,
    required String childName,
    required String dateOfBirth,
    required String sex,
    double? weightKg,
    double? heightCm,
    double? heightValue,
    String heightUnit = 'cm',
    double? muacCm,
    String? guardianName,
    String? location,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/api/v1/assess'));

    request.files.add(
      await http.MultipartFile.fromPath('image', frontImagePath),
    );

    if (sideImagePath != null && sideImagePath.isNotEmpty) {
      request.files.add(
        await http.MultipartFile.fromPath('image_side', sideImagePath),
      );
    }

    if (backImagePath != null && backImagePath.isNotEmpty) {
      request.files.add(
        await http.MultipartFile.fromPath('image_back', backImagePath),
      );
    }

    request.fields['child_name'] = childName;
    request.fields['date_of_birth'] = dateOfBirth;
    request.fields['sex'] = sex;

    if (weightKg != null) {
      request.fields['weight_kg'] = weightKg.toString();
    }
    if (heightCm != null) {
      request.fields['height_cm'] = heightCm.toString();
    }
    if (heightValue != null) {
      request.fields['height_value'] = heightValue.toString();
      request.fields['height_unit'] = heightUnit;
    }
    if (muacCm != null) {
      request.fields['muac_cm'] = muacCm.toString();
    }
    if (guardianName != null && guardianName.isNotEmpty) {
      request.fields['guardian_name'] = guardianName;
    }
    if (location != null && location.isNotEmpty) {
      request.fields['location'] = location;
    }

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw _apiError(response, fallback: 'Assessment failed');
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return AssessmentResult.fromJson(decoded);
  }

  ApiException _apiError(http.Response response, {required String fallback}) {
    String message = '$fallback (${response.statusCode})';

    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic>) {
        if (decoded['detail'] is String) {
          message = decoded['detail'] as String;
        } else if (decoded['message'] is String) {
          message = decoded['message'] as String;
        }
      }
    } catch (_) {
      if (response.body.isNotEmpty) {
        message = '$message: ${response.body}';
      }
    }

    return ApiException(message, statusCode: response.statusCode);
  }
}
