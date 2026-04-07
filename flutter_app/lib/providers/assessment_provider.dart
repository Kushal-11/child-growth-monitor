import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/assessment_result.dart';

/// Holds the latest assessment result. Set after successful submission,
/// read by ResultScreen. Cleared when starting a new assessment.
final assessmentResultProvider = StateProvider<AssessmentResult?>((ref) => null);
