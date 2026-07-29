import 'package:child_growth_monitor_app/models/child_detail.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('parses guided visit timeline fields without weakening legacy fields',
      () {
    final detail = ChildDetail.fromJson({
      'id': 1,
      'name': 'Child 001',
      'date_of_birth': '2024-01-01',
      'sex': 'F',
      'visits': [
        {
          'visit_id': 9,
          'local_uuid': '10000000-0000-0000-0000-000000000001',
          'visit_date': '2026-07-29T00:00:00',
          'age_months': 30,
          'entry_method': 'guided_capture',
          'capture_state': 'estimated_report',
          'has_measured_report': false,
          'camera_result_summary': {
            'result_uuid': '30000000-0000-0000-0000-000000000001',
            'version': 2,
            'estimated_height_cm': 88,
            'estimated_weight_kg': 11,
            'method': 'camera_screening_v1',
            'model_version': 'camera-v2',
            'non_clinical': true,
          },
          'required_asset_acknowledgement': {
            'front': 'acknowledged',
            'side': 'pending',
          },
          'required_assets_acknowledged': false,
          'media_deleted_at': null,
        },
      ],
    });

    final visit = detail.visits.single;
    expect(visit.localUuid, '10000000-0000-0000-0000-000000000001');
    expect(visit.captureState, 'estimated_report');
    expect(visit.hasMeasuredReport, isFalse);
    expect(visit.cameraResultSummary?.version, 2);
    expect(visit.cameraResultSummary?.modelVersion, 'camera-v2');
    expect(visit.requiredAssetAcknowledgement['front'], 'acknowledged');
    expect(visit.requiredAssetAcknowledgement['side'], 'pending');
    expect(visit.requiredAssetsAcknowledged, isFalse);
    expect(visit.mediaDeletedAt, isNull);
  });

  test('legacy visits remain parseable when guided fields are absent', () {
    final detail = ChildDetail.fromJson({
      'id': 1,
      'name': 'Legacy child',
      'date_of_birth': '2024-01-01',
      'sex': 'M',
      'visits': [
        {
          'visit_id': 3,
          'visit_date': '2026-07-29T00:00:00',
          'age_months': 30,
          'measurement': {'predicted_height_cm': 88},
        },
      ],
    });

    final visit = detail.visits.single;
    expect(visit.localUuid, isNull);
    expect(visit.captureState, isNull);
    expect(visit.cameraResultSummary, isNull);
    expect(visit.requiredAssetAcknowledgement, isEmpty);
    expect(visit.measurement?.predictedHeightCm, 88);
  });
}
