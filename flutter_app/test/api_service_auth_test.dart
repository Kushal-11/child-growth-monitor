import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/api_service.dart';

void main() {
  test('authToken is included in header builder', () {
    final svc = ApiService(baseUrl: 'http://test', authToken: 'abc');
    expect(svc.authToken, 'abc');
    expect(svc.authHeaders['Authorization'], 'Bearer abc');
  });

  test('no token => empty auth headers', () {
    final svc = ApiService(baseUrl: 'http://test');
    expect(svc.authHeaders.containsKey('Authorization'), false);
  });
}
