import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

void main() {
  group('AuthLoginResult', () {
    test('parses from login json', () {
      final json = {
        'access_token': 'tok123',
        'token_type': 'bearer',
        'user': {'id': 1, 'username': 'asha', 'full_name': 'Asha', 'role': 'worker'},
      };
      final result = AuthLoginResult.fromJson(json);
      expect(result.token, 'tok123');
      expect(result.user.username, 'asha');
      expect(result.user.role, 'worker');
    });

    test('AuthUser round-trips through json', () {
      final u = AuthUser(id: 2, username: 'b', fullName: 'B', role: 'admin');
      final back = AuthUser.fromJson(u.toJson());
      expect(back.id, 2);
      expect(back.username, 'b');
      expect(back.role, 'admin');
    });
  });
}
