import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/local_auth.dart';

void main() {
  group('LocalAuth.tryLogin', () {
    test('correct credential returns fixed field-test identity', () {
      final result = LocalAuth.tryLogin('cgmtester@test.com', 'cgmtester');
      expect(result, isNotNull);
      expect(result!.user.id, 9001);
      expect(result.user.username, 'cgmtester@test.com');
      expect(result.user.fullName, 'CGM Field Tester');
      expect(result.user.role, 'field_worker');
      expect(result.token, 'local-9001');
    });

    test('username is matched case-insensitively and trimmed', () {
      final result = LocalAuth.tryLogin('  CGMTester@Test.com  ', 'cgmtester');
      expect(result, isNotNull);
      expect(result!.user.id, 9001);
    });

    test('correct username with wrong password returns null', () {
      expect(LocalAuth.tryLogin('cgmtester@test.com', 'wrong'), isNull);
    });

    test('unknown username returns null', () {
      expect(LocalAuth.tryLogin('someone@else.com', 'cgmtester'), isNull);
    });

    test('password is case-sensitive', () {
      expect(LocalAuth.tryLogin('cgmtester@test.com', 'CGMTESTER'), isNull);
    });

    test('disabled (release build) returns null even for the correct '
        'credential', () {
      // In release builds the gate defaults to off (kDebugMode == false); the
      // backdoor must not authenticate anyone.
      expect(
        LocalAuth.tryLogin('cgmtester@test.com', 'cgmtester', enabled: false),
        isNull,
      );
    });

    test('explicitly enabled returns the field-test identity', () {
      final result =
          LocalAuth.tryLogin('cgmtester@test.com', 'cgmtester', enabled: true);
      expect(result, isNotNull);
      expect(result!.user.id, 9001);
    });
  });
}
