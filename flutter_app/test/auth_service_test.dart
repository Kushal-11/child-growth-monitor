import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// HTTP client that fails the test if any request is sent. Used to prove the
/// local-credential path never touches the network.
class _ThrowingClient extends http.BaseClient {
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    throw StateError('HTTP must not be called for local credential');
  }
}

/// HTTP client that records that it was called and returns a fixed 401.
class _RecordingClient extends http.BaseClient {
  bool called = false;
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    called = true;
    return http.StreamedResponse(
      Stream.value(<int>[]),
      401,
    );
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // flutter_secure_storage talks to a platform channel that does not exist in
  // unit tests; back it with an in-memory map so _persist() succeeds.
  const channel = MethodChannel('plugins.it_nomads.com/flutter_secure_storage');
  final store = <String, String>{};
  setUp(() {
    store.clear(); // isolate each test from prior writes
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      switch (call.method) {
        case 'write':
          store[call.arguments['key'] as String] =
              call.arguments['value'] as String;
          return null;
        case 'read':
          return store[call.arguments['key'] as String];
        case 'delete':
          store.remove(call.arguments['key'] as String);
          return null;
        case 'readAll':
          return Map<String, String>.from(store);
        case 'deleteAll':
          store.clear();
          return null;
        case 'containsKey':
          return store.containsKey(call.arguments['key'] as String);
        default:
          return null;
      }
    });
  });
  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

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

  group('AuthService.login offline-first', () {
    test('local credential succeeds without any HTTP call', () async {
      final service = AuthService(
        baseUrl: 'http://unused',
        storage: const FlutterSecureStorage(),
        httpClient: _ThrowingClient(),
      );
      final result = await service.login('cgmtester@test.com', 'cgmtester');
      expect(result.user.id, 9001);
      expect(result.user.role, 'field_worker');
    });

    test('non-local credential falls through to HTTP', () async {
      final client = _RecordingClient();
      final service = AuthService(
        baseUrl: 'http://unused',
        storage: const FlutterSecureStorage(),
        httpClient: client,
      );
      await expectLater(
        service.login('asha', 'somepassword'),
        throwsA(isA<AuthException>()),
      );
      expect(client.called, isTrue);
    });
  });
}
