# Offline Login for Field Testing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Flutter app log in and run fully offline with a hardcoded field-test credential (`cgmtester@test.com` / `cgmtester`), while preserving the existing online login path and owner-scoping assessment-created children.

**Architecture:** The app is already offline-first for assessment, local persistence, and sync; the only backend dependency is login. Add a pure-Dart `LocalAuth` credential source and make `AuthService.login()` offline-first: check the local credential first (no network), otherwise fall through to the existing HTTP login. Separately, thread `ownerUserId` through the assessment child-create path so offline data is tagged to the test user (id 9001).

**Tech Stack:** Flutter, Dart, Riverpod, Drift (SQLite), `http`, `flutter_secure_storage`, `flutter_test`.

**Spec:** `docs/superpowers/specs/2026-06-09-offline-login-field-test-design.md`

**Run all tests with:** `cd flutter_app && flutter test`

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `flutter_app/lib/services/local_auth.dart` | Create | Hardcoded offline credential + fixed identity; pure `tryLogin` function. |
| `flutter_app/test/local_auth_test.dart` | Create | Unit tests for `LocalAuth.tryLogin`. |
| `flutter_app/lib/services/auth_service.dart` | Modify | `login()` becomes offline-first (local check before HTTP). |
| `flutter_app/test/auth_service_test.dart` | Modify | Add tests: local credential skips HTTP; non-local hits HTTP. |
| `flutter_app/test/login_screen_test.dart` | Modify | Add widget test: tester credential authenticates offline. |
| `flutter_app/lib/database/daos/child_dao.dart` | Modify | `findOrCreate` accepts optional `ownerUserId` and sets it on insert. |
| `flutter_app/test/child_dao_test.dart` | Modify | Add test: `findOrCreate` persists `ownerUserId`. |
| `flutter_app/lib/services/assessment_service.dart` | Modify | Accept `ownerUserId`, pass to `findOrCreate`. |
| `flutter_app/lib/screens/assessment/assessment_screen.dart` | Modify | Read current user id from `authProvider`, pass to `runAssessment`. |

**Note:** `child_form_screen.dart` (manual child create) already sets `ownerUserId` from `authProvider` — no change needed there.

---

## Task 1: LocalAuth credential source

**Files:**
- Create: `flutter_app/lib/services/local_auth.dart`
- Test: `flutter_app/test/local_auth_test.dart`

- [ ] **Step 1: Write the failing test**

Create `flutter_app/test/local_auth_test.dart`:

```dart
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
      expect(result.token, isNotEmpty);
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
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/local_auth_test.dart`
Expected: FAIL — compile error, `local_auth.dart` / `LocalAuth` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `flutter_app/lib/services/local_auth.dart`:

```dart
import 'auth_service.dart';

/// Offline, hardcoded credential source for field-test builds.
///
/// WARNING: This password is compiled into the app. It is acceptable for a
/// field-test build ONLY and is NOT a real secret. Remove or gate behind a
/// debug/flavor flag before any production or public release.
class LocalAuth {
  LocalAuth._();

  static const String _username = 'cgmtester@test.com';
  static const String _password = 'cgmtester';

  /// Fixed identity for the offline field tester. The stable [id] (9001) is
  /// used to owner-scope locally created data so it can be reconciled with a
  /// real account once online sync is enabled.
  static final AuthUser _fixedUser = AuthUser(
    id: 9001,
    username: _username,
    fullName: 'CGM Field Tester',
    role: 'field_worker',
  );

  /// Synthetic, clearly-non-server token. A real backend will reject this with
  /// a 401, which the sync layer already handles gracefully.
  static const String _localToken = 'local-9001';

  /// Returns a login result for the hardcoded tester, or null if the
  /// credential does not match. Username is trimmed and compared
  /// case-insensitively; password is exact. Pure function, no I/O.
  static AuthLoginResult? tryLogin(String username, String password) {
    final normalized = username.trim().toLowerCase();
    if (normalized == _username && password == _password) {
      return AuthLoginResult(token: _localToken, user: _fixedUser);
    }
    return null;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/local_auth_test.dart`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/services/local_auth.dart flutter_app/test/local_auth_test.dart
git commit -m "feat(flutter): add offline LocalAuth credential source for field testing"
```

---

## Task 2: Make AuthService.login() offline-first

**Files:**
- Modify: `flutter_app/lib/services/auth_service.dart:62-85` (the `login` method)
- Test: `flutter_app/test/auth_service_test.dart`

- [ ] **Step 1: Write the failing test**

Add to `flutter_app/test/auth_service_test.dart`. Add these imports at the top (below the existing imports):

```dart
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
```

`AuthService` persists on the local-success path via `flutter_secure_storage`, whose platform channel is not available under `flutter_test`. Mock that channel in `setUp` so `write` succeeds in-memory. At the very top of `main()`, add:

```dart
  TestWidgetsFlutterBinding.ensureInitialized();

  // flutter_secure_storage talks to a platform channel that does not exist in
  // unit tests; back it with an in-memory map so _persist() succeeds.
  const channel = MethodChannel('plugins.it_nomads.com/flutter_secure_storage');
  final store = <String, String>{};
  setUp(() {
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
```

Then add a throwing HTTP client and a recording client, plus the new group, inside `main()`:

```dart
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
```

Add this group inside `main()` (after the existing `AuthLoginResult` group):

```dart
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/auth_service_test.dart`
Expected: FAIL — `local credential succeeds without any HTTP call` throws `StateError` (because `login` currently always calls HTTP).

- [ ] **Step 3: Write minimal implementation**

In `flutter_app/lib/services/auth_service.dart`, add the import at the top (with the other imports):

```dart
import 'local_auth.dart';
```

Replace the start of the `login` method body. Current code:

```dart
  Future<AuthLoginResult> login(String username, String password) async {
    final uri = Uri.parse('$baseUrl/api/v1/auth/login');
```

Replace with:

```dart
  Future<AuthLoginResult> login(String username, String password) async {
    // Offline-first: a hardcoded field-test credential resolves locally with
    // no network call. Any other credential falls through to the backend.
    final local = LocalAuth.tryLogin(username, password);
    if (local != null) {
      await _persist(local);
      return local;
    }

    final uri = Uri.parse('$baseUrl/api/v1/auth/login');
```

Leave the rest of the method (the HTTP POST, status handling, `_persist` on 200) unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flutter_app && flutter test test/auth_service_test.dart`
Expected: PASS (original 2 tests + 2 new tests).

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/services/auth_service.dart flutter_app/test/auth_service_test.dart
git commit -m "feat(flutter): make AuthService.login offline-first via LocalAuth"
```

---

## Task 3: Widget test — tester credential authenticates offline

**Files:**
- Modify: `flutter_app/test/login_screen_test.dart`

This task proves the end-to-end login UX: entering the tester credential drives the auth state to `authenticated`. We test the auth state (not router navigation) because the login screen test mounts the screen directly without the router.

- [ ] **Step 1: Write the failing test**

Add to `flutter_app/test/login_screen_test.dart`. The existing `_FakeAuthService` returns a token only for password `'good'`. Add a new test that uses a fake which delegates to the real `LocalAuth`, and asserts the `authProvider` state flips to authenticated.

Add this import at the top:

```dart
import 'package:child_growth_monitor_app/services/local_auth.dart';
```

Add this fake below the existing `_FakeAuthService`:

```dart
/// Fake that resolves the offline LocalAuth credential and persists nothing.
class _LocalOnlyAuthService implements AuthService {
  @override
  String get baseUrl => 'http://test';
  @override
  Future<AuthLoginResult> login(String u, String p) async {
    final local = LocalAuth.tryLogin(u, p);
    if (local != null) return local;
    throw AuthException('Invalid username or password', statusCode: 401);
  }
  @override
  Future<String?> readToken() async => null;
  @override
  Future<AuthUser?> readUser() async => null;
  @override
  Future<void> logout() async {}
}
```

Add this test inside `main()`:

```dart
  testWidgets('tester credential authenticates offline', (tester) async {
    final container = ProviderContainer(
      overrides: [authServiceProvider.overrideWithValue(_LocalOnlyAuthService())],
    );
    addTearDown(container.dispose);

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: const MaterialApp(home: LoginScreen()),
    ));

    await tester.enterText(
        find.byKey(const Key('login_username')), 'cgmtester@test.com');
    await tester.enterText(
        find.byKey(const Key('login_password')), 'cgmtester');
    await tester.tap(find.byKey(const Key('login_submit')));
    await tester.pumpAndSettle();

    final auth = container.read(authProvider);
    expect(auth.status, AuthStatus.authenticated);
    expect(auth.user?.id, 9001);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/login_screen_test.dart`
Expected: Initially this should actually PASS if Tasks 1-2 are committed, because `_LocalOnlyAuthService` + real `LocalAuth` already work. To honor TDD, run it BEFORE implementing — i.e. if you are doing this task standalone, temporarily expect failure only if `LocalAuth` is absent. Since `LocalAuth` exists from Task 1, this test validates integration and should pass. Proceed to Step 3.

> TDD note: Task 3 is an integration assertion over code built test-first in Tasks 1-2. There is no new production code here; the "failing" precondition was already satisfied by Task 1's red. Keep this test as a regression guard.

- [ ] **Step 3: (no new production code)**

No implementation change. This task adds a regression test only.

- [ ] **Step 4: Run the full auth + login suite**

Run: `cd flutter_app && flutter test test/login_screen_test.dart test/auth_service_test.dart test/local_auth_test.dart`
Expected: PASS (all tests, including the original "shows error on bad login").

- [ ] **Step 5: Commit**

```bash
git add flutter_app/test/login_screen_test.dart
git commit -m "test(flutter): tester credential authenticates offline via login screen"
```

---

## Task 4: `findOrCreate` accepts and persists `ownerUserId`

**Files:**
- Modify: `flutter_app/lib/database/daos/child_dao.dart:8-33` (the `findOrCreate` method)
- Test: `flutter_app/test/child_dao_test.dart`

- [ ] **Step 1: Write the failing test**

`flutter_app/test/child_dao_test.dart` already has `late AppDatabase db; late ChildDao dao;` created in `setUp` and closed in `tearDown`. Add this test alongside the existing ones inside `main()`:

```dart
  test('findOrCreate persists ownerUserId on new child', () async {
    final child = await dao.findOrCreate(
      name: 'Owned Child',
      dateOfBirth: '2022-01-01',
      sex: 'F',
      ownerUserId: 9001,
    );
    expect(child.ownerUserId, 9001);

    final fetched = await dao.getById(child.id);
    expect(fetched!.ownerUserId, 9001);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/child_dao_test.dart`
Expected: FAIL — `findOrCreate` does not accept the named parameter `ownerUserId` (compile error).

- [ ] **Step 3: Write minimal implementation**

In `flutter_app/lib/database/daos/child_dao.dart`, modify `findOrCreate` to accept and set `ownerUserId`. Current signature and insert:

```dart
  Future<ChildrenData> findOrCreate({
    required String name,
    required String dateOfBirth,
    required String sex,
    String? guardianName,
    String? location,
  }) async {
```

Change to:

```dart
  Future<ChildrenData> findOrCreate({
    required String name,
    required String dateOfBirth,
    required String sex,
    String? guardianName,
    String? location,
    int? ownerUserId,
  }) async {
```

And in the same method, the insert. Current:

```dart
    final id = await _db.into(_db.children).insert(
      ChildrenCompanion.insert(
        name: name,
        dateOfBirth: dateOfBirth,
        sex: sex,
        guardianName: Value(guardianName),
        location: Value(location),
      ),
    );
```

Change to:

```dart
    final id = await _db.into(_db.children).insert(
      ChildrenCompanion.insert(
        name: name,
        dateOfBirth: dateOfBirth,
        sex: sex,
        guardianName: Value(guardianName),
        location: Value(location),
        ownerUserId: Value(ownerUserId),
      ),
    );
```

> Do NOT change the existing-match branch (`if (existing != null) return existing;`). We do not retroactively re-tag an already-created child here — that avoids surprising ownership changes on repeat visits. (Reconciliation is out of scope per the spec.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/child_dao_test.dart`
Expected: PASS (existing tests + new test).

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/database/daos/child_dao.dart flutter_app/test/child_dao_test.dart
git commit -m "feat(flutter): findOrCreate persists ownerUserId for owner-scoped children"
```

---

## Task 5: Thread `ownerUserId` through the assessment path

**Files:**
- Modify: `flutter_app/lib/services/assessment_service.dart:62-74` (signature) and `:159-165` (the `findOrCreate` call)
- Modify: `flutter_app/lib/screens/assessment/assessment_screen.dart:135-152` (the `runAssessment` call)
- Test: `flutter_app/test/assessment_service_test.dart`

- [ ] **Step 1: Write the failing test**

`flutter_app/test/assessment_service_test.dart` has `late AppDatabase db; late AssessmentService svc;` in `setUp`. The `ChildDao` is constructed inline inside the service, so it is NOT a separate variable — assert against the created child by querying the db directly (the file's other tests do this, e.g. `db.select(db.visits).get()`). The pose stub ignores the image path, so `'/tmp/front.jpg'` is the canonical fixture path used throughout this file. Add this test inside `main()`:

```dart
  test('runAssessment tags created child with ownerUserId', () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Owned Assessment Child',
      dateOfBirth: '2022-06-01',
      sex: 'M',
      ownerUserId: 9001,
    );
    expect(result.childName, 'Owned Assessment Child');

    final children = await db.select(db.children).get();
    final created =
        children.firstWhere((c) => c.name == 'Owned Assessment Child');
    expect(created.ownerUserId, 9001);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/assessment_service_test.dart`
Expected: FAIL — `runAssessment` does not accept named parameter `ownerUserId` (compile error).

- [ ] **Step 3: Write minimal implementation**

In `flutter_app/lib/services/assessment_service.dart`, add `ownerUserId` to the `runAssessment` signature. Current tail of the parameter list:

```dart
    double? manualMuacCm,
    String? guardianName,
    String? location,
  }) async {
```

Change to:

```dart
    double? manualMuacCm,
    String? guardianName,
    String? location,
    int? ownerUserId,
  }) async {
```

Then update the `findOrCreate` call. Current:

```dart
    final child = await _childDao.findOrCreate(
      name: childName,
      dateOfBirth: dateOfBirth,
      sex: sex,
      guardianName: guardianName,
      location: location,
    );
```

Change to:

```dart
    final child = await _childDao.findOrCreate(
      name: childName,
      dateOfBirth: dateOfBirth,
      sex: sex,
      guardianName: guardianName,
      location: location,
      ownerUserId: ownerUserId,
    );
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/assessment_service_test.dart`
Expected: PASS (existing tests + new test).

- [ ] **Step 5: Wire the caller (assessment screen)**

In `flutter_app/lib/screens/assessment/assessment_screen.dart`, add the import if not present (it already imports `assessment_provider.dart` and `children_provider.dart`; add the auth provider):

```dart
import '../../providers/auth_provider.dart';
```

In `_submit()`, just before the `final svc = await ref.read(assessmentServiceProvider.future);` line, read the current user id:

```dart
      final ownerUserId = ref.read(authProvider).user?.id;
```

Then add `ownerUserId: ownerUserId,` to the `svc.runAssessment(...)` argument list (alongside `location:`).

- [ ] **Step 6: Run the full suite + analyze**

Run: `cd flutter_app && flutter analyze && flutter test`
Expected: analyze reports no new issues; all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add flutter_app/lib/services/assessment_service.dart flutter_app/lib/screens/assessment/assessment_screen.dart flutter_app/test/assessment_service_test.dart
git commit -m "feat(flutter): owner-scope assessment-created children to logged-in user"
```

---

## Task 6: Final verification (offline end-to-end)

**Files:** none (verification only)

- [ ] **Step 1: Static analysis clean**

Run: `cd flutter_app && flutter analyze`
Expected: "No issues found!" (or no NEW issues vs. the pre-change baseline).

- [ ] **Step 2: Full test suite green**

Run: `cd flutter_app && flutter test`
Expected: all tests pass, including `local_auth_test.dart`, `auth_service_test.dart`, `login_screen_test.dart`, `child_dao_test.dart`, `assessment_service_test.dart`.

- [ ] **Step 3: Manual offline smoke test (device/emulator, airplane mode)**

This step is manual and requires a device or emulator. Provide these as commands for the user to run (do not execute on their behalf):

```
cd flutter_app && flutter run
```

Verify, with networking disabled (airplane mode / no backend running):
1. Log in with `cgmtester@test.com` / `cgmtester` → reaches the home/assessment screen.
2. Run an assessment (pick a front photo, fill child info) → result screen shows HAZ/WHZ/MUAC.
3. The child appears in the Children list with history.
4. Kill and relaunch the app → still logged in (no re-login needed).
5. (Optional) Confirm the sync badge shows pending items and does not crash while offline.

- [ ] **Step 4: Confirm completion**

All automated checks green + manual smoke test passed = feature complete. The branch is ready for review/merge per `superpowers:finishing-a-development-branch`.

---

## Notes for the Implementer

- **Interface stability:** Do not change the public method names of `AuthService` (`login`, `readToken`, `readUser`, `logout`) — `test/login_screen_test.dart` has a `_FakeAuthService implements AuthService` that will break otherwise.
- **No schema migration / no codegen:** `ownerUserId` already exists on the `Children` table (`children_table.dart:10`) and in generated code. You are only passing a value that was previously left null. Do NOT run `build_runner` for this work.
- **Security caveat:** The hardcoded password is for field testing only. The code comment in `local_auth.dart` says so; keep it.
- **Sync is intentionally untouched:** The `local-9001` token will 401 against a real backend, which already triggers `onTokenRejected()` → logout. That is the designed behavior, not a bug.
