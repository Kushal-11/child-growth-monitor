# Offline Login for Field Testing — Design

**Date:** 2026-06-09
**Status:** Approved (design), pending implementation
**Author:** Brainstormed with Claude

## Goal

Make the Flutter app fully usable offline on field devices so it can be tested
in the field with **no backend reachable**. A hardcoded test credential
(`cgmtester@test.com` / `cgmtester`) must allow login with zero network
dependency. Data is saved locally as it already is. Online support and sync
already exist and are left intact for later use.

## Key Finding: The App Is Already Offline-First

A full trace of the data path shows the assessment and data layers already run
entirely on-device. **The only component that requires the backend is login.**

| Capability | Where | Offline today? |
| --- | --- | --- |
| Assessment pipeline (pose → ML wasting → WHO z-scores → MUAC) | `lib/services/assessment_service.dart` | ✅ 100% on-device (MediaPipe + TFLite + WHO LMS assets) |
| Saving assessments | `assessment_service.dart` → Drift/SQLite | ✅ Local DB write + `syncQueue.enqueue` |
| Children list | `lib/providers/children_provider.dart` | ✅ `StreamProvider` over local DB |
| Child detail & history / growth chart | `children_provider.dart` | ✅ Local DB |
| Manual measurement entry | `lib/screens/child_management/` | ✅ Local DB |
| Sync | `lib/providers/sync_provider.dart`, `lib/services/sync_service.dart` | ✅ Fires only on connectivity; fails gracefully offline; never blocks |
| **Login** | `lib/services/auth_service.dart:62` | ❌ **HTTP POST — cannot return a token offline** |

Therefore this is a **single-seam change**: make login resolve offline. No
database, sync, assessment, ML, or WHO changes are required for the core goal.

## Decisions (from brainstorming)

1. **Login mode:** Offline-first, backend optional. Check the local credential
   first; if no match, fall through to the existing HTTP login so a real
   backend still works when present.
2. **Tester identity:** Fixed field-test identity —
   `AuthUser(id: 9001, username: 'cgmtester@test.com', fullName: 'CGM Field Tester', role: 'field_worker')`.
   Stable `id` so offline data is owner-scoped consistently and can be
   reconciled under a real account later.
3. **Credential store:** A single hardcoded credential in a new
   `local_auth.dart` file. Easy to extend to a small list later by editing one
   place.

## Architecture

### New file: `lib/services/local_auth.dart`

The offline credential source. Pure Dart, no I/O, fully unit-testable.

- One hardcoded credential constant: `cgmtester@test.com` / `cgmtester`.
- The fixed identity above.
- `LocalAuth.tryLogin(String username, String password) → AuthLoginResult?`:
  - On exact match (case-insensitive username, exact password) → returns an
    `AuthLoginResult` whose `user` is the fixed identity and whose `token` is a
    synthetic local token, e.g. `'local-9001'` (a stable, clearly-non-server
    sentinel; documented in code as not backend-valid).
  - Otherwise → `null`.
- Username comparison trims and lowercases to tolerate keyboard auto-capitalize
  on `@test.com`.

Rationale for returning the existing `AuthLoginResult` type: it lets
`AuthService` reuse its existing `_persist()` path unchanged, so the restored
session and router gating behave identically to an online login.

### Modified: `AuthService.login()` (`lib/services/auth_service.dart`)

Becomes offline-first. New control flow:

1. `final local = LocalAuth.tryLogin(username, password);`
   - If non-null → `await _persist(local); return local;` **No network call is
     made.**
2. If null → execute the **existing** HTTP path verbatim (POST
   `/api/v1/auth/login`, 200 → persist+return, 401 → `AuthException`, timeouts
   and `ClientException` → `AuthException`).

The public method signature, the `AuthService` constructor, and all other
methods (`readToken`, `readUser`, `logout`, `_persist`) are unchanged. This
matters because `test/login_screen_test.dart` uses a
`_FakeAuthService implements AuthService` — keeping the interface stable keeps
that fake valid.

### Unchanged but relevant

- `auth_provider.dart` `restore()` already treats "token present ⇒
  authenticated" in an offline-tolerant way (`auth_provider.dart:37`). A
  restored local session works across app restarts with no change.
- `main.dart` already calls `restore()` on startup and starts the sync trigger.
- Router gating (`router.dart`) already routes authenticated users to `/` and
  keeps `/settings` reachable pre-login.

## Owner-Scoping (separable follow-on, included by default)

Children are created with `ownerUserId` nullable and currently unset
(`children_table.dart:10`). To tag the tester's data correctly (and make it
cleanly reconcilable when real sync is enabled), stamp `ownerUserId` with the
logged-in user's id (`9001`) at child-creation time.

- Touch points: `AssessmentService.runAssessment` (the `findOrCreate` call) and
  the child-create path in `lib/screens/child_management/child_form_screen.dart`
  / `ChildDao`.
- `AssessmentService` is a plain class with no Riverpod access, so the current
  user id must be **passed in** as a parameter (e.g. `ownerUserId`) by the
  caller, which reads it from `authProvider`. Do not reach into a provider from
  inside the service.

**This is separable.** If we want this PR to be strictly "just login," split
owner-scoping into a follow-up. Default plan: include it, but as its own commit
so it can be dropped independently.

## What Stays Untouched

- **Sync service / queue / connectivity trigger.** The synthetic local token
  simply rides along. When the app is later pointed at a real server, that
  server will reject the token with `401`, which already triggers
  `onTokenRejected()` → `logout()` cleanly (`sync_service.dart:164`,
  `sync_provider.dart:21`). No silent failure — consistent with the project's
  "no silent failures in the assessment/sync pipeline" rule.
- **Assessment, ML, WHO, MUAC pipelines.** Zero changes.
- **Online login path.** Preserved as the fallback for non-local credentials.

## Error Handling

- Offline login with the correct credential: succeeds with no network I/O.
- Offline login with a wrong credential: falls through to HTTP, which fails with
  a network `AuthException` ("Network error during login" / "Login timed out").
  This is acceptable — an unknown credential offline genuinely cannot be
  verified. The existing login screen already renders `AuthException.message`.
- Corrupt secure storage on restore: already falls back to `unauthenticated`
  (`auth_provider.dart:46`). Unchanged.

## Testing

Extend the existing suite (do not duplicate it):

- **`test/local_auth_test.dart` (new), unit:**
  - Correct credential → returns the fixed identity (id 9001, role
    `field_worker`) and a non-empty token.
  - Correct username, wrong password → `null`.
  - Unknown username → `null`.
  - Username case/whitespace variations (`CGMTester@Test.com `) → still matches.
- **`test/auth_service_test.dart` (extend), unit:**
  - `login()` with the local credential returns without touching the injected
    `http.Client` (assert the client is never called — use a client that throws
    if invoked).
  - `login()` with a non-local credential still issues the HTTP POST.
- **`test/login_screen_test.dart` (extend), widget:**
  - Existing "shows error on bad login" test stays.
  - New: entering the tester credential against a fake offline service lands the
    user authenticated (router would redirect to `/`). Reuse the
    `authServiceProvider` override pattern already in the file.

All Flutter tests run with `cd flutter_app && flutter test`. CI must stay green.

## Risks / Honest Caveats

- **The password is compiled into the app.** This is acceptable for a
  field-test build but is **not a real secret**. Documented in code via a
  comment and here. Before any production/public release, this credential must
  be removed or gated behind a debug/flavor flag.
- **The synthetic token is not backend-valid.** Sync against a real server will
  `401` until a real login replaces the session — by design, and handled
  gracefully (see "What Stays Untouched").
- No new dependencies, no schema migration, no model regeneration.

## Out of Scope (future work)

- Real offline-capable auth (e.g. cached server-issued tokens with expiry).
- Multiple field-tester accounts or an on-device "add tester" screen.
- Reconciliation/merge of locally-created `ownerUserId 9001` data into a real
  user account on first real login.
- Any backend changes.

## Acceptance Criteria

1. With **no backend reachable**, a fresh install can log in with
   `cgmtester@test.com` / `cgmtester` and reach the home screen.
2. After login, a full assessment (photo → result) and a manual measurement can
   be completed and appear in the children list/history — all offline.
3. App restart keeps the user logged in offline.
4. Pointing the app at a real server still allows normal online login with
   other credentials, and the offline session's sync attempts fail gracefully
   (401 → re-login prompt) rather than crashing or silently dropping data.
5. `flutter test` passes, including the new/extended auth tests.
