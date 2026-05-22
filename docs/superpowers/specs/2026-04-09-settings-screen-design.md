# Settings Screen — Design Spec

**Date**: 2026-04-09
**Scope**: Add a Settings tab to the Flutter app for configuring the API base URL.

## Problem

The API base URL is hardcoded to `http://10.0.2.2:8000` (Android emulator alias), which doesn't work on physical devices. Users must edit source code to change it. A settings screen lets users configure the backend URL at runtime.

## Design

### Navigation

- Add a **third tab** to the bottom navigation bar: **Assess | Children | Settings**
- Icon: `Icons.settings`
- Route: `/settings`

### Settings Screen Content

Single section — **Server Connection**:

1. **Text field** — editable API base URL, pre-filled with the currently saved value
2. **"Save & Test" button** — validates URL, saves to SharedPreferences, pings backend health endpoint
3. **Connection status indicator** — shown inline below the button:
   - Success: green checkmark + response latency (e.g., "Connected — 45ms")
   - Failure: red error message (e.g., "Connection failed: timeout")
   - Loading: circular progress indicator during the test
4. **"Reset to Default" text button** — reverts URL to `http://10.0.2.2:8000` (development default)

### Behavior

- **On screen load**: reads saved URL from SharedPreferences via existing `loadSavedBaseUrl()`
- **On Save & Test**:
  1. Validate against existing `isAllowlistedHost()` — reject with inline error if invalid
  2. Save to SharedPreferences via existing `saveBaseUrl()`
  3. Update `baseUrlProvider` so all API calls immediately use the new URL
  4. Ping `GET /api/v1/health` (or `/docs`) with a 5-second timeout
  5. Display result inline
- **On Reset to Default**: populate field with default URL, do not auto-save (user must tap Save & Test)

### Validation

Uses the existing `isAllowlistedHost()` function which accepts:
- `localhost`, `127.0.0.1`, `10.0.2.2`
- `api.child-growth-monitor.org`
- Private IPv4 ranges: `10.x.x.x`, `172.16-31.x.x`, `192.168.x.x`
- Only `http` and `https` schemes

### Files Changed

| File | Change |
|------|--------|
| `lib/screens/settings/settings_screen.dart` | **New** — settings screen widget |
| `lib/screens/shared/app_scaffold.dart` | Add third `NavigationDestination` (Settings) |
| `lib/router.dart` | Add `/settings` route |
| `lib/l10n/translations.dart` | Add translation keys for settings UI strings |

### Files NOT Changed

- `lib/providers/api_provider.dart` — already has `loadSavedBaseUrl()`, `saveBaseUrl()`, `baseUrlProvider`, and `isAllowlistedHost()`
- `lib/screens/shared/app_scaffold.dart` AppBar actions — language toggle stays as-is

### Translation Keys Needed

| Key | English | Marathi |
|-----|---------|---------|
| `nav_settings` | Settings | सेटिंग्ज |
| `settings_heading` | Settings | सेटिंग्ज |
| `server_connection` | Server Connection | सर्व्हर कनेक्शन |
| `api_base_url` | Backend URL | बॅकएंड URL |
| `save_and_test` | Save & Test | सेव्ह आणि तपासा |
| `reset_default` | Reset to Default | डीफॉल्ट वर रीसेट |
| `connected_ms` | Connected — {ms}ms | कनेक्ट — {ms}ms |
| `connection_failed` | Connection failed | कनेक्शन अयशस्वी |
| `invalid_url` | Invalid URL — must be a private IP or approved host | अवैध URL — खाजगी IP किंवा मान्य होस्ट असणे आवश्यक |
