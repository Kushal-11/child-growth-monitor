# Settings Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Settings tab (third bottom nav item) where users can configure the API base URL and test the connection, replacing the debug-only URL field currently on the assessment screen.

**Architecture:** New `SettingsScreen` widget reads/writes the base URL via existing `api_provider.dart` functions. The assessment screen's debug URL field and health check logic are removed since they move to settings. The bottom nav gains a third tab.

**Tech Stack:** Flutter, Riverpod, SharedPreferences (existing), GoRouter (existing)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `lib/screens/settings/settings_screen.dart` | Create | Settings screen with URL field, Save & Test, connection status |
| `lib/l10n/translations.dart` | Modify | Add settings translation keys |
| `lib/router.dart` | Modify | Add `/settings` route |
| `lib/screens/shared/app_scaffold.dart` | Modify | Add third NavigationDestination |
| `lib/screens/assessment/assessment_screen.dart` | Modify | Remove debug URL field and health check |

---

### Task 1: Add translation keys

**Files:**
- Modify: `lib/l10n/translations.dart:254` (append before closing brace)

- [ ] **Step 1: Add settings translation entries**

Add these entries at the end of the `translations` map, before the closing `};`:

```dart
  'nav_settings': {'en': 'Settings', 'mr': 'सेटिंग्ज'},
  'settings_heading': {'en': 'Settings', 'mr': 'सेटिंग्ज'},
  'server_connection': {'en': 'Server Connection', 'mr': 'सर्व्हर कनेक्शन'},
  'save_and_test': {'en': 'Save & Test', 'mr': 'सेव्ह आणि तपासा'},
  'reset_default': {'en': 'Reset to Default', 'mr': 'डीफॉल्ट वर रीसेट'},
  'connected_ms': {'en': 'Connected', 'mr': 'कनेक्ट'},
  'connection_failed': {'en': 'Connection failed', 'mr': 'कनेक्शन अयशस्वी'},
  'invalid_url': {
    'en': 'Invalid URL — must be a private IP or approved host',
    'mr': 'अवैध URL — खाजगी IP किंवा मान्य होस्ट असणे आवश्यक',
  },
```

- [ ] **Step 2: Verify no duplicate keys**

Run: `cd flutter_app && grep -c "nav_settings\|settings_heading\|server_connection\|save_and_test\|reset_default\|connected_ms\|connection_failed\|invalid_url" lib/l10n/translations.dart`

Expected: each key appears exactly once.

- [ ] **Step 3: Commit**

```bash
git add flutter_app/lib/l10n/translations.dart
git commit -m "feat: add settings screen translation keys"
```

---

### Task 2: Create SettingsScreen widget

**Files:**
- Create: `lib/screens/settings/settings_screen.dart`

- [ ] **Step 1: Create the settings screen file**

Create `flutter_app/lib/screens/settings/settings_screen.dart`:

```dart
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/api_provider.dart';
import '../shared/app_scaffold.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  final _urlController = TextEditingController();
  bool _loading = false;
  bool? _healthy;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadUrl();
  }

  Future<void> _loadUrl() async {
    final saved = await loadSavedBaseUrl();
    if (!mounted) return;
    _urlController.text = saved;
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _saveAndTest() async {
    final url = _urlController.text.trim();
    if (!isAllowlistedHost(url)) {
      setState(() {
        _healthy = false;
        _error = t('invalid_url', ref);
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _healthy = null;
    });

    try {
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;
      final healthy = await ref.read(apiProvider).checkHealth();
      if (!mounted) return;
      setState(() => _healthy = healthy);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _healthy = false;
        _error = e.toString();
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _resetToDefault() {
    _urlController.text = 'http://10.0.2.2:8000';
    setState(() {
      _healthy = null;
      _error = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AppScaffold(
      currentIndex: 2,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            t('settings_heading', ref),
            style: theme.textTheme.headlineSmall,
          ),
          const SizedBox(height: 24),

          // Server Connection card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                Text(
                  t('server_connection', ref),
                  style: theme.textTheme.titleMedium,
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _urlController,
                  decoration: InputDecoration(
                    labelText: t('api_base_url', ref),
                    border: const OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.url,
                ),
                const SizedBox(height: 12),

                // Status indicator
                if (_loading)
                  const Padding(
                    padding: EdgeInsets.only(bottom: 12),
                    child: LinearProgressIndicator(),
                  ),
                if (_healthy != null && !_loading)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Row(
                      children: [
                        Icon(
                          _healthy! ? Icons.check_circle : Icons.error,
                          color: _healthy! ? Colors.green : Colors.red,
                          size: 18,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            _healthy!
                                ? t('connected_ms', ref)
                                : _error ?? t('connection_failed', ref),
                            style: TextStyle(
                              color: _healthy! ? Colors.green : Colors.red,
                              fontSize: 13,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                // Buttons
                Row(
                  children: [
                    Expanded(
                      child: FilledButton(
                        onPressed: _loading ? null : _saveAndTest,
                        child: Text(t('save_and_test', ref)),
                      ),
                    ),
                    const SizedBox(width: 8),
                    TextButton(
                      onPressed: _loading ? null : _resetToDefault,
                      child: Text(t('reset_default', ref)),
                    ),
                  ],
                ),
              ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 2: Verify the file compiles**

Run: `cd flutter_app && flutter analyze lib/screens/settings/settings_screen.dart`

Expected: No issues found.

- [ ] **Step 3: Commit**

```bash
git add flutter_app/lib/screens/settings/settings_screen.dart
git commit -m "feat: add SettingsScreen with API URL config and health check"
```

---

### Task 3: Add /settings route

**Files:**
- Modify: `lib/router.dart`

- [ ] **Step 1: Add the import and route**

Add the import at the top of `router.dart`:

```dart
import 'screens/settings/settings_screen.dart';
```

Add this route inside the `routes` list, after the `/children/:id` route:

```dart
    GoRoute(
      path: '/settings',
      builder: (context, state) => const SettingsScreen(),
    ),
```

- [ ] **Step 2: Commit**

```bash
git add flutter_app/lib/router.dart
git commit -m "feat: add /settings route"
```

---

### Task 4: Add Settings tab to bottom navigation

**Files:**
- Modify: `lib/screens/shared/app_scaffold.dart`

- [ ] **Step 1: Add third NavigationDestination**

In `app_scaffold.dart`, update the `onDestinationSelected` switch to handle index 2:

```dart
        onDestinationSelected: (index) {
          switch (index) {
            case 0:
              context.go('/');
            case 1:
              context.go('/children');
            case 2:
              context.go('/settings');
          }
        },
```

Add the third destination to the `destinations` list:

```dart
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.assessment),
            label: t('nav_assess', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.people),
            label: t('nav_children', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.settings),
            label: t('nav_settings', ref),
          ),
        ],
```

- [ ] **Step 2: Add the l10n import if not present**

The file already imports `l10n_provider.dart`, so no change needed. Verify `t('nav_settings', ref)` resolves — it was added in Task 1.

- [ ] **Step 3: Commit**

```bash
git add flutter_app/lib/screens/shared/app_scaffold.dart
git commit -m "feat: add Settings tab to bottom navigation"
```

---

### Task 5: Remove debug URL field from AssessmentScreen

**Files:**
- Modify: `lib/screens/assessment/assessment_screen.dart`

- [ ] **Step 1: Remove `_baseUrlController` and `_healthy` state**

In `_AssessmentScreenState`, remove these declarations:

```dart
  final _baseUrlController = TextEditingController();
```

```dart
  bool? _healthy;
```

Remove the `_baseUrlController.dispose();` line from `dispose()`.

Remove the entire `_initBaseUrl()` method (lines 57–62):

```dart
  Future<void> _initBaseUrl() async {
    final saved = await loadSavedBaseUrl();
    if (!mounted) return;
    _baseUrlController.text = saved;
    ref.read(baseUrlProvider.notifier).state = saved;
  }
```

Remove the `_initBaseUrl();` call from `initState()`.

Remove the entire `_checkHealth()` method (lines 128–149):

```dart
  Future<void> _checkHealth() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final url = _baseUrlController.text.trim();
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;
      final healthy = await ref.read(apiProvider).checkHealth();
      if (!mounted) return;
      setState(() => _healthy = healthy);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _healthy = false;
        _error = e.toString();
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }
```

- [ ] **Step 2: Remove the debug URL widget block from build()**

Remove the entire `if (kDebugMode) ...[ ]` block (lines 224–260) that contains the `TextFormField` for `_baseUrlController` and the `_healthy` status indicator.

- [ ] **Step 3: Remove URL save logic from `_submit()`**

In the `_submit()` method, remove these lines (around lines 170–172):

```dart
      final url = _baseUrlController.text.trim();
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;
```

The `_submit()` method should go straight to calling `ref.read(apiProvider).submitAssessment(...)`.

- [ ] **Step 4: Clean up unused imports**

Remove `import 'package:flutter/foundation.dart';` if `kDebugMode` is no longer used anywhere in the file.

Remove `import '../../providers/api_provider.dart';` only if `apiProvider` is no longer referenced (it's still used in `_submit()` for `ref.read(apiProvider)`, so keep it).

- [ ] **Step 5: Load saved URL on app startup**

The settings screen now owns URL persistence. But we need the saved URL loaded into `baseUrlProvider` on app startup so the assessment screen uses it. In `lib/main.dart`, update:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/api_provider.dart';
import 'router.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final savedUrl = await loadSavedBaseUrl();
  runApp(
    ProviderScope(
      overrides: [
        baseUrlProvider.overrideWith((ref) => savedUrl),
      ],
      child: const ChildGrowthApp(),
    ),
  );
}

class ChildGrowthApp extends StatelessWidget {
  const ChildGrowthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'SNEH Growth Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      routerConfig: appRouter,
    );
  }
}
```

- [ ] **Step 6: Run flutter analyze**

Run: `cd flutter_app && flutter analyze`

Expected: No issues found.

- [ ] **Step 7: Commit**

```bash
git add flutter_app/lib/screens/assessment/assessment_screen.dart flutter_app/lib/main.dart
git commit -m "refactor: move API URL config from assessment screen to settings"
```

---

### Task 6: Manual testing

- [ ] **Step 1: Start the backend**

```bash
cd /storage/projects/child-growth-monitor/child-growth-monitor-main
PYTHONPATH=. .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 2: Run the Flutter app on device**

```bash
cd flutter_app
flutter run -d 192.168.0.160:35961
```

- [ ] **Step 3: Verify settings screen**

1. Tap the Settings tab in bottom nav
2. Verify the URL field shows the saved base URL
3. Enter `http://192.168.0.119:8000` and tap **Save & Test**
4. Verify green "Connected" status appears
5. Enter an invalid URL like `http://evil.com:8000` and tap **Save & Test**
6. Verify red "Invalid URL" error appears
7. Tap **Reset to Default** — verify field resets to `http://10.0.2.2:8000`
8. Navigate to Assess tab — verify assessment works with the saved URL
