import 'package:child_growth_monitor_app/providers/sync_provider.dart';
import 'package:child_growth_monitor_app/screens/settings/settings_screen.dart';
import 'package:child_growth_monitor_app/services/image_storage_service.dart';
import 'package:child_growth_monitor_app/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _FakeImageStorageService extends ImageStorageService {
  int bytesUsed = 2048;
  bool cleared = false;

  @override
  Future<int> totalUsedBytes() async => bytesUsed;

  @override
  Future<void> clearAll() async {
    cleared = true;
    bytesUsed = 0;
  }
}

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
  });

  testWidgets('shows operational settings and validates the server URL', (
    tester,
  ) async {
    final storage = _FakeImageStorageService();
    await _pumpSettings(tester, storage);

    expect(
      find.text('Keep the app ready for offline field work.'),
      findsOneWidget,
    );
    expect(find.text('Server Connection'), findsOneWidget);
    expect(find.text('Pending: 3'), findsOneWidget);

    await tester.enterText(
      find.byKey(const Key('settings_base_url')),
      'https://evil.example',
    );
    await tester.tap(find.byKey(const Key('settings_save_test')));
    await tester.pump();

    expect(
      find.text('Invalid URL — must be a private IP or approved host'),
      findsOneWidget,
    );

    await tester.drag(find.byType(ListView), const Offset(0, -600));
    await tester.pumpAndSettle();
    expect(find.text('Used: 2.0 KB'), findsOneWidget);
  });

  testWidgets('reset and clear actions update their local state', (
    tester,
  ) async {
    final storage = _FakeImageStorageService();
    await _pumpSettings(tester, storage);

    await tester.enterText(
      find.byKey(const Key('settings_base_url')),
      'http://192.168.1.20:8000',
    );
    await tester.tap(find.byKey(const Key('settings_reset_default')));
    await tester.pump();
    final urlField = tester.widget<TextFormField>(
      find.byKey(const Key('settings_base_url')),
    );
    expect(urlField.controller?.text, 'http://10.0.2.2:8000');

    await tester.drag(find.byType(ListView), const Offset(0, -600));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('settings_clear_images')));
    await tester.pumpAndSettle();

    expect(storage.cleared, isTrue);
    expect(find.text('Used: 0 B'), findsOneWidget);
  });
}

Future<void> _pumpSettings(
  WidgetTester tester,
  ImageStorageService storage,
) async {
  await tester.binding.setSurfaceSize(const Size(390, 844));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await tester.pumpWidget(
    ProviderScope(
      overrides: [
        pendingSyncCountProvider.overrideWith((ref) => Stream.value(3)),
      ],
      child: MaterialApp(
        theme: AppTheme.light(),
        home: SettingsScreen(imageStorageService: storage),
      ),
    ),
  );
  await tester.pumpAndSettle();
}
