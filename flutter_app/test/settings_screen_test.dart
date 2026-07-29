import 'package:child_growth_monitor_app/features/guided_capture/services/guided_sync_service.dart';
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

  @override
  Future<int> totalUsedBytes() async => bytesUsed;
}

class _FakeGuidedSyncGateway implements GuidedSyncGateway {
  _FakeGuidedSyncGateway(this.storage);

  final _FakeImageStorageService storage;
  bool cleaned = false;
  GuidedMediaStatus status = const GuidedMediaStatus(
    acknowledged: 1,
    pending: 2,
    failed: 1,
    deletionRequested: 1,
  );

  @override
  Future<int> cleanupAcknowledgedMedia(int ownerUserId) async {
    expect(ownerUserId, 7);
    cleaned = true;
    storage.bytesUsed = 0;
    status = const GuidedMediaStatus(
      acknowledged: 0,
      pending: 2,
      failed: 1,
      deletionRequested: 1,
    );
    return 1;
  }

  @override
  Future<GuidedMediaStatus> mediaStatus(int ownerUserId) async => status;

  @override
  Future<void> requestMediaDeletion({
    required int ownerUserId,
    required String visitUuid,
    required String assetUuid,
  }) async {}

  @override
  Future<void> runOnce(int ownerUserId) async {}
}

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
  });

  testWidgets('shows operational settings and validates the server URL', (
    tester,
  ) async {
    final storage = _FakeImageStorageService();
    final guidedSync = _FakeGuidedSyncGateway(storage);
    await _pumpSettings(tester, storage, guidedSync);

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
    expect(find.text('Acknowledged media: 1'), findsOneWidget);
    expect(find.text('Pending media: 2'), findsOneWidget);
    expect(find.text('Failed media: 1'), findsOneWidget);
    expect(find.text('Deletion requested: 1'), findsOneWidget);
  });

  testWidgets('reset and clear actions update their local state', (
    tester,
  ) async {
    final storage = _FakeImageStorageService();
    final guidedSync = _FakeGuidedSyncGateway(storage);
    await _pumpSettings(tester, storage, guidedSync);

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

    expect(guidedSync.cleaned, isTrue);
    expect(find.text('Used: 0 B'), findsOneWidget);
    expect(find.text('Acknowledged media: 0'), findsOneWidget);
  });
}

Future<void> _pumpSettings(
  WidgetTester tester,
  ImageStorageService storage,
  GuidedSyncGateway guidedSync,
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
        home: SettingsScreen(
          imageStorageService: storage,
          guidedSyncGateway: guidedSync,
          ownerUserId: 7,
        ),
      ),
    ),
  );
  await tester.pumpAndSettle();
}
