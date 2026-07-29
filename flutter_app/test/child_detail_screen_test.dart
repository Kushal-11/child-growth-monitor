import 'package:child_growth_monitor_app/models/child_detail.dart';
import 'package:child_growth_monitor_app/providers/children_provider.dart';
import 'package:child_growth_monitor_app/providers/sync_provider.dart';
import 'package:child_growth_monitor_app/screens/children/child_detail_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

ChildDetail guidedChild(List<ChildVisit> visits) => ChildDetail(
      id: 1,
      name: 'Child 001',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      visits: visits,
    );

ChildVisit guidedVisit(String state) => ChildVisit(
      visitId: 9,
      localUuid: '10000000-0000-0000-0000-000000000001',
      visitDate: '2026-07-29T00:00:00',
      ageMonths: 30,
      entryMethod: 'guided_capture',
      captureState: state,
      hasMeasuredReport: state == 'measured_report',
      requiredAssetAcknowledgement: const {
        'front': 'acknowledged',
        'side': 'pending',
      },
      requiredAssetsAcknowledged: false,
      cameraResultSummary: state == 'estimated_report'
          ? const CameraResultSummary(
              resultUuid: '30000000-0000-0000-0000-000000000001',
              version: 1,
              method: 'camera_screening_v1',
              modelVersion: 'camera-v1',
              nonClinical: true,
            )
          : null,
    );

Future<void> pumpScreen(
  WidgetTester tester, {
  required ChildDetail child,
}) async {
  final router = GoRouter(
    initialLocation: '/children/1',
    routes: [
      GoRoute(
        path: '/children/:id',
        builder: (_, __) => const ChildDetailScreen(childId: 1),
      ),
      GoRoute(
        path: '/visits/:visitUuid/report',
        builder: (_, state) =>
            Scaffold(body: Text('report:${state.pathParameters['visitUuid']}')),
      ),
      GoRoute(
        path: '/visits/:visitUuid/measured-details',
        builder: (_, state) => Scaffold(
          body: Text(
            'measure:${state.pathParameters['visitUuid']}|'
            '${state.uri.queryParameters['visitDate']}',
          ),
        ),
      ),
      GoRoute(
        path: '/visits/:visitUuid/capture',
        builder: (_, state) => Scaffold(
            body: Text('capture:${state.pathParameters['visitUuid']}')),
      ),
      GoRoute(
        path: '/children',
        builder: (_, __) => const Scaffold(body: Text('children')),
      ),
      GoRoute(
        path: '/settings',
        builder: (_, __) => const Scaffold(body: Text('settings')),
      ),
      GoRoute(
        path: '/',
        builder: (_, __) => const Scaffold(body: Text('home')),
      ),
      GoRoute(
        path: '/children/:id/edit',
        builder: (_, __) => const Scaffold(body: Text('edit')),
      ),
      GoRoute(
        path: '/children/:id/measure',
        builder: (_, __) => const Scaffold(body: Text('legacy measure')),
      ),
    ],
  );
  await tester.pumpWidget(
    ProviderScope(
      overrides: [
        childDetailProvider(1).overrideWith((_) => Stream.value(child)),
        pendingSyncCountProvider.overrideWith((_) => Stream.value(0)),
      ],
      child: MaterialApp.router(routerConfig: router),
    ),
  );
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('maps every guided state to the exact timeline label',
      (tester) async {
    await pumpScreen(
      tester,
      child: guidedChild([
        guidedVisit('incomplete_capture'),
        guidedVisit('processing'),
        guidedVisit('estimated_report'),
        guidedVisit('processing_failed'),
        guidedVisit('measured_report'),
      ]),
    );

    expect(find.text('Incomplete capture'), findsOneWidget);
    expect(find.text('Processing estimate'), findsOneWidget);
    expect(find.text('Estimated report'), findsOneWidget);
    expect(find.text('Estimate failed — retry'), findsOneWidget);
    expect(find.text('Measured report added'), findsOneWidget);
  });

  testWidgets('camera-only visit action targets its exact visit UUID and date',
      (tester) async {
    await pumpScreen(
      tester,
      child: guidedChild([guidedVisit('estimated_report')]),
    );

    await tester.ensureVisible(find.text('Add Measured Details'));
    await tester.tap(find.text('Add Measured Details'));
    await tester.pumpAndSettle();

    expect(
      find.text(
        'measure:10000000-0000-0000-0000-000000000001|2026-07-29',
      ),
      findsOneWidget,
    );
  });
}
