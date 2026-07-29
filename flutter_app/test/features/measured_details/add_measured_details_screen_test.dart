import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/measured_details/domain/measured_details.dart';
import 'package:child_growth_monitor_app/features/measured_details/providers/measured_details_provider.dart';
import 'package:child_growth_monitor_app/features/measured_details/screens/add_measured_details_screen.dart';
import 'package:child_growth_monitor_app/features/measured_details/services/measured_report_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

class FakeMeasuredReportGateway implements MeasuredReportGateway {
  MeasuredDetails? saved;

  @override
  Future<MeasuredVisitContext> loadContext({
    required int ownerUserId,
    required String visitUuid,
  }) async =>
      MeasuredVisitContext(
        visitUuid: '10000000-0000-0000-0000-000000000001',
        ownerUserId: 7,
        childId: 11,
        visitDate: DateTime(2026, 7, 29),
        ageMonths: 30,
        completedAgeMonths: 30,
        sex: 'F',
      );

  @override
  Future<void> save({
    required int ownerUserId,
    required String visitUuid,
    required int editorUserId,
    required MeasuredDetails details,
  }) async {
    saved = details;
  }
}

void main() {
  testWidgets('locks visit date and accepts a height-only follow-up',
      (tester) async {
    tester.view.physicalSize = const Size(800, 1200);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final gateway = FakeMeasuredReportGateway();

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          measuredReportGatewayProvider.overrideWith((ref) async => gateway),
        ],
        child: const MaterialApp(
          home: AddMeasuredDetailsScreen(
            visitUuid: '10000000-0000-0000-0000-000000000001',
            ownerUserId: 7,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('29 Jul 2026'), findsOneWidget);
    expect(find.textContaining('visit date is locked'), findsOneWidget);
    expect(find.text('Create a new visit instead'), findsOneWidget);

    await tester.enterText(
      find.byKey(const Key('measured_height')),
      '83.58',
    );
    await tester.tap(find.text('Save measured details'));
    await tester.pumpAndSettle();

    expect(gateway.saved, isNotNull);
    expect(gateway.saved!.heightCm, 83.58);
    expect(gateway.saved!.weightKg, isNull);
    expect(gateway.saved!.measurementMode, MeasurementMode.standingHeight);
    expect(gateway.saved!.oedema, OedemaStatus.notChecked);
  });

  testWidgets('all-empty measured form is rejected without saving',
      (tester) async {
    tester.view.physicalSize = const Size(800, 1200);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final gateway = FakeMeasuredReportGateway();

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          measuredReportGatewayProvider.overrideWith((ref) async => gateway),
        ],
        child: const MaterialApp(
          home: AddMeasuredDetailsScreen(
            visitUuid: '10000000-0000-0000-0000-000000000001',
            ownerUserId: 7,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('Save measured details'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Enter at least one measured detail'),
        findsOneWidget);
    expect(gateway.saved, isNull);
  });
}
