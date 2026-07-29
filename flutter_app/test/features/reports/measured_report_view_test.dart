import 'package:child_growth_monitor_app/features/reports/providers/visit_report_provider.dart';
import 'package:child_growth_monitor_app/features/reports/widgets/measured_report_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('renders the three measurement-based report sections',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: MeasuredReportView(
              report: const MeasuredReportSnapshot(
                heightCm: 83.6,
                weightKg: 11,
                muacCm: 12.4,
                hazZscore: -2.01,
                hazStatus: 'Moderate Stunting',
                whzZscore: -1.2,
                whzStatus: 'Normal',
                muacStatus: 'MAM',
                muacEligible: true,
                oedema: 'no',
                whoAcuteStatus: 'MAM',
                whoAcuteTriggeredBy: ['muac'],
                poshanStatus: 'MAM',
                poshanTriggeredBy: ['MUAC'],
                poshanComplete: true,
                classificationMethod: 'poshan_setu_v1',
              ),
              visitDate: DateTime(2026, 7, 29),
              onEditMeasuredDetails: () {},
            ),
          ),
        ),
      ),
    );

    expect(find.text('Measurement-based Growth Report'), findsOneWidget);
    expect(find.text('WHO HAZ stunting'), findsOneWidget);
    expect(find.text('WHO acute malnutrition'), findsOneWidget);
    expect(find.text('Poshan Setu v1'), findsOneWidget);
    expect(find.text('Moderate Stunting'), findsOneWidget);
    expect(find.text('MAM'), findsNWidgets(2));
    expect(find.textContaining('HAZ -2.01'), findsOneWidget);
  });

  testWidgets('missing measured components are honest and non-clinical',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: MeasuredReportView(
              report: const MeasuredReportSnapshot(
                oedema: 'not_checked',
                whoAcuteStatus: 'UNKNOWN',
                whoAcuteTriggeredBy: [],
                poshanStatus: 'Indeterminate',
                poshanTriggeredBy: [],
                poshanComplete: false,
                classificationMethod: 'poshan_setu_v1',
              ),
              visitDate: DateTime(2026, 7, 29),
              onEditMeasuredDetails: () {},
            ),
          ),
        ),
      ),
    );

    expect(find.text('Not measured'), findsWidgets);
    expect(find.text('Normal'), findsNothing);
    expect(find.textContaining('Indeterminate'), findsNothing);
    expect(find.textContaining('UNKNOWN'), findsNothing);
  });
}
