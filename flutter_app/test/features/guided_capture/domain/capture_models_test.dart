import 'dart:convert';
import 'dart:io';

import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final contractFile = [
    File('../docs/contracts/guided_capture_v1.json'),
    File('docs/contracts/guided_capture_v1.json'),
  ].firstWhere((file) => file.existsSync());
  final fixture =
      jsonDecode(contractFile.readAsStringSync()) as Map<String, dynamic>;

  test('Dart enum wire values match the language-neutral fixture', () {
    expect(
      CaptureState.values.map((value) => value.wireValue),
      fixture['visit_states'],
    );
    expect(
      CaptureAssetRole.values.map((value) => value.wireValue),
      fixture['asset_roles'],
    );
    expect(
      MeasurementMode.values.map((value) => value.wireValue),
      fixture['measurement_modes'],
    );
    expect(
      OedemaStatus.values.map((value) => value.wireValue),
      fixture['oedema_values'],
    );
    expect(
      CaptureAssetRole.requiredRoles.map((value) => value.wireValue),
      fixture['required_asset_roles'],
    );
  });

  test('all canonical values round-trip and unknown values fail closed', () {
    for (final value in CaptureState.values) {
      expect(CaptureState.fromWire(value.wireValue), value);
    }
    for (final value in CaptureAssetRole.values) {
      expect(CaptureAssetRole.fromWire(value.wireValue), value);
    }
    for (final value in MeasurementMode.values) {
      expect(MeasurementMode.fromWire(value.wireValue), value);
    }
    for (final value in OedemaStatus.values) {
      expect(OedemaStatus.fromWire(value.wireValue), value);
    }

    expect(() => CaptureState.fromWire('unknown'), throwsFormatException);
    expect(() => CaptureAssetRole.fromWire('unknown'), throwsFormatException);
    expect(() => MeasurementMode.fromWire('unknown'), throwsFormatException);
    expect(() => OedemaStatus.fromWire('unknown'), throwsFormatException);
  });

  test('required roles are exactly front and side', () {
    expect(
      CaptureAssetRole.requiredRoles,
      orderedEquals([CaptureAssetRole.front, CaptureAssetRole.side]),
    );
  });

  test('only documented state transitions are allowed', () {
    final expectedTransitions =
        fixture['allowed_transitions'] as Map<String, dynamic>;

    for (final current in CaptureState.values) {
      final expected = (expectedTransitions[current.wireValue] as List<dynamic>)
          .cast<String>()
          .toSet();
      final actual = CaptureState.values
          .where((target) => canTransitionCaptureState(current, target))
          .map((target) => target.wireValue)
          .toSet();
      expect(actual, expected);
    }

    expect(
      () => requireCaptureStateTransition(
        CaptureState.measuredReport,
        CaptureState.estimatedReport,
      ),
      throwsStateError,
    );
  });
}
