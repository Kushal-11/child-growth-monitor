import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../../../database/database.dart';
import '../../../providers/assessment_provider.dart';
import '../../../providers/assessment_service_provider.dart';
import '../../../providers/auth_provider.dart';
import '../../../providers/children_provider.dart';
import '../../../providers/database_provider.dart';

final assessmentChildrenProvider =
    StreamProvider.autoDispose<List<ChildrenData>>((ref) {
      final ownerId = ref.watch(authProvider).user?.id;
      if (ownerId == null) return Stream.value(const <ChildrenData>[]);
      return ref.watch(childDaoProvider).watchForOwner(ownerId);
    });

class AssessmentFormState {
  AssessmentFormState({
    this.step = 0,
    this.selectedChildId,
    this.childName = '',
    String? dateOfBirth,
    this.ageMonths = '',
    this.useDob = true,
    this.sex = 'M',
    this.guardianName = '',
    this.location = '',
    this.weightKg = '',
    this.heightValue = '',
    this.heightUnit = 'cm',
    this.muacCm = '',
    this.frontImagePath,
    this.sideImagePath,
    this.backImagePath,
    this.showValidationErrors = false,
    this.isSubmitting = false,
    this.error,
    this.revision = 0,
  }) : dateOfBirth =
           dateOfBirth ??
           DateFormat(
             'yyyy-MM-dd',
           ).format(DateTime.now().subtract(const Duration(days: 365 * 3)));

  final int step;
  final int? selectedChildId;
  final String childName;
  final String dateOfBirth;
  final String ageMonths;
  final bool useDob;
  final String sex;
  final String guardianName;
  final String location;
  final String weightKg;
  final String heightValue;
  final String heightUnit;
  final String muacCm;
  final String? frontImagePath;
  final String? sideImagePath;
  final String? backImagePath;
  final bool showValidationErrors;
  final bool isSubmitting;
  final String? error;
  final int revision;

  bool get childDetailsValid {
    final dob = DateTime.tryParse(resolvedDateOfBirth);
    return childName.trim().isNotEmpty &&
        dob != null &&
        !dob.isAfter(DateTime.now());
  }

  bool get photoValid => frontImagePath != null;

  bool get measurementsValid =>
      _optionalPositive(weightKg) &&
      _optionalPositive(heightValue) &&
      _optionalPositive(muacCm);

  String get resolvedDateOfBirth {
    if (useDob) return dateOfBirth.trim();
    final months = double.tryParse(ageMonths.trim());
    if (months == null || months < 0) return dateOfBirth.trim();
    return DateFormat('yyyy-MM-dd').format(
      DateTime.now().subtract(Duration(days: (months * 30.4375).round())),
    );
  }

  bool _optionalPositive(String value) {
    if (value.trim().isEmpty) return true;
    final parsed = double.tryParse(value.trim());
    return parsed != null && parsed > 0;
  }

  AssessmentFormState copyWith({
    int? step,
    int? selectedChildId,
    bool clearSelectedChild = false,
    String? childName,
    String? dateOfBirth,
    String? ageMonths,
    bool? useDob,
    String? sex,
    String? guardianName,
    String? location,
    String? weightKg,
    String? heightValue,
    String? heightUnit,
    String? muacCm,
    String? frontImagePath,
    String? sideImagePath,
    String? backImagePath,
    bool clearFrontImage = false,
    bool clearSideImage = false,
    bool clearBackImage = false,
    bool? showValidationErrors,
    bool? isSubmitting,
    String? error,
    bool clearError = false,
    int? revision,
  }) {
    return AssessmentFormState(
      step: step ?? this.step,
      selectedChildId: clearSelectedChild
          ? null
          : (selectedChildId ?? this.selectedChildId),
      childName: childName ?? this.childName,
      dateOfBirth: dateOfBirth ?? this.dateOfBirth,
      ageMonths: ageMonths ?? this.ageMonths,
      useDob: useDob ?? this.useDob,
      sex: sex ?? this.sex,
      guardianName: guardianName ?? this.guardianName,
      location: location ?? this.location,
      weightKg: weightKg ?? this.weightKg,
      heightValue: heightValue ?? this.heightValue,
      heightUnit: heightUnit ?? this.heightUnit,
      muacCm: muacCm ?? this.muacCm,
      frontImagePath: clearFrontImage
          ? null
          : (frontImagePath ?? this.frontImagePath),
      sideImagePath: clearSideImage
          ? null
          : (sideImagePath ?? this.sideImagePath),
      backImagePath: clearBackImage
          ? null
          : (backImagePath ?? this.backImagePath),
      showValidationErrors: showValidationErrors ?? this.showValidationErrors,
      isSubmitting: isSubmitting ?? this.isSubmitting,
      error: clearError ? null : (error ?? this.error),
      revision: revision ?? this.revision,
    );
  }
}

class AssessmentFormNotifier extends StateNotifier<AssessmentFormState> {
  AssessmentFormNotifier(this._ref) : super(AssessmentFormState());

  final Ref _ref;

  void updateChildName(String value) => state = state.copyWith(
    childName: value,
    clearSelectedChild: state.selectedChildId != null,
    clearError: true,
  );

  void updateDateOfBirth(String value) =>
      state = state.copyWith(dateOfBirth: value, clearError: true);

  void updateAgeMonths(String value) =>
      state = state.copyWith(ageMonths: value, clearError: true);

  void updateUseDob(bool value) =>
      state = state.copyWith(useDob: value, showValidationErrors: false);

  void updateSex(String value) =>
      state = state.copyWith(sex: value, clearError: true);

  void updateGuardianName(String value) =>
      state = state.copyWith(guardianName: value);

  void updateLocation(String value) => state = state.copyWith(location: value);

  void updateWeight(String value) => state = state.copyWith(weightKg: value);

  void updateHeight(String value) => state = state.copyWith(heightValue: value);

  void updateHeightUnit(String value) =>
      state = state.copyWith(heightUnit: value);

  void updateMuac(String value) => state = state.copyWith(muacCm: value);

  void selectChild(ChildrenData child) {
    state = state.copyWith(
      selectedChildId: child.id,
      childName: child.name,
      dateOfBirth: child.dateOfBirth,
      useDob: true,
      sex: child.sex,
      guardianName: child.guardianName ?? '',
      location: child.location ?? '',
      showValidationErrors: false,
      clearError: true,
      revision: state.revision + 1,
    );
  }

  void useNewChild() {
    state = AssessmentFormState(revision: state.revision + 1);
  }

  void setImage(String role, String path) {
    state = switch (role) {
      'front' => state.copyWith(
        frontImagePath: path,
        showValidationErrors: false,
        clearError: true,
      ),
      'side' => state.copyWith(sideImagePath: path),
      'back' => state.copyWith(backImagePath: path),
      _ => state,
    };
  }

  void removeImage(String role) {
    state = switch (role) {
      'front' => state.copyWith(clearFrontImage: true),
      'side' => state.copyWith(clearSideImage: true),
      'back' => state.copyWith(clearBackImage: true),
      _ => state,
    };
  }

  bool next() {
    final valid = state.step == 0
        ? state.childDetailsValid
        : state.photoValid && state.measurementsValid;
    if (!valid) {
      state = state.copyWith(showValidationErrors: true);
      return false;
    }
    if (state.step < 2) {
      state = state.copyWith(
        step: state.step + 1,
        showValidationErrors: false,
        clearError: true,
      );
    }
    return true;
  }

  void back() {
    if (state.step > 0) {
      state = state.copyWith(
        step: state.step - 1,
        showValidationErrors: false,
        clearError: true,
      );
    }
  }

  Future<bool> submit() async {
    if (!state.childDetailsValid ||
        !state.photoValid ||
        !state.measurementsValid) {
      state = state.copyWith(showValidationErrors: true);
      return false;
    }

    final heightValue = double.tryParse(state.heightValue.trim());
    final heightCm = heightValue == null
        ? null
        : state.heightUnit == 'inch'
        ? heightValue * 2.54
        : heightValue;

    state = state.copyWith(isSubmitting: true, clearError: true);
    try {
      final service = await _ref.read(assessmentServiceProvider.future);
      final result = await service.runAssessment(
        frontImagePath: state.frontImagePath!,
        sideImagePath: state.sideImagePath,
        backImagePath: state.backImagePath,
        childName: state.childName.trim(),
        dateOfBirth: state.resolvedDateOfBirth,
        sex: state.sex,
        manualWeightKg: double.tryParse(state.weightKg.trim()),
        manualHeightCm: heightCm,
        manualMuacCm: double.tryParse(state.muacCm.trim()),
        guardianName: _optional(state.guardianName),
        location: _optional(state.location),
        ownerUserId: _ref.read(authProvider).user?.id,
      );
      _ref.read(assessmentResultProvider.notifier).state = result;
      _ref.invalidate(childrenProvider);
      state = state.copyWith(isSubmitting: false);
      return true;
    } catch (error) {
      state = state.copyWith(isSubmitting: false, error: error.toString());
      return false;
    }
  }

  String? _optional(String value) {
    final trimmed = value.trim();
    return trimmed.isEmpty ? null : trimmed;
  }
}

final assessmentFormProvider =
    StateNotifierProvider.autoDispose<
      AssessmentFormNotifier,
      AssessmentFormState
    >((ref) {
      return AssessmentFormNotifier(ref);
    });
