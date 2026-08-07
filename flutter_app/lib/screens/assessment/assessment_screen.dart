import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';

import '../../constants/config.dart';
import '../../constants/feature_flags.dart';
import '../../database/database.dart';
import '../../features/assessment/providers/assessment_form_provider.dart';
import '../../features/assessment/widgets/assessment_header.dart';
import '../../features/assessment/widgets/assessment_progress.dart';
import '../../features/assessment/widgets/child_selector_card.dart';
import '../../features/assessment/widgets/photo_guidance_tile.dart';
import '../../l10n/l10n_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';
import '../shared/app_scaffold.dart';
import 'capture_screen.dart';

class AssessmentScreen extends ConsumerWidget {
  const AssessmentScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(assessmentFormProvider);
    final notifier = ref.read(assessmentFormProvider.notifier);
    final stepLabels = [
      t('step_child_details', ref),
      t('step_photos_measurements', ref),
      t('step_review', ref),
    ];

    return AppScaffold(
      currentIndex: 0,
      showAppBar: false,
      child: SafeArea(
        bottom: false,
        child: Column(
          children: [
            const AssessmentHeader(),
            AssessmentProgress(step: state.step, label: stepLabels[state.step]),
            Expanded(
              child: SingleChildScrollView(
                key: PageStorageKey('assessment_page_${state.step}'),
                padding: const EdgeInsets.fromLTRB(
                  AppSpacing.xl,
                  AppSpacing.lg,
                  AppSpacing.xl,
                  AppSpacing.section,
                ),
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 180),
                  child: switch (state.step) {
                    0 => _ChildDetailsStep(
                        key: const ValueKey('child-details'),
                        state: state,
                      ),
                    1 => _PhotosStep(
                        key: const ValueKey('photos'),
                        state: state,
                        onPickImage: (source, role) =>
                            _pickImage(context, ref, source, role),
                      ),
                    _ => _ReviewStep(
                        key: const ValueKey('review'),
                        state: state,
                      ),
                  },
                ),
              ),
            ),
            _ActionBar(
              state: state,
              onBack: notifier.back,
              onContinue: () async {
                if (state.step < 2) {
                  notifier.next();
                  return;
                }
                final completed = await notifier.submit();
                if (completed && context.mounted) context.push('/result');
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _pickImage(
    BuildContext context,
    WidgetRef ref,
    ImageSource source,
    String role,
  ) async {
    XFile? file;
    if (source == ImageSource.camera && FeatureFlags.liveCaptureEnabled) {
      final result = await context.push<CaptureResult>('/capture/$role');
      if (result == null) return;
      file = result.useSystemCamera
          ? await ImagePicker().pickImage(source: source, imageQuality: 90)
          : XFile(result.imagePath!);
    } else {
      file = await ImagePicker().pickImage(source: source, imageQuality: 90);
    }
    if (file == null || !context.mounted) return;
    ref.read(assessmentFormProvider.notifier).setImage(role, file.path);
  }
}

class _ChildDetailsStep extends ConsumerWidget {
  const _ChildDetailsStep({super.key, required this.state});

  final AssessmentFormState state;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final notifier = ref.read(assessmentFormProvider.notifier);
    final children = ref.watch(assessmentChildrenProvider).value ?? const [];
    final selected = children
        .where((child) => child.id == state.selectedChildId)
        .firstOrNull;

    final dob = DateTime.tryParse(state.resolvedDateOfBirth);
    final resolvedAge = state.resolvedAgeMonths;
    final invalidDob = dob == null ||
        dob.isAfter(DateTime.now()) ||
        resolvedAge == null ||
        resolvedAge < 0 ||
        resolvedAge >= maxUnderFiveAgeMonths;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('who_are_we_assessing', ref),
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          t('child_details_help', ref),
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: AppSpacing.xl),
        ChildSelectorCard(
          title: selected?.name ?? t('select_registered_child', ref),
          subtitle: selected == null
              ? t('or_enter_new_profile', ref)
              : '${selected.sex == 'M' ? t('male', ref) : t('female', ref)} · '
                  '${selected.dateOfBirth}',
          onTap: () => _showChildSelector(context, ref, children),
        ),
        const SizedBox(height: AppSpacing.xxl),
        Text(
          t('child_information', ref),
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: AppSpacing.lg),
        TextFormField(
          key: ValueKey('assessment_child_name_${state.revision}'),
          initialValue: state.childName,
          onChanged: notifier.updateChildName,
          maxLength: 100,
          textInputAction: TextInputAction.next,
          decoration: InputDecoration(
            labelText: '${t('child_name', ref)} *',
            errorText:
                state.showValidationErrors && state.childName.trim().isEmpty
                    ? t('required_field', ref)
                    : null,
          ),
        ),
        const SizedBox(height: AppSpacing.md),
        Text(t('sex', ref), style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: AppSpacing.sm),
        SizedBox(
          width: double.infinity,
          child: SegmentedButton<String>(
            segments: [
              ButtonSegment(value: 'M', label: Text(t('male', ref))),
              ButtonSegment(value: 'F', label: Text(t('female', ref))),
            ],
            selected: state.sex.isEmpty ? const {} : {state.sex},
            emptySelectionAllowed: true,
            onSelectionChanged: (selection) =>
                notifier.updateSex(selection.firstOrNull ?? ''),
          ),
        ),
        if (state.showValidationErrors && state.sex.isEmpty)
          Padding(
            padding: const EdgeInsets.only(top: AppSpacing.xs),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                t('required_field', ref),
                style: TextStyle(
                  color: Theme.of(context).colorScheme.error,
                  fontSize: 12,
                ),
              ),
            ),
          ),
        const SizedBox(height: AppSpacing.lg),
        SegmentedButton<bool>(
          segments: [
            ButtonSegment(value: true, label: Text(t('date_of_birth', ref))),
            ButtonSegment(value: false, label: Text(t('age_months', ref))),
          ],
          selected: {state.useDob},
          onSelectionChanged: (selection) =>
              notifier.updateUseDob(selection.first),
        ),
        const SizedBox(height: AppSpacing.md),
        if (state.useDob)
          InkWell(
            key: const Key('assessment_dob'),
            borderRadius: BorderRadius.circular(12),
            onTap: () => _selectDob(context, ref),
            child: InputDecorator(
              decoration: InputDecoration(
                labelText: '${t('date_of_birth', ref)} *',
                errorText: state.showValidationErrors && invalidDob
                    ? resolvedAge != null &&
                            resolvedAge >= maxUnderFiveAgeMonths
                        ? t('under_five_error', ref)
                        : t('err_invalid_date', ref)
                    : null,
                suffixIcon: const Icon(Icons.calendar_today_outlined),
              ),
              child: Text(
                state.dateOfBirth.isEmpty
                    ? t('select_date_of_birth', ref)
                    : state.dateOfBirth,
              ),
            ),
          )
        else
          TextFormField(
            key: ValueKey('assessment_age_${state.revision}'),
            initialValue: state.ageMonths,
            onChanged: notifier.updateAgeMonths,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: InputDecoration(
              labelText: '${t('age_months', ref)} *',
              hintText: t('placeholder_age_months', ref),
              errorText: state.showValidationErrors &&
                      (double.tryParse(state.ageMonths) == null ||
                          !(double.tryParse(state.ageMonths)?.isFinite ??
                              false) ||
                          (double.tryParse(state.ageMonths) ?? -1) < 0 ||
                          (double.tryParse(state.ageMonths) ?? 60) >=
                              maxUnderFiveAgeMonths)
                  ? t('age_required_feedback', ref)
                  : null,
            ),
          ),
        const SizedBox(height: AppSpacing.lg),
        TextFormField(
          key: ValueKey('assessment_guardian_${state.revision}'),
          initialValue: state.guardianName,
          onChanged: notifier.updateGuardianName,
          textInputAction: TextInputAction.next,
          decoration: InputDecoration(
            labelText: t('guardian_name', ref),
            hintText: t('placeholder_optional', ref),
          ),
        ),
        const SizedBox(height: AppSpacing.md),
        TextFormField(
          key: ValueKey('assessment_location_${state.revision}'),
          initialValue: state.location,
          onChanged: notifier.updateLocation,
          decoration: InputDecoration(
            labelText: t('location_clinic', ref),
            hintText: t('placeholder_optional', ref),
          ),
        ),
      ],
    );
  }

  Future<void> _selectDob(BuildContext context, WidgetRef ref) async {
    final state = ref.read(assessmentFormProvider);
    final now = DateTime.now();
    final selected = await showDatePicker(
      context: context,
      initialDate: DateTime.tryParse(state.dateOfBirth) ??
          DateTime(now.year - 3, now.month, now.day),
      firstDate: DateTime(now.year - 5, now.month, now.day)
          .add(const Duration(days: 1)),
      lastDate: now,
    );
    if (selected != null && context.mounted) {
      ref
          .read(assessmentFormProvider.notifier)
          .updateDateOfBirth(DateFormat('yyyy-MM-dd').format(selected));
    }
  }

  Future<void> _showChildSelector(
    BuildContext context,
    WidgetRef ref,
    List<ChildrenData> children,
  ) async {
    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      builder: (sheetContext) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.xl,
              0,
              AppSpacing.xl,
              AppSpacing.xl,
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  t('select_registered_child', ref),
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: AppSpacing.md),
                ListTile(
                  key: const Key('select_new_child'),
                  contentPadding: EdgeInsets.zero,
                  leading: const CircleAvatar(
                    backgroundColor: AppColors.primaryContainer,
                    child: Icon(Icons.person_add_alt_1_rounded),
                  ),
                  title: Text(t('new_child', ref)),
                  subtitle: Text(t('enter_new_child_details', ref)),
                  onTap: () {
                    ref.read(assessmentFormProvider.notifier).useNewChild();
                    Navigator.pop(sheetContext);
                  },
                ),
                if (children.isEmpty)
                  Padding(
                    padding: const EdgeInsets.symmetric(
                      vertical: AppSpacing.xl,
                    ),
                    child: Text(t('empty_children', ref)),
                  )
                else
                  Flexible(
                    child: ListView.builder(
                      shrinkWrap: true,
                      itemCount: children.length,
                      itemBuilder: (context, index) {
                        final child = children[index];
                        return ListTile(
                          key: Key('select_child_${child.id}'),
                          contentPadding: EdgeInsets.zero,
                          leading: CircleAvatar(
                            child: Text(
                              child.name.isEmpty
                                  ? '?'
                                  : child.name.characters.first.toUpperCase(),
                            ),
                          ),
                          title: Text(child.name),
                          subtitle: Text(child.dateOfBirth),
                          onTap: () {
                            ref
                                .read(assessmentFormProvider.notifier)
                                .selectChild(child);
                            Navigator.pop(sheetContext);
                          },
                        );
                      },
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _PhotosStep extends ConsumerWidget {
  const _PhotosStep({
    super.key,
    required this.state,
    required this.onPickImage,
  });

  final AssessmentFormState state;
  final void Function(ImageSource source, String role) onPickImage;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final notifier = ref.read(assessmentFormProvider.notifier);
    final frontError = state.showValidationErrors && !state.photoValid
        ? t('front_image_required', ref)
        : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('capture_measurements_heading', ref),
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          t('capture_measurements_help', ref),
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: AppSpacing.xl),
        PhotoGuidanceTile(
          role: 'front',
          title: '${t('front_view_photo', ref)} *',
          help: t('front_photo_compact_help', ref),
          cameraLabel: t('capture', ref),
          galleryLabel: t('gallery', ref),
          imagePath: state.frontImagePath,
          showGuide: true,
          errorText: frontError,
          onCamera: () => onPickImage(ImageSource.camera, 'front'),
          onGallery: () => onPickImage(ImageSource.gallery, 'front'),
          onRemove: () => notifier.removeImage('front'),
        ),
        const SizedBox(height: AppSpacing.xxl),
        PhotoGuidanceTile(
          role: 'side',
          title: t('side_view', ref),
          help: t('side_view_help', ref),
          cameraLabel: t('capture', ref),
          galleryLabel: t('gallery', ref),
          imagePath: state.sideImagePath,
          optionalLabel: t('optional_label', ref),
          onCamera: () => onPickImage(ImageSource.camera, 'side'),
          onGallery: () => onPickImage(ImageSource.gallery, 'side'),
          onRemove: () => notifier.removeImage('side'),
        ),
        const SizedBox(height: AppSpacing.xxl),
        PhotoGuidanceTile(
          role: 'back',
          title: t('back_view', ref),
          help: t('back_view_help', ref),
          cameraLabel: t('capture', ref),
          galleryLabel: t('gallery', ref),
          imagePath: state.backImagePath,
          optionalLabel: t('optional_label', ref),
          onCamera: () => onPickImage(ImageSource.camera, 'back'),
          onGallery: () => onPickImage(ImageSource.gallery, 'back'),
          onRemove: () => notifier.removeImage('back'),
        ),
        const SizedBox(height: AppSpacing.section),
        _MeasurementsCard(state: state),
      ],
    );
  }
}

class _MeasurementsCard extends ConsumerWidget {
  const _MeasurementsCard({required this.state});

  final AssessmentFormState state;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final notifier = ref.read(assessmentFormProvider.notifier);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('optional_measurements', ref),
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: AppSpacing.xs),
            Text(
              t('manual_measurements_priority_note', ref),
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: AppSpacing.lg),
            _NumericField(
              fieldKey: 'assessment_weight',
              initialValue: state.weightKg,
              label: t('weight_kg', ref),
              hint: t('weight_placeholder', ref),
              errorText: _rangeError(
                state.weightKg,
                ref,
                min: minPlausibleWeightKg,
                max: maxPlausibleWeightKg,
                unit: 'kg',
              ),
              onChanged: notifier.updateWeight,
            ),
            const SizedBox(height: AppSpacing.md),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: _NumericField(
                    fieldKey: 'assessment_height',
                    initialValue: state.heightValue,
                    label: t('height', ref),
                    hint: t('height_placeholder', ref),
                    errorText: _rangeError(
                      state.heightValue,
                      ref,
                      min: minPlausibleHeightCm,
                      max: maxPlausibleHeightCm,
                      unit: 'cm',
                      multiplier: state.heightUnit == 'inch' ? 2.54 : 1,
                    ),
                    onChanged: notifier.updateHeight,
                  ),
                ),
                const SizedBox(width: AppSpacing.sm),
                DropdownButton<String>(
                  value: state.heightUnit,
                  items: [
                    DropdownMenuItem(
                      value: 'cm',
                      child: Text(t('unit_cm', ref)),
                    ),
                    DropdownMenuItem(
                      value: 'inch',
                      child: Text(t('unit_inch', ref)),
                    ),
                  ],
                  onChanged: (value) {
                    if (value != null) notifier.updateHeightUnit(value);
                  },
                ),
              ],
            ),
            const SizedBox(height: AppSpacing.md),
            _NumericField(
              fieldKey: 'assessment_muac',
              initialValue: state.muacCm,
              label: t('muac_cm', ref),
              hint: t('muac_placeholder', ref),
              errorText: _rangeError(
                state.muacCm,
                ref,
                min: minPlausibleMuacCm,
                max: maxPlausibleMuacCm,
                unit: 'cm',
              ),
              onChanged: notifier.updateMuac,
            ),
          ],
        ),
      ),
    );
  }

  String? _rangeError(
    String value,
    WidgetRef ref, {
    required double min,
    required double max,
    required String unit,
    double multiplier = 1,
  }) {
    if (!state.showValidationErrors || value.trim().isEmpty) return null;
    final parsed = double.tryParse(value.trim());
    final canonical = parsed == null ? null : parsed * multiplier;
    return canonical == null ||
            !canonical.isFinite ||
            canonical < min ||
            canonical > max
        ? '${t('measurement_range_prefix', ref)} $min–$max $unit'
        : null;
  }
}

class _NumericField extends StatelessWidget {
  const _NumericField({
    required this.fieldKey,
    required this.initialValue,
    required this.label,
    required this.hint,
    required this.errorText,
    required this.onChanged,
  });

  final String fieldKey;
  final String initialValue;
  final String label;
  final String hint;
  final String? errorText;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      key: Key(fieldKey),
      initialValue: initialValue,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      onChanged: onChanged,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        errorText: errorText,
      ),
    );
  }
}

class _ReviewStep extends ConsumerWidget {
  const _ReviewStep({super.key, required this.state});

  final AssessmentFormState state;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('review_assessment_heading', ref),
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          t('review_assessment_help', ref),
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: AppSpacing.xl),
        _ReviewCard(
          title: t('child_information', ref),
          icon: Icons.child_care_rounded,
          rows: [
            (t('child_name', ref), state.childName),
            (t('date_of_birth', ref), state.resolvedDateOfBirth),
            (
              t('sex', ref),
              state.sex == 'M' ? t('male', ref) : t('female', ref),
            ),
            if (state.guardianName.trim().isNotEmpty)
              (t('guardian_name', ref), state.guardianName),
            if (state.location.trim().isNotEmpty)
              (t('location_clinic', ref), state.location),
          ],
        ),
        const SizedBox(height: AppSpacing.lg),
        _ReviewCard(
          title: t('photos', ref),
          icon: Icons.photo_camera_outlined,
          rows: [
            (
              t('front_view_photo', ref),
              state.frontImagePath == null
                  ? t('not_selected', ref)
                  : t('ready', ref),
            ),
            (
              t('side_view', ref),
              state.sideImagePath == null
                  ? t('not_selected', ref)
                  : t('ready', ref),
            ),
            (
              t('back_view', ref),
              state.backImagePath == null
                  ? t('not_selected', ref)
                  : t('back_view_archived_only', ref),
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.lg),
        _ReviewCard(
          title: t('optional_measurements', ref),
          icon: Icons.straighten_rounded,
          rows: [
            (
              t('weight_kg', ref),
              state.weightKg.trim().isEmpty
                  ? t('not_provided', ref)
                  : state.weightKg,
            ),
            (
              t('height', ref),
              state.heightValue.trim().isEmpty
                  ? t('not_provided', ref)
                  : '${state.heightValue} ${state.heightUnit}',
            ),
            (
              t('muac_cm', ref),
              state.muacCm.trim().isEmpty
                  ? t('not_provided', ref)
                  : state.muacCm,
            ),
          ],
        ),
        if (state.error != null) ...[
          const SizedBox(height: AppSpacing.lg),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(AppSpacing.md),
            decoration: BoxDecoration(
              color: const Color(0xFFFFEDEA),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              state.error!,
              style: const TextStyle(color: AppColors.error),
            ),
          ),
        ],
      ],
    );
  }
}

class _ReviewCard extends StatelessWidget {
  const _ReviewCard({
    required this.title,
    required this.icon,
    required this.rows,
  });

  final String title;
  final IconData icon;
  final List<(String, String)> rows;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          children: [
            Row(
              children: [
                Icon(icon, color: AppColors.primary),
                const SizedBox(width: AppSpacing.sm),
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
              ],
            ),
            const Divider(height: AppSpacing.xxl),
            for (final row in rows)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 5),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text(
                        row.$1,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                    const SizedBox(width: AppSpacing.md),
                    Flexible(
                      child: Text(
                        row.$2,
                        textAlign: TextAlign.end,
                        style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _ActionBar extends ConsumerWidget {
  const _ActionBar({
    required this.state,
    required this.onBack,
    required this.onContinue,
  });

  final AssessmentFormState state;
  final VoidCallback onBack;
  final Future<void> Function() onContinue;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      padding: const EdgeInsets.fromLTRB(
        AppSpacing.xl,
        AppSpacing.md,
        AppSpacing.xl,
        AppSpacing.lg,
      ),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: AppColors.progressTrack)),
      ),
      child: Row(
        children: [
          if (state.step > 0) ...[
            OutlinedButton.icon(
              key: const Key('assessment_back'),
              onPressed: state.isSubmitting ? null : onBack,
              icon: const Icon(Icons.arrow_back_rounded),
              label: Text(t('back', ref)),
            ),
            const SizedBox(width: AppSpacing.md),
          ],
          Expanded(
            child: FilledButton.icon(
              key: Key(
                state.step == 2 ? 'assessment_submit' : 'assessment_next',
              ),
              onPressed: state.isSubmitting ? null : onContinue,
              icon: state.isSubmitting
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : Icon(
                      state.step == 2
                          ? Icons.monitor_heart_outlined
                          : Icons.arrow_forward_rounded,
                    ),
              label: Text(
                state.isSubmitting
                    ? t('processing', ref)
                    : state.step == 2
                        ? t('run_assessment', ref)
                        : t('continue', ref),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
