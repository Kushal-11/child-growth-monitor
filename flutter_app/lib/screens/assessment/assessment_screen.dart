import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/api_provider.dart';
import '../../providers/assessment_provider.dart';
import '../../providers/children_provider.dart';
import '../shared/app_scaffold.dart';

class AssessmentScreen extends ConsumerStatefulWidget {
  const AssessmentScreen({super.key});

  @override
  ConsumerState<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends ConsumerState<AssessmentScreen> {
  final _formKey = GlobalKey<FormState>();
  final _baseUrlController = TextEditingController();
  final _childNameController = TextEditingController();
  final _dobController = TextEditingController();
  final _ageMonthsController = TextEditingController();
  final _weightController = TextEditingController();
  final _heightValueController = TextEditingController();
  final _muacController = TextEditingController();
  final _guardianController = TextEditingController();
  final _locationController = TextEditingController();

  final ImagePicker _picker = ImagePicker();

  String _sex = 'M';
  String _heightUnit = 'cm';
  bool _useDob = true;
  bool _loading = false;
  bool? _healthy;
  String? _error;

  XFile? _frontImage;
  XFile? _sideImage;
  XFile? _backImage;

  @override
  void initState() {
    super.initState();
    _dobController.text = DateFormat('yyyy-MM-dd').format(
      DateTime.now().subtract(const Duration(days: 365 * 3)),
    );
    _initBaseUrl();
  }

  Future<void> _initBaseUrl() async {
    final saved = await loadSavedBaseUrl();
    if (!mounted) return;
    _baseUrlController.text = saved;
    ref.read(baseUrlProvider.notifier).state = saved;
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    _childNameController.dispose();
    _dobController.dispose();
    _ageMonthsController.dispose();
    _weightController.dispose();
    _heightValueController.dispose();
    _muacController.dispose();
    _guardianController.dispose();
    _locationController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source, String role) async {
    final file = await _picker.pickImage(source: source, imageQuality: 90);
    if (!mounted || file == null) return;
    setState(() {
      switch (role) {
        case 'front':
          _frontImage = file;
        case 'side':
          _sideImage = file;
        case 'back':
          _backImage = file;
      }
    });
  }

  Future<void> _selectDob() async {
    final initial = DateTime.tryParse(_dobController.text) ?? DateTime.now();
    final selected = await showDatePicker(
      context: context,
      initialDate: initial,
      firstDate: DateTime(2000),
      lastDate: DateTime.now(),
    );
    if (!mounted || selected == null) return;
    setState(() {
      _dobController.text = DateFormat('yyyy-MM-dd').format(selected);
      // Sync age months
      final days = DateTime.now().difference(selected).inDays;
      _ageMonthsController.text = (days / 30.4375).toStringAsFixed(0);
    });
  }

  void _onAgeMonthsChanged(String value) {
    final months = double.tryParse(value);
    if (months == null || months < 0) return;
    final dob = DateTime.now().subtract(Duration(days: (months * 30.4375).round()));
    _dobController.text = DateFormat('yyyy-MM-dd').format(dob);
  }

  String _resolvedDob() {
    if (_useDob) return _dobController.text.trim();
    final months = double.tryParse(_ageMonthsController.text.trim());
    if (months != null && months >= 0) {
      final dob =
          DateTime.now().subtract(Duration(days: (months * 30.4375).round()));
      return DateFormat('yyyy-MM-dd').format(dob);
    }
    return _dobController.text.trim();
  }

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

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (_frontImage == null) {
      setState(() => _error = t('front_image_required', ref));
      return;
    }

    final heightValue = double.tryParse(_heightValueController.text.trim());
    double? heightCm;
    if (heightValue != null) {
      heightCm = _heightUnit == 'inch' ? heightValue * 2.54 : heightValue;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final url = _baseUrlController.text.trim();
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;

      final result = await ref.read(apiProvider).submitAssessment(
            frontImagePath: _frontImage!.path,
            sideImagePath: _sideImage?.path,
            backImagePath: _backImage?.path,
            childName: _childNameController.text.trim(),
            dateOfBirth: _resolvedDob(),
            sex: _sex,
            weightKg: double.tryParse(_weightController.text.trim()),
            heightCm: heightCm,
            muacCm: double.tryParse(_muacController.text.trim()),
            guardianName: _guardianController.text.trim(),
            location: _locationController.text.trim(),
          );
      if (!mounted) return;
      ref.read(assessmentResultProvider.notifier).state = result;
      // Invalidate children list so it refreshes when viewed
      ref.invalidate(childrenProvider);
      context.go('/result');
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AppScaffold(
      currentIndex: 0,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Text(
                t('assess_heading', ref),
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 4),
              Text(
                t('assess_subtitle', ref),
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),

              // Debug: API URL
              if (kDebugMode) ...[
                TextFormField(
                  controller: _baseUrlController,
                  decoration: InputDecoration(
                    labelText: t('api_base_url', ref),
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.check_circle_outline),
                      onPressed: _loading ? null : _checkHealth,
                      tooltip: t('check_health', ref),
                    ),
                  ),
                ),
                if (_healthy != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 4, bottom: 8),
                    child: Row(
                      children: [
                        Icon(
                          _healthy! ? Icons.check_circle : Icons.error,
                          color: _healthy! ? Colors.green : Colors.red,
                          size: 16,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _healthy!
                              ? t('backend_healthy', ref)
                              : t('backend_unhealthy', ref),
                          style: TextStyle(
                            color: _healthy! ? Colors.green : Colors.red,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                const SizedBox(height: 8),
              ],

              // === IMAGES ===
              _sectionHeader(t('front_view_photo', ref), required: true),
              _photoGuidanceTips(),
              const SizedBox(height: 8),
              _imagePickerRow('front', _frontImage),
              const SizedBox(height: 16),

              _sectionHeader(
                '${t('side_view', ref)} (${t('optional_label', ref)})',
              ),
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.teal.shade50,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    t('side_accuracy_badge', ref),
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.teal.shade800,
                    ),
                  ),
                ),
              ),
              Text(
                t('side_view_help', ref),
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              _imagePickerRow('side', _sideImage),
              const SizedBox(height: 16),

              _sectionHeader(
                '${t('back_view', ref)} (${t('optional_label', ref)})',
              ),
              Text(
                t('back_view_help', ref),
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              _imagePickerRow('back', _backImage),

              const Divider(height: 32),

              // === CHILD INFO ===
              _sectionHeader(t('child_information', ref)),
              const SizedBox(height: 8),
              TextFormField(
                controller: _childNameController,
                decoration: InputDecoration(
                  labelText: '${t('child_name', ref)} *',
                  border: const OutlineInputBorder(),
                ),
                maxLength: 100,
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? t('required_field', ref) : null,
              ),
              const SizedBox(height: 12),
              // Sex selection
              Text(t('sex', ref),
                  style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 4),
              SegmentedButton<String>(
                segments: [
                  ButtonSegment(value: 'M', label: Text(t('male', ref))),
                  ButtonSegment(value: 'F', label: Text(t('female', ref))),
                ],
                selected: {_sex},
                onSelectionChanged: (v) => setState(() => _sex = v.first),
              ),
              const SizedBox(height: 12),

              // Age toggle
              if (_useDob) ...[
                TextFormField(
                  controller: _dobController,
                  decoration: InputDecoration(
                    labelText: t('date_of_birth', ref),
                    border: const OutlineInputBorder(),
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.calendar_today),
                      onPressed: _selectDob,
                    ),
                  ),
                  readOnly: true,
                  onTap: _selectDob,
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) {
                      return t('required_field', ref);
                    }
                    final parsed = DateTime.tryParse(v);
                    if (parsed == null) return t('use_date_format', ref);
                    if (parsed.isAfter(DateTime.now())) {
                      return t('dob_future_error', ref);
                    }
                    return null;
                  },
                ),
                TextButton(
                  onPressed: () => setState(() => _useDob = false),
                  child: Text(t('toggle_age_months', ref)),
                ),
              ] else ...[
                TextFormField(
                  controller: _ageMonthsController,
                  decoration: InputDecoration(
                    labelText: t('age_months', ref),
                    hintText: t('placeholder_age_months', ref),
                    border: const OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                  onChanged: _onAgeMonthsChanged,
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) {
                      return t('required_field', ref);
                    }
                    final n = double.tryParse(v);
                    if (n == null || n < 0) {
                      return t('positive_number_error', ref);
                    }
                    return null;
                  },
                ),
                TextButton(
                  onPressed: () => setState(() => _useDob = true),
                  child: Text(t('toggle_dob', ref)),
                ),
              ],

              const Divider(height: 24),

              // === OPTIONAL MEASUREMENTS ===
              _sectionHeader(
                '${t('optional_measurements', ref)} ${t('optional_measurements_note', ref)}',
              ),
              const SizedBox(height: 8),
              TextFormField(
                controller: _weightController,
                decoration: InputDecoration(
                  labelText: t('weight_kg', ref),
                  hintText: t('weight_placeholder', ref),
                  helperText: t('weight_help', ref),
                  border: const OutlineInputBorder(),
                ),
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                validator: (v) => _validatePositive(v),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _muacController,
                decoration: InputDecoration(
                  labelText: t('muac_cm', ref),
                  hintText: t('muac_placeholder', ref),
                  helperText: t('muac_help', ref),
                  border: const OutlineInputBorder(),
                ),
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                validator: (v) => _validatePositive(v),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: _heightValueController,
                      decoration: InputDecoration(
                        labelText: t('height', ref),
                        hintText: t('height_placeholder', ref),
                        helperText: t('height_fallback', ref),
                        border: const OutlineInputBorder(),
                      ),
                      keyboardType:
                          const TextInputType.numberWithOptions(decimal: true),
                      validator: (v) => _validatePositive(v),
                    ),
                  ),
                  const SizedBox(width: 8),
                  SegmentedButton<String>(
                    segments: [
                      ButtonSegment(
                          value: 'cm', label: Text(t('unit_cm', ref))),
                      ButtonSegment(
                          value: 'inch', label: Text(t('unit_inch', ref))),
                    ],
                    selected: {_heightUnit},
                    onSelectionChanged: (v) =>
                        setState(() => _heightUnit = v.first),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _guardianController,
                decoration: InputDecoration(
                  labelText: t('guardian_name', ref),
                  hintText: t('placeholder_optional', ref),
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _locationController,
                decoration: InputDecoration(
                  labelText: t('location_clinic', ref),
                  hintText: t('placeholder_optional', ref),
                  border: const OutlineInputBorder(),
                ),
              ),

              const SizedBox(height: 24),

              // === SUBMIT ===
              SizedBox(
                width: double.infinity,
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _submit,
                  child: _loading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : Text(t('run_assessment', ref)),
                ),
              ),

              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_error!, style: const TextStyle(color: Colors.red)),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionHeader(String text, {bool required = false}) {
    return Text(
      text,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
    );
  }

  Widget _photoGuidanceTips() {
    final tips = [
      t('tip_front_1', ref),
      t('tip_front_2', ref),
      t('tip_front_3', ref),
      t('tip_front_4', ref),
    ];
    return Container(
      margin: const EdgeInsets.only(top: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: tips
            .map((tip) => Padding(
                  padding: const EdgeInsets.only(bottom: 2),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('• ', style: TextStyle(fontSize: 12)),
                      Expanded(
                        child: Text(tip, style: const TextStyle(fontSize: 12)),
                      ),
                    ],
                  ),
                ))
            .toList(),
      ),
    );
  }

  Widget _imagePickerRow(String role, XFile? image) {
    return Row(
      children: [
        OutlinedButton.icon(
          onPressed: _loading
              ? null
              : () => _pickImage(ImageSource.camera, role),
          icon: const Icon(Icons.camera_alt, size: 18),
          label: Text(t('capture', ref)),
        ),
        const SizedBox(width: 8),
        OutlinedButton.icon(
          onPressed: _loading
              ? null
              : () => _pickImage(ImageSource.gallery, role),
          icon: const Icon(Icons.photo_library, size: 18),
          label: Text(t('gallery', ref)),
        ),
        const SizedBox(width: 8),
        if (image != null)
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: Image.file(
              File(image.path),
              width: 48,
              height: 48,
              fit: BoxFit.cover,
            ),
          )
        else
          Text(
            t('not_selected', ref),
            style: Theme.of(context).textTheme.bodySmall,
          ),
      ],
    );
  }

  String? _validatePositive(String? value) {
    if (value == null || value.trim().isEmpty) return null;
    final n = double.tryParse(value);
    if (n == null || n <= 0) return t('positive_number_error', ref);
    return null;
  }
}
