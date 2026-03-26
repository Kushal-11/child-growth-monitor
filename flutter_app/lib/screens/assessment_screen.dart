import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/assessment_result.dart';
import '../models/child.dart';
import '../models/child_detail.dart';
import '../services/api_service.dart';

class AssessmentScreen extends StatefulWidget {
  const AssessmentScreen({super.key});

  @override
  State<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends State<AssessmentScreen> {
  static const _developmentBaseUrl = 'http://10.0.2.2:8000';
  static const defaultBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: _developmentBaseUrl,
  );
  static const _prefsBaseUrlKey = 'api_base_url';

  final _formKey = GlobalKey<FormState>();
  final _baseUrlController = TextEditingController(text: defaultBaseUrl);
  final _childNameController = TextEditingController();
  final _dobController = TextEditingController();
  final _weightController = TextEditingController();
  final _heightCmController = TextEditingController();
  final _heightValueController = TextEditingController();
  final _muacController = TextEditingController();
  final _guardianController = TextEditingController();
  final _locationController = TextEditingController();

  final ImagePicker _picker = ImagePicker();

  String _sex = 'M';
  String _heightUnit = 'cm';

  bool _loading = false;
  bool _loadingChildDetail = false;
  bool? _healthy;
  String? _error;

  XFile? _frontImage;
  XFile? _sideImage;
  XFile? _backImage;

  AssessmentResult? _result;
  List<ChildSummary> _children = const [];
  ChildDetail? _selectedChild;

  ApiService get _api => ApiService(baseUrl: _baseUrlController.text.trim());

  @override
  void initState() {
    super.initState();
    _dobController.text = DateFormat('yyyy-MM-dd').format(
      DateTime.now().subtract(const Duration(days: 365 * 3)),
    );
    _loadBaseUrl();
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    _childNameController.dispose();
    _dobController.dispose();
    _weightController.dispose();
    _heightCmController.dispose();
    _heightValueController.dispose();
    _muacController.dispose();
    _guardianController.dispose();
    _locationController.dispose();
    super.dispose();
  }

  Future<void> _loadBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    final persisted = prefs.getString(_prefsBaseUrlKey);

    final hasCompileTimeApiUrl = defaultBaseUrl != _developmentBaseUrl;
    if (hasCompileTimeApiUrl) {
      return;
    }

    if (persisted != null && persisted.isNotEmpty) {
      setState(() => _baseUrlController.text = persisted);
    }
  }

  Future<void> _saveBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsBaseUrlKey, _baseUrlController.text.trim());
  }

  Future<void> _pickImage(ImageSource source, String role) async {
    final file = await _picker.pickImage(source: source, imageQuality: 90);
    if (!mounted) return;
    if (file == null) {
      return;
    }

    setState(() {
      if (role == 'front') {
        _frontImage = file;
      } else if (role == 'side') {
        _sideImage = file;
      } else {
        _backImage = file;
      }
    });
  }

  Future<void> _selectDob() async {
    final initial = DateTime.tryParse(_dobController.text) ?? DateTime.now();
    final selected = await showDatePicker(
      context: context,
      initialDate: initial,
      firstDate: DateTime(2000, 1, 1),
      lastDate: DateTime.now(),
    );
    if (!mounted) return;

    if (selected != null) {
      setState(() {
        _dobController.text = DateFormat('yyyy-MM-dd').format(selected);
      });
    }
  }

  Future<void> _refreshChildren() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final children = await _api.getChildren();
      if (!mounted) return;
      setState(() => _children = children);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Future<void> _loadChildDetail(int childId) async {
    setState(() {
      _loadingChildDetail = true;
      _error = null;
    });

    try {
      final detail = await _api.getChildDetail(childId);
      if (!mounted) return;
      setState(() => _selectedChild = detail);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _loadingChildDetail = false);
    }
  }

  Future<void> _checkHealth() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      await _saveBaseUrl();
      if (!mounted) return;
      final healthy = await _api.checkHealth();
      if (!mounted) return;
      setState(() => _healthy = healthy);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _healthy = false;
        _error = e.toString();
      });
    } finally {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  String? _validateDob(String? value) {
    if (value == null || value.trim().isEmpty) {
      return 'Required';
    }

    final parsed = DateTime.tryParse(value);
    if (parsed == null) {
      return 'Use YYYY-MM-DD';
    }

    if (parsed.isAfter(DateTime.now())) {
      return 'DOB cannot be in the future';
    }

    return null;
  }

  String? _validatePositive(String? value, {bool required = false}) {
    if (value == null || value.trim().isEmpty) {
      return required ? 'Required' : null;
    }

    final number = double.tryParse(value);
    if (number == null || number <= 0) {
      return 'Must be a positive number';
    }

    return null;
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    if (_frontImage == null) {
      setState(() => _error = 'Please select a front image.');
      return;
    }

    final parsedHeightCm = double.tryParse(_heightCmController.text.trim());
    final parsedHeightValue = double.tryParse(_heightValueController.text.trim());
    if (parsedHeightCm != null && parsedHeightValue != null) {
      setState(() {
        _error = 'Please provide either height in cm OR height value/unit, not both.';
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      await _saveBaseUrl();
      if (!mounted) return;
      final result = await _api.submitAssessment(
        frontImagePath: _frontImage!.path,
        sideImagePath: _sideImage?.path,
        backImagePath: _backImage?.path,
        childName: _childNameController.text.trim(),
        dateOfBirth: _dobController.text.trim(),
        sex: _sex,
        weightKg: double.tryParse(_weightController.text.trim()),
        heightCm: parsedHeightCm,
        heightValue: parsedHeightValue,
        heightUnit: _heightUnit,
        muacCm: double.tryParse(_muacController.text.trim()),
        guardianName: _guardianController.text.trim(),
        location: _locationController.text.trim(),
      );
      if (!mounted) return;
      setState(() => _result = result);
      await _refreshChildren();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Child Growth Monitor')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              TextFormField(
                controller: _baseUrlController,
                decoration: const InputDecoration(
                  labelText: 'FastAPI Base URL',
                  helperText: 'Use http://10.0.2.2:8000 for Android emulator',
                ),
                validator: (value) =>
                    (value == null || value.trim().isEmpty) ? 'Required' : null,
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  ElevatedButton(
                    onPressed: _loading ? null : _checkHealth,
                    child: const Text('Check Health'),
                  ),
                  OutlinedButton(
                    onPressed: _loading ? null : _refreshChildren,
                    child: const Text('Load Children'),
                  ),
                  OutlinedButton(
                    onPressed: _loading
                        ? null
                        : () => _pickImage(ImageSource.camera, 'front'),
                    child: const Text('Capture Front'),
                  ),
                  OutlinedButton(
                    onPressed: _loading
                        ? null
                        : () => _pickImage(ImageSource.gallery, 'front'),
                    child: const Text('Pick Front'),
                  ),
                  OutlinedButton(
                    onPressed: _loading
                        ? null
                        : () => _pickImage(ImageSource.gallery, 'side'),
                    child: const Text('Pick Side'),
                  ),
                  OutlinedButton(
                    onPressed: _loading
                        ? null
                        : () => _pickImage(ImageSource.gallery, 'back'),
                    child: const Text('Pick Back'),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              _imageSummary('Front image', _frontImage, required: true),
              _imageSummary('Side image', _sideImage),
              _imageSummary('Back image', _backImage),
              if (_healthy != null)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    _healthy! ? 'Backend is healthy.' : 'Backend health check failed.',
                  ),
                ),
              const Divider(height: 24),
              TextFormField(
                controller: _childNameController,
                decoration: const InputDecoration(labelText: 'Child name'),
                validator: (value) =>
                    (value == null || value.trim().isEmpty) ? 'Required' : null,
              ),
              TextFormField(
                controller: _dobController,
                decoration: InputDecoration(
                  labelText: 'Date of birth (YYYY-MM-DD)',
                  suffixIcon: IconButton(
                    icon: const Icon(Icons.calendar_today),
                    onPressed: _selectDob,
                  ),
                ),
                validator: _validateDob,
                readOnly: true,
              ),
              DropdownButtonFormField<String>(
                value: _sex,
                decoration: const InputDecoration(labelText: 'Sex'),
                items: const [
                  DropdownMenuItem(value: 'M', child: Text('Male (M)')),
                  DropdownMenuItem(value: 'F', child: Text('Female (F)')),
                ],
                onChanged: (value) => setState(() => _sex = value ?? 'M'),
              ),
              TextFormField(
                controller: _weightController,
                decoration: const InputDecoration(labelText: 'Weight kg (optional)'),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                validator: _validatePositive,
              ),
              TextFormField(
                controller: _heightCmController,
                decoration: const InputDecoration(
                  labelText: 'Height cm (optional manual override)',
                ),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                validator: _validatePositive,
              ),
              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: _heightValueController,
                      decoration: const InputDecoration(
                        labelText: 'Height value (optional)',
                      ),
                      keyboardType:
                          const TextInputType.numberWithOptions(decimal: true),
                      validator: _validatePositive,
                    ),
                  ),
                  const SizedBox(width: 12),
                  DropdownButton<String>(
                    value: _heightUnit,
                    items: const [
                      DropdownMenuItem(value: 'cm', child: Text('cm')),
                      DropdownMenuItem(value: 'inch', child: Text('inch')),
                    ],
                    onChanged: (value) => setState(() => _heightUnit = value ?? 'cm'),
                  ),
                ],
              ),
              TextFormField(
                controller: _muacController,
                decoration: const InputDecoration(labelText: 'MUAC cm (optional)'),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                validator: _validatePositive,
              ),
              TextFormField(
                controller: _guardianController,
                decoration: const InputDecoration(labelText: 'Guardian name (optional)'),
              ),
              TextFormField(
                controller: _locationController,
                decoration: const InputDecoration(labelText: 'Location (optional)'),
              ),
              const SizedBox(height: 12),
              FilledButton(
                onPressed: _loading ? null : _submit,
                child: const Text('Submit Assessment'),
              ),
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_error!, style: const TextStyle(color: Colors.red)),
                ),
              if (_result != null) _buildResultCard(_result!),
              if (_children.isNotEmpty) _buildChildrenCard(),
              if (_selectedChild != null) _buildChildDetailCard(_selectedChild!),
              if (_loadingChildDetail)
                const Padding(
                  padding: EdgeInsets.only(top: 12),
                  child: Center(child: CircularProgressIndicator()),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _imageSummary(String label, XFile? file, {bool required = false}) {
    final value = file == null ? (required ? 'Not selected (required)' : 'Not selected') : file.name;
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Text('$label: $value'),
    );
  }

  Widget _buildResultCard(AssessmentResult result) {
    return Card(
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Assessment: ${result.childName}',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            Text('Age months: ${result.ageMonths.toStringAsFixed(1)} | Sex: ${result.sex}'),
            Text(
              'Predicted height: ${result.measurement.predictedHeightCm?.toStringAsFixed(1) ?? '-'} cm',
            ),
            Text(
              'Predicted weight: ${result.measurement.predictedWeightKg?.toStringAsFixed(1) ?? '-'} kg',
            ),
            Text('HAZ status: ${result.nutrition.hazStatus ?? '-'}'),
            Text('WHZ status: ${result.nutrition.whzStatus ?? '-'}'),
            if (result.mlPrediction != null)
              Text('ML wasting status: ${result.mlPrediction!.wastingStatus ?? '-'}'),
            if (result.muac != null)
              Text(
                'MUAC: ${result.muac!.muacCm?.toStringAsFixed(1) ?? '-'} cm (${result.muac!.muacStatus ?? '-'})',
              ),
            const SizedBox(height: 6),
            Text(result.summary),
          ],
        ),
      ),
    );
  }

  Widget _buildChildrenCard() {
    return Card(
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Registered Children', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ..._children.map(
              (c) => ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text('${c.name} (${c.sex})'),
                subtitle: Text('DOB: ${c.dateOfBirth} • Visits: ${c.visitCount}'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => _loadChildDetail(c.id),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChildDetailCard(ChildDetail child) {
    return Card(
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Child Detail: ${child.name}', style: Theme.of(context).textTheme.titleMedium),
            Text('DOB: ${child.dateOfBirth} • Sex: ${child.sex}'),
            Text('Guardian: ${child.guardianName ?? '-'}'),
            Text('Location: ${child.location ?? '-'}'),
            const SizedBox(height: 8),
            Text('Visits: ${child.visits.length}'),
            const SizedBox(height: 6),
            ...child.visits.map(
              (visit) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Text(
                  '${visit.visitDate ?? 'Unknown date'} | Age: ${visit.ageMonths?.toStringAsFixed(1) ?? '-'} mo | '
                  'HAZ: ${visit.measurement?.hazStatus ?? '-'} | WHZ: ${visit.measurement?.whzStatus ?? '-'}',
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
