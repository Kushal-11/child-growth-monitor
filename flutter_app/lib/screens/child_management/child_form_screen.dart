import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';

import '../../providers/auth_provider.dart';
import '../../providers/database_provider.dart';
import '../../services/image_storage_service.dart';

class ChildFormScreen extends ConsumerStatefulWidget {
  const ChildFormScreen({super.key, this.childId});
  final int? childId;

  @override
  ConsumerState<ChildFormScreen> createState() => _ChildFormScreenState();
}

class _ChildFormScreenState extends ConsumerState<ChildFormScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _guardian = TextEditingController();
  final _location = TextEditingController();
  DateTime? _dob;
  String _sex = 'M';
  String? _photoPath;
  bool _loading = false;
  bool _saving = false;

  bool get _isEdit => widget.childId != null;

  @override
  void initState() {
    super.initState();
    if (_isEdit) _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    final child = await ref.read(childDaoProvider).getById(widget.childId!);
    if (child != null) {
      _name.text = child.name;
      _guardian.text = child.guardianName ?? '';
      _location.text = child.location ?? '';
      _dob = DateTime.tryParse(child.dateOfBirth);
      _sex = child.sex;
      _photoPath = child.photoPath;
    }
    if (mounted) setState(() => _loading = false);
  }

  @override
  void dispose() {
    _name.dispose();
    _guardian.dispose();
    _location.dispose();
    super.dispose();
  }

  Future<void> _pickPhoto() async {
    final picked = await ImagePicker().pickImage(
      source: ImageSource.camera,
      maxWidth: 1024,
    );
    if (picked == null) return;
    final stored = await ImageStorageService().persist(picked.path);
    if (mounted) setState(() => _photoPath = stored);
  }

  Future<void> _pickDob() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: _dob ?? DateTime(now.year - 1, now.month, now.day),
      firstDate: DateTime(now.year - 6),
      lastDate: now,
    );
    if (picked != null && mounted) setState(() => _dob = picked);
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    if (_dob == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select date of birth')),
      );
      return;
    }
    setState(() => _saving = true);
    final dao = ref.read(childDaoProvider);
    final dobStr = DateFormat('yyyy-MM-dd').format(_dob!);
    final ownerId = ref.read(authProvider).user?.id;
    try {
      if (_isEdit) {
        await dao.updateChild(
          id: widget.childId!,
          name: _name.text.trim(),
          guardianName:
              _guardian.text.trim().isEmpty ? null : _guardian.text.trim(),
          location:
              _location.text.trim().isEmpty ? null : _location.text.trim(),
          photoPath: _photoPath,
        );
        if (mounted) context.pop();
      } else {
        final id = await dao.createChild(
          name: _name.text.trim(),
          dateOfBirth: dobStr,
          sex: _sex,
          guardianName:
              _guardian.text.trim().isEmpty ? null : _guardian.text.trim(),
          location:
              _location.text.trim().isEmpty ? null : _location.text.trim(),
          ownerUserId: ownerId,
          photoPath: _photoPath,
        );
        // Replace only the form, preserving the children list underneath it.
        // This makes both the app-bar and system back buttons return to the
        // list instead of leaving a stale completed form in the stack.
        if (mounted) context.pushReplacement('/children/$id');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Save failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: Text(_isEdit ? 'Edit child' : 'New child')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Center(
              child: Column(
                children: [
                  CircleAvatar(
                    radius: 48,
                    backgroundImage: _photoPath != null
                        ? FileImage(File(_photoPath!))
                        : null,
                    child: _photoPath == null
                        ? const Icon(Icons.person, size: 48)
                        : null,
                  ),
                  TextButton.icon(
                    key: const Key('child_photo_btn'),
                    onPressed: _pickPhoto,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Photo'),
                  ),
                ],
              ),
            ),
            TextFormField(
              key: const Key('child_name'),
              controller: _name,
              decoration: const InputDecoration(
                labelText: 'Name',
                border: OutlineInputBorder(),
              ),
              validator: (v) =>
                  (v == null || v.trim().isEmpty) ? 'Name is required' : null,
            ),
            const SizedBox(height: 12),
            InputDecorator(
              decoration: const InputDecoration(
                labelText: 'Date of birth',
                border: OutlineInputBorder(),
              ),
              child: InkWell(
                onTap: _pickDob,
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Text(
                    _dob == null
                        ? 'Select date'
                        : DateFormat('yyyy-MM-dd').format(_dob!),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: _sex,
              decoration: const InputDecoration(
                labelText: 'Sex',
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem(value: 'M', child: Text('Male')),
                DropdownMenuItem(value: 'F', child: Text('Female')),
              ],
              onChanged:
                  _isEdit ? null : (v) => setState(() => _sex = v ?? 'M'),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _guardian,
              decoration: const InputDecoration(
                labelText: 'Guardian (optional)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _location,
              decoration: const InputDecoration(
                labelText: 'Location (optional)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton(
              key: const Key('child_save'),
              onPressed: _saving ? null : _save,
              child: _saving
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Text(_isEdit ? 'Save changes' : 'Create child'),
            ),
          ],
        ),
      ),
    );
  }
}
