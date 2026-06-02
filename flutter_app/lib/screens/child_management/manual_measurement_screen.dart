import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';

import '../../constants/config.dart';
import '../../providers/assessment_service_provider.dart';
import '../../providers/auth_provider.dart';
import '../../providers/database_provider.dart';
import '../../providers/sync_provider.dart';
import '../../services/muac_service.dart';
import '../../services/nutrition_service.dart';

/// Manual monthly measurement entry.
///
/// SAFETY: this screen always computes WHO HAZ/WHZ z-scores via the on-device
/// [NutritionService] and MUAC status via [MuacService]. There is no path that
/// saves a record without running this z-score pipeline. The persistence is a
/// single atomic transaction (see ManualVisitDao); any error before/within it
/// surfaces to the user and leaves nothing partial.
class ManualMeasurementScreen extends ConsumerStatefulWidget {
  const ManualMeasurementScreen({super.key, required this.childId});
  final int childId;

  @override
  ConsumerState<ManualMeasurementScreen> createState() =>
      _ManualMeasurementScreenState();
}

class _ManualMeasurementScreenState
    extends ConsumerState<ManualMeasurementScreen> {
  final _formKey = GlobalKey<FormState>();
  final _height = TextEditingController();
  final _weight = TextEditingController();
  final _muac = TextEditingController();
  final _notes = TextEditingController();
  DateTime _visitDate = DateTime.now();
  bool _saving = false;
  String? _error;

  @override
  void dispose() {
    _height.dispose();
    _weight.dispose();
    _muac.dispose();
    _notes.dispose();
    super.dispose();
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _visitDate,
      firstDate: DateTime(2015),
      lastDate: DateTime.now(),
    );
    if (picked != null && mounted) setState(() => _visitDate = picked);
  }

  String? _positiveNumber(String? v) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final n = double.tryParse(v);
    if (n == null || n <= 0) return 'Enter a positive number';
    return null;
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      final child = await ref.read(childDaoProvider).getById(widget.childId);
      if (child == null) throw Exception('Child not found');
      final ageMonths =
          _visitDate.difference(DateTime.parse(child.dateOfBirth)).inDays /
              30.4375;
      final heightCm = double.parse(_height.text);
      final weightKg = double.parse(_weight.text);
      final muacCm = _muac.text.trim().isEmpty ? null : double.parse(_muac.text);

      // SAFETY: WHO z-scores via the existing on-device pipeline.
      final who = await ref.read(whoDataServiceProvider.future);
      final nutrition = NutritionService(who);
      final haz = nutrition.computeHaz(child.sex, ageMonths.round(), heightCm);
      final whz =
          nutrition.computeWhz(child.sex, ageMonths, heightCm, weightKg);
      final hazStatus = haz != null ? classifyHaz(haz) : null;
      final whzStatus = whz != null ? classifyWhz(whz) : null;
      // SAFETY: MUAC status via MuacService (manual value wins, classified by WHO thresholds).
      final muacResult = MuacService.estimate(
        ageMonths: ageMonths,
        sex: child.sex,
        whz: whz,
        manualMuacCm: muacCm,
      );

      final ownerId = ref.read(authProvider).user?.id;
      final visitId =
          await ref.read(manualVisitDaoProvider).createManualVisit(
                childId: widget.childId,
                ownerUserId: ownerId,
                ageMonths: ageMonths,
                visitDate: _visitDate,
                heightCm: heightCm,
                weightKg: weightKg,
                muacCm: muacCm,
                hazZscore: haz,
                whzZscore: whz,
                hazStatus: hazStatus,
                whzStatus: whzStatus,
                muacStatus: muacResult.muacStatus,
                notes: _notes.text.trim().isEmpty ? null : _notes.text.trim(),
              );
      // Opportunistic sync (fire-and-forget).
      ref.read(syncServiceProvider).runOnce();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Measurement saved (visit $visitId)')),
        );
        context.pop();
      }
    } catch (e) {
      // SAFETY: no partial save — the DAO transaction is atomic, so any throw
      // leaves nothing persisted. Surface the error inline.
      if (mounted) setState(() => _error = 'Could not save: $e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Monthly measurement')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            if (_error != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(
                  _error!,
                  style:
                      TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ),
            InputDecorator(
              decoration: const InputDecoration(
                labelText: 'Visit date',
                border: OutlineInputBorder(),
              ),
              child: InkWell(
                onTap: _pickDate,
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Text(DateFormat('yyyy-MM-dd').format(_visitDate)),
                ),
              ),
            ),
            const SizedBox(height: 12),
            TextFormField(
              key: const Key('measure_height'),
              controller: _height,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(
                labelText: 'Height (cm)',
                border: OutlineInputBorder(),
              ),
              validator: _positiveNumber,
            ),
            const SizedBox(height: 12),
            TextFormField(
              key: const Key('measure_weight'),
              controller: _weight,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(
                labelText: 'Weight (kg)',
                border: OutlineInputBorder(),
              ),
              validator: _positiveNumber,
            ),
            const SizedBox(height: 12),
            TextFormField(
              key: const Key('measure_muac'),
              controller: _muac,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(
                labelText: 'MUAC (cm, optional)',
                border: OutlineInputBorder(),
              ),
              validator: (v) =>
                  (v == null || v.trim().isEmpty) ? null : _positiveNumber(v),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _notes,
              maxLines: 3,
              decoration: const InputDecoration(
                labelText: 'Notes (optional)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton(
              key: const Key('measure_save'),
              onPressed: _saving ? null : _save,
              child: _saving
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Save measurement'),
            ),
          ],
        ),
      ),
    );
  }
}
