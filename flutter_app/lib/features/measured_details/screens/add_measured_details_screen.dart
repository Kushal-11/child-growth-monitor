import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';

import '../../../providers/auth_provider.dart';
import '../../guided_capture/domain/capture_models.dart';
import '../domain/measured_details.dart';
import '../providers/measured_details_provider.dart';

class AddMeasuredDetailsScreen extends ConsumerStatefulWidget {
  const AddMeasuredDetailsScreen({
    super.key,
    required this.visitUuid,
    this.ownerUserId,
  });

  final String visitUuid;
  final int? ownerUserId;

  @override
  ConsumerState<AddMeasuredDetailsScreen> createState() =>
      _AddMeasuredDetailsScreenState();
}

class _AddMeasuredDetailsScreenState
    extends ConsumerState<AddMeasuredDetailsScreen> {
  final _formKey = GlobalKey<FormState>();
  final _height = TextEditingController();
  final _weight = TextEditingController();
  final _muac = TextEditingController();
  final _notes = TextEditingController();
  final _reason = TextEditingController();
  MeasurementMode _measurementMode = MeasurementMode.standingHeight;
  OedemaStatus _oedema = OedemaStatus.notChecked;
  String? _localError;

  @override
  void dispose() {
    _height.dispose();
    _weight.dispose();
    _muac.dispose();
    _notes.dispose();
    _reason.dispose();
    super.dispose();
  }

  int? get _ownerUserId =>
      widget.ownerUserId ?? ref.read(authProvider).user?.id;

  MeasuredVisitRequest? get _request {
    final ownerUserId = _ownerUserId;
    return ownerUserId == null
        ? null
        : MeasuredVisitRequest(
            visitUuid: widget.visitUuid,
            ownerUserId: ownerUserId,
          );
  }

  double? _optionalDouble(TextEditingController controller) {
    final value = controller.text.trim();
    return value.isEmpty ? null : double.parse(value);
  }

  Future<void> _save(
    MeasuredVisitRequest request,
    MeasuredVisitContext visit,
  ) async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _localError = null);
    try {
      final details = MeasuredDetails(
        measurementDate: visit.visitDate,
        measuredAt: DateTime.now(),
        measurementMode: _measurementMode,
        oedema: _oedema,
        heightCm: _optionalDouble(_height),
        weightKg: _optionalDouble(_weight),
        muacCm: _optionalDouble(_muac),
        notes: _notes.text.trim().isEmpty ? null : _notes.text.trim(),
        reason: _reason.text.trim().isEmpty ? null : _reason.text.trim(),
      );
      await ref.read(measuredDetailsProvider(request).notifier).save(
            editorUserId: request.ownerUserId,
            details: details,
          );
      if (!mounted) return;
      GoRouter.maybeOf(context)?.go('/visits/${widget.visitUuid}/report');
    } catch (error) {
      if (mounted) {
        setState(() {
          _localError = error is ArgumentError
              ? error.message?.toString()
              : error.toString();
        });
      }
    }
  }

  void _createNewVisit(MeasuredVisitContext visit) {
    GoRouter.maybeOf(context)?.go('/children/${visit.childId}/measure');
  }

  @override
  Widget build(BuildContext context) {
    final request = _request;
    if (request == null) {
      return const Scaffold(
        body: Center(child: Text('An authenticated operator is required')),
      );
    }
    final visit = ref.watch(measuredVisitContextProvider(request));
    final saveState = ref.watch(measuredDetailsProvider(request));
    return Scaffold(
      appBar: AppBar(title: const Text('Add Measured Details')),
      body: visit.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(child: Text(error.toString())),
        data: (visitContext) => Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Card(
                color: Theme.of(context).colorScheme.secondaryContainer,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        DateFormat('dd MMM yyyy')
                            .format(visitContext.visitDate),
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 6),
                      const Text(
                        'This visit date is locked so measurements remain '
                        'attached to the same photo assessment.',
                      ),
                      TextButton(
                        onPressed: () => _createNewVisit(visitContext),
                        child: const Text('Create a new visit instead'),
                      ),
                    ],
                  ),
                ),
              ),
              if (_localError != null || saveState.errorMessage != null) ...[
                const SizedBox(height: 8),
                Text(
                  _localError ?? saveState.errorMessage!,
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.error,
                  ),
                ),
              ],
              const SizedBox(height: 12),
              DropdownButtonFormField<MeasurementMode>(
                initialValue: _measurementMode,
                decoration: const InputDecoration(
                  labelText: 'Measurement mode',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(
                    value: MeasurementMode.standingHeight,
                    child: Text('Standing height'),
                  ),
                  DropdownMenuItem(
                    value: MeasurementMode.recumbentLength,
                    child: Text('Recumbent length'),
                  ),
                ],
                onChanged: saveState.saving
                    ? null
                    : (value) => setState(
                          () => _measurementMode =
                              value ?? MeasurementMode.standingHeight,
                        ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                key: const Key('measured_height'),
                controller: _height,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Height or length (cm, optional)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => MeasuredDetailsValidators.optionalText(
                  value,
                  label: 'height',
                  minimum: measuredHeightMinCm,
                  maximum: measuredHeightMaxCm,
                ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                key: const Key('measured_weight'),
                controller: _weight,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Scale weight (kg, optional)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => MeasuredDetailsValidators.optionalText(
                  value,
                  label: 'weight',
                  minimum: measuredWeightMinKg,
                  maximum: measuredWeightMaxKg,
                ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                key: const Key('measured_muac'),
                controller: _muac,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Tape MUAC (cm, optional)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => MeasuredDetailsValidators.optionalText(
                  value,
                  label: 'MUAC',
                  minimum: measuredMuacMinCm,
                  maximum: measuredMuacMaxCm,
                ),
              ),
              if (visitContext.ageMonths < 6 || visitContext.ageMonths >= 60)
                const Padding(
                  padding: EdgeInsets.only(top: 6),
                  child: Text(
                    'MUAC will be stored but is not classification-eligible '
                    'outside 6–59 completed months.',
                  ),
                ),
              const SizedBox(height: 12),
              DropdownButtonFormField<OedemaStatus>(
                initialValue: _oedema,
                decoration: const InputDecoration(
                  labelText: 'Bilateral pitting oedema',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(
                    value: OedemaStatus.notChecked,
                    child: Text('Not checked'),
                  ),
                  DropdownMenuItem(
                    value: OedemaStatus.no,
                    child: Text('No'),
                  ),
                  DropdownMenuItem(
                    value: OedemaStatus.yes,
                    child: Text('Yes'),
                  ),
                ],
                onChanged: saveState.saving
                    ? null
                    : (value) => setState(
                          () => _oedema = value ?? OedemaStatus.notChecked,
                        ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _notes,
                maxLines: 3,
                maxLength: 2000,
                decoration: const InputDecoration(
                  labelText: 'Notes (optional)',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _reason,
                maxLength: 500,
                decoration: const InputDecoration(
                  labelText: 'Update reason (optional)',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: saveState.saving
                    ? null
                    : () => _save(request, visitContext),
                icon: const Icon(Icons.save),
                label: Text(
                  saveState.saving ? 'Saving…' : 'Save measured details',
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
