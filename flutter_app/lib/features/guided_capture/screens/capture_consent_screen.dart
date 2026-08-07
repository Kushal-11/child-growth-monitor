import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../providers/auth_provider.dart';
import '../providers/guided_capture_provider.dart';

class CaptureConsentScreen extends ConsumerStatefulWidget {
  const CaptureConsentScreen({
    super.key,
    required this.childId,
    this.ownerUserId,
    this.operatorIdentifier,
  });

  final int childId;
  final int? ownerUserId;
  final String? operatorIdentifier;

  @override
  ConsumerState<CaptureConsentScreen> createState() =>
      _CaptureConsentScreenState();
}

class _CaptureConsentScreenState extends ConsumerState<CaptureConsentScreen> {
  bool _starting = false;
  String? _localError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _initialize());
  }

  Future<void> _initialize() async {
    final user = ref.read(authProvider).user;
    final ownerUserId = widget.ownerUserId ?? user?.id;
    if (ownerUserId == null) {
      setState(() => _localError = 'An authenticated operator is required');
      return;
    }
    try {
      await ref.read(guidedCaptureProvider.notifier).initializeNew(
            childId: widget.childId,
            ownerUserId: ownerUserId,
          );
    } catch (_) {
      // Provider state contains the owner-scoped error.
    }
  }

  Future<void> _accept() async {
    final user = ref.read(authProvider).user;
    final operator = widget.operatorIdentifier ?? user?.username;
    if (operator == null) {
      setState(() => _localError = 'An authenticated operator is required');
      return;
    }
    setState(() => _starting = true);
    try {
      final visitUuid =
          await ref.read(guidedCaptureProvider.notifier).acceptConsent(
                operatorIdentifier: operator,
                deviceMetadataJson: jsonEncode({
                  'workflow': 'guided_capture',
                  'entry': 'child_profile',
                }),
              );
      if (mounted) context.push('/visits/$visitUuid/capture');
    } catch (error) {
      if (mounted) setState(() => _localError = error.toString());
    } finally {
      if (mounted) setState(() => _starting = false);
    }
  }

  void _decline() {
    ref.read(guidedCaptureProvider.notifier).declineConsent();
    if (context.canPop()) {
      context.pop();
    } else {
      context.go('/children/${widget.childId}');
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(guidedCaptureProvider);
    if (state.phase == GuidedCapturePhase.loading ||
        state.phase == GuidedCapturePhase.idle) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final error = _localError ?? state.errorMessage;
    return Scaffold(
      appBar: AppBar(title: const Text('Photo assessment consent')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Icon(
              Icons.privacy_tip_outlined,
              size: 56,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 16),
            Text(
              state.child?.name ?? 'Child profile',
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Text(
                  'Confirm that the caregiver agrees to child photographs '
                  'being used for estimated growth screening and model '
                  'evaluation. Results are estimates from photos and are not '
                  'a replacement for measured clinical assessment.',
                ),
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'The visit is created only after consent. No height, weight, '
              'or MUAC is requested during photo capture.',
            ),
            if (error != null) ...[
              const SizedBox(height: 12),
              Text(error, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _starting ? null : _accept,
              icon: const Icon(Icons.check),
              label: Text(
                _starting ? 'Creating visit…' : 'I have caregiver consent',
              ),
            ),
            const SizedBox(height: 8),
            OutlinedButton(
              onPressed: _starting ? null : _decline,
              child: const Text('Decline and return'),
            ),
          ],
        ),
      ),
    );
  }
}
