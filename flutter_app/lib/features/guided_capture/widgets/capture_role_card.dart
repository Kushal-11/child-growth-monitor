import 'package:flutter/material.dart';

import '../domain/capture_models.dart';

class CaptureRoleCard extends StatelessWidget {
  const CaptureRoleCard({
    super.key,
    required this.role,
    required this.capturedFrameCount,
  });

  final CaptureAssetRole role;
  final int capturedFrameCount;

  bool get _required => CaptureAssetRole.requiredRoles.contains(role);

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    captureRoleTitle(role),
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                ),
                Chip(label: Text(_required ? 'Required' : 'Optional')),
              ],
            ),
            const SizedBox(height: 8),
            Text(captureRoleGuidance(role)),
            const SizedBox(height: 12),
            Row(
              children: [
                Icon(
                  capturedFrameCount > 0
                      ? Icons.check_circle
                      : Icons.radio_button_unchecked,
                  color:
                      capturedFrameCount > 0 ? colors.primary : colors.outline,
                ),
                const SizedBox(width: 8),
                Text(
                  capturedFrameCount > 0
                      ? '$capturedFrameCount accepted frame(s)'
                      : 'Not captured',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

String captureRoleTitle(CaptureAssetRole role) => switch (role) {
      CaptureAssetRole.front => 'Front full-body view',
      CaptureAssetRole.side => 'Side full-body view',
      CaptureAssetRole.back => 'Back full-body view',
      CaptureAssetRole.armFront => 'Front upper-arm view',
      CaptureAssetRole.armSide => 'Side upper-arm view',
    };

String captureRoleShortLabel(CaptureAssetRole role) => switch (role) {
      CaptureAssetRole.front => 'Front view',
      CaptureAssetRole.side => 'Side view',
      CaptureAssetRole.back => 'Back view',
      CaptureAssetRole.armFront => 'Front arm view',
      CaptureAssetRole.armSide => 'Side arm view',
    };

String captureRoleActionLabel(CaptureAssetRole role) => switch (role) {
      CaptureAssetRole.front => 'Capture front view',
      CaptureAssetRole.side => 'Capture side view',
      CaptureAssetRole.back => 'Capture back view',
      CaptureAssetRole.armFront => 'Capture front arm view',
      CaptureAssetRole.armSide => 'Capture side arm view',
    };

String captureRoleGuidance(CaptureAssetRole role) => switch (role) {
      CaptureAssetRole.front =>
        'Keep the child facing the camera with head and both feet visible.',
      CaptureAssetRole.side =>
        'Turn the child sideways and keep the complete body visible.',
      CaptureAssetRole.back =>
        'Keep the child facing away with the complete body visible.',
      CaptureAssetRole.armFront =>
        'Keep the upper arm visible from the front without covering it.',
      CaptureAssetRole.armSide =>
        'Keep the upper arm visible from the side without covering it.',
    };
