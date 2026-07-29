import 'dart:io';

import 'package:flutter/material.dart';

import '../../../theme/app_colors.dart';
import '../../../theme/app_spacing.dart';

class PhotoGuidanceTile extends StatelessWidget {
  const PhotoGuidanceTile({
    super.key,
    required this.role,
    required this.title,
    required this.help,
    required this.cameraLabel,
    required this.galleryLabel,
    required this.onCamera,
    required this.onGallery,
    this.imagePath,
    this.optionalLabel,
    this.onRemove,
    this.showGuide = false,
    this.errorText,
  });

  final String role;
  final String title;
  final String help;
  final String cameraLabel;
  final String galleryLabel;
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  final String? imagePath;
  final String? optionalLabel;
  final VoidCallback? onRemove;
  final bool showGuide;
  final String? errorText;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                title,
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
            if (optionalLabel != null)
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 4,
                ),
                decoration: BoxDecoration(
                  color: AppColors.primaryContainer,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  optionalLabel!,
                  style: Theme.of(context).textTheme.labelMedium?.copyWith(
                        color: AppColors.primary,
                        fontWeight: FontWeight.w600,
                      ),
                ),
              ),
          ],
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(help, style: Theme.of(context).textTheme.bodyMedium),
        const SizedBox(height: AppSpacing.md),
        Container(
          height: showGuide ? 250 : 154,
          width: double.infinity,
          clipBehavior: Clip.antiAlias,
          decoration: BoxDecoration(
            color: const Color(0xFFEDF7F4),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: errorText == null ? AppColors.outline : AppColors.error,
            ),
          ),
          child: imagePath == null
              ? _EmptyPhoto(
                  showGuide: showGuide,
                  cameraLabel: cameraLabel,
                  galleryLabel: galleryLabel,
                  onCamera: onCamera,
                  onGallery: onGallery,
                  role: role,
                )
              : Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.file(File(imagePath!), fit: BoxFit.cover),
                    Positioned(
                      top: AppSpacing.sm,
                      right: AppSpacing.sm,
                      child: IconButton.filledTonal(
                        key: Key('photo_${role}_remove'),
                        tooltip: 'Remove',
                        onPressed: onRemove,
                        icon: const Icon(Icons.close_rounded),
                      ),
                    ),
                  ],
                ),
        ),
        if (errorText != null) ...[
          const SizedBox(height: 6),
          Text(
            errorText!,
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: AppColors.error),
          ),
        ],
      ],
    );
  }
}

class _EmptyPhoto extends StatelessWidget {
  const _EmptyPhoto({
    required this.showGuide,
    required this.cameraLabel,
    required this.galleryLabel,
    required this.onCamera,
    required this.onGallery,
    required this.role,
  });

  final bool showGuide;
  final String cameraLabel;
  final String galleryLabel;
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  final String role;

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        if (showGuide)
          Positioned(
            top: 16,
            bottom: 58,
            child: Image.asset(
              'assets/images/body_positioning_guide.png',
              fit: BoxFit.contain,
            ),
          )
        else
          const Positioned(
            top: 28,
            child: Icon(
              Icons.add_a_photo_outlined,
              size: 42,
              color: AppColors.guide,
            ),
          ),
        Positioned(
          left: AppSpacing.md,
          right: AppSpacing.md,
          bottom: AppSpacing.md,
          child: Row(
            children: [
              Expanded(
                child: FilledButton.tonalIcon(
                  key: Key('photo_${role}_camera'),
                  onPressed: onCamera,
                  icon: const Icon(Icons.camera_alt_outlined),
                  label: Text(cameraLabel),
                ),
              ),
              const SizedBox(width: AppSpacing.sm),
              Expanded(
                child: OutlinedButton.icon(
                  key: Key('photo_${role}_gallery'),
                  onPressed: onGallery,
                  icon: const Icon(Icons.photo_library_outlined),
                  label: Text(galleryLabel),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
