import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

/// Manages the lifecycle of captured images on device storage.
///
/// All images live under `<app documents>/images/`. Cleanup is allow-list
/// based so pending or failed guided-capture media cannot be deleted broadly.
class ImageStorageService {
  ImageStorageService({Directory? rootOverride}) : _rootOverride = rootOverride;

  final Directory? _rootOverride;
  static const _uuid = Uuid();

  Future<Directory> _imagesDir() async {
    final base = _rootOverride ?? await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(base.path, 'images'));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  /// Copies [tempPath] into the persistent images directory and returns the
  /// new absolute path.
  Future<String> persist(String tempPath) async {
    final src = File(tempPath);
    if (!await src.exists()) {
      throw FileSystemException('Source image not found', tempPath);
    }
    final ext = p.extension(tempPath).isEmpty ? '.jpg' : p.extension(tempPath);
    final dir = await _imagesDir();
    final dst = File(p.join(dir.path, '${_uuid.v4()}$ext'));
    await src.copy(dst.path);
    return dst.path;
  }

  /// Sum of bytes for every file under the images directory.
  Future<int> totalUsedBytes() async {
    final dir = await _imagesDir();
    var total = 0;
    await for (final entity in dir.list(recursive: true, followLinks: false)) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  /// Deletes only the managed files explicitly supplied by a caller that has
  /// already verified their individual server acknowledgements.
  Future<int> deleteAcknowledged(Iterable<String> acknowledgedPaths) async {
    final dir = await _imagesDir();
    final canonicalRoot = await dir.resolveSymbolicLinks();
    var deleted = 0;
    for (final rawPath in acknowledgedPaths.toSet()) {
      final file = File(rawPath);
      if (!await file.exists()) continue;
      final canonicalFile = await file.resolveSymbolicLinks();
      if (!p.isWithin(canonicalRoot, canonicalFile)) {
        throw FileSystemException(
          'Refusing to delete media outside managed image storage',
          rawPath,
        );
      }
      await file.delete();
      deleted += 1;
    }
    return deleted;
  }
}
