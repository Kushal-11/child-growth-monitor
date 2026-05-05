import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

/// Manages the lifecycle of captured images on device storage.
///
/// All images live under `<app documents>/images/`. The service never
/// auto-deletes — callers (or the user via Settings) trigger `clearAll`.
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

  /// Deletes every file in the images directory. The directory itself remains.
  Future<void> clearAll() async {
    final dir = await _imagesDir();
    await for (final entity in dir.list()) {
      if (entity is File) await entity.delete();
    }
  }
}
