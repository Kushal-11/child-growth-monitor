import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/services/image_storage_service.dart';

void main() {
  late Directory tempRoot;
  late ImageStorageService svc;

  setUp(() async {
    tempRoot = await Directory.systemTemp.createTemp('cgm_test_');
    svc = ImageStorageService(rootOverride: tempRoot);
  });

  tearDown(() async {
    if (await tempRoot.exists()) {
      await tempRoot.delete(recursive: true);
    }
  });

  test('persist copies file into images dir and returns new path', () async {
    final src = File('${tempRoot.path}/src.jpg');
    await src.writeAsBytes(List.filled(2048, 0xAB));

    final newPath = await svc.persist(src.path);
    expect(File(newPath).existsSync(), isTrue);
    expect(newPath, contains('${tempRoot.path}/images/'));
    expect((await File(newPath).readAsBytes()).length, 2048);
  });

  test('totalUsedBytes returns sum of all image bytes', () async {
    final a = File('${tempRoot.path}/a.jpg')
      ..writeAsBytesSync(List.filled(1000, 1));
    final b = File('${tempRoot.path}/b.jpg')
      ..writeAsBytesSync(List.filled(500, 2));
    await svc.persist(a.path);
    await svc.persist(b.path);
    expect(await svc.totalUsedBytes(), 1500);
  });

  test('guarded cleanup deletes only explicitly acknowledged files', () async {
    final acknowledged = File('${tempRoot.path}/ack.jpg')
      ..writeAsBytesSync([1, 2, 3]);
    final pending = File('${tempRoot.path}/pending.jpg')
      ..writeAsBytesSync([4, 5, 6]);
    final acknowledgedPath = await svc.persist(acknowledged.path);
    final pendingPath = await svc.persist(pending.path);

    final deleted = await svc.deleteAcknowledged([acknowledgedPath]);

    expect(deleted, 1);
    expect(File(acknowledgedPath).existsSync(), isFalse);
    expect(File(pendingPath).existsSync(), isTrue);
    expect(await svc.totalUsedBytes(), 3);
  });

  test('guarded cleanup refuses paths outside managed image storage', () async {
    final external = File('${tempRoot.path}/outside.jpg')
      ..writeAsBytesSync([1, 2, 3]);

    expect(
      () => svc.deleteAcknowledged([external.path]),
      throwsA(isA<FileSystemException>()),
    );
    expect(external.existsSync(), isTrue);
  });
}
