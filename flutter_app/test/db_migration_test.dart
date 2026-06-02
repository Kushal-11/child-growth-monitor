import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';

void main() {
  test('schema v3 has new columns and inserts work', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final id = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Kid',
            dateOfBirth: '2024-01-01',
            sex: 'M',
          ),
        );
    final child = await (db.select(db.children)..where((c) => c.id.equals(id)))
        .getSingle();
    expect(child.isArchived, false);
    expect(child.ownerUserId, isNull);
    expect(child.photoPath, isNull);
    await db.close();
  });
}
