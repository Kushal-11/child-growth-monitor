// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'database.dart';

// ignore_for_file: type=lint
class $ChildrenTable extends Children
    with TableInfo<$ChildrenTable, ChildrenData> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ChildrenTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
      'name', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _dateOfBirthMeta =
      const VerificationMeta('dateOfBirth');
  @override
  late final GeneratedColumn<String> dateOfBirth = GeneratedColumn<String>(
      'date_of_birth', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _sexMeta = const VerificationMeta('sex');
  @override
  late final GeneratedColumn<String> sex = GeneratedColumn<String>(
      'sex', aliasedName, false,
      additionalChecks:
          GeneratedColumn.checkTextLength(minTextLength: 1, maxTextLength: 1),
      type: DriftSqlType.string,
      requiredDuringInsert: true);
  static const VerificationMeta _guardianNameMeta =
      const VerificationMeta('guardianName');
  @override
  late final GeneratedColumn<String> guardianName = GeneratedColumn<String>(
      'guardian_name', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _locationMeta =
      const VerificationMeta('location');
  @override
  late final GeneratedColumn<String> location = GeneratedColumn<String>(
      'location', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _createdAtMeta =
      const VerificationMeta('createdAt');
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
      'created_at', aliasedName, false,
      type: DriftSqlType.dateTime,
      requiredDuringInsert: false,
      defaultValue: currentDateAndTime);
  static const VerificationMeta _updatedAtMeta =
      const VerificationMeta('updatedAt');
  @override
  late final GeneratedColumn<DateTime> updatedAt = GeneratedColumn<DateTime>(
      'updated_at', aliasedName, false,
      type: DriftSqlType.dateTime,
      requiredDuringInsert: false,
      defaultValue: currentDateAndTime);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        name,
        dateOfBirth,
        sex,
        guardianName,
        location,
        createdAt,
        updatedAt
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'children';
  @override
  VerificationContext validateIntegrity(Insertable<ChildrenData> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('name')) {
      context.handle(
          _nameMeta, name.isAcceptableOrUnknown(data['name']!, _nameMeta));
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('date_of_birth')) {
      context.handle(
          _dateOfBirthMeta,
          dateOfBirth.isAcceptableOrUnknown(
              data['date_of_birth']!, _dateOfBirthMeta));
    } else if (isInserting) {
      context.missing(_dateOfBirthMeta);
    }
    if (data.containsKey('sex')) {
      context.handle(
          _sexMeta, sex.isAcceptableOrUnknown(data['sex']!, _sexMeta));
    } else if (isInserting) {
      context.missing(_sexMeta);
    }
    if (data.containsKey('guardian_name')) {
      context.handle(
          _guardianNameMeta,
          guardianName.isAcceptableOrUnknown(
              data['guardian_name']!, _guardianNameMeta));
    }
    if (data.containsKey('location')) {
      context.handle(_locationMeta,
          location.isAcceptableOrUnknown(data['location']!, _locationMeta));
    }
    if (data.containsKey('created_at')) {
      context.handle(_createdAtMeta,
          createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta));
    }
    if (data.containsKey('updated_at')) {
      context.handle(_updatedAtMeta,
          updatedAt.isAcceptableOrUnknown(data['updated_at']!, _updatedAtMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  ChildrenData map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return ChildrenData(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      name: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}name'])!,
      dateOfBirth: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}date_of_birth'])!,
      sex: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}sex'])!,
      guardianName: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}guardian_name']),
      location: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}location']),
      createdAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}created_at'])!,
      updatedAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}updated_at'])!,
    );
  }

  @override
  $ChildrenTable createAlias(String alias) {
    return $ChildrenTable(attachedDatabase, alias);
  }
}

class ChildrenData extends DataClass implements Insertable<ChildrenData> {
  final int id;
  final String name;
  final String dateOfBirth;
  final String sex;
  final String? guardianName;
  final String? location;
  final DateTime createdAt;
  final DateTime updatedAt;
  const ChildrenData(
      {required this.id,
      required this.name,
      required this.dateOfBirth,
      required this.sex,
      this.guardianName,
      this.location,
      required this.createdAt,
      required this.updatedAt});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['name'] = Variable<String>(name);
    map['date_of_birth'] = Variable<String>(dateOfBirth);
    map['sex'] = Variable<String>(sex);
    if (!nullToAbsent || guardianName != null) {
      map['guardian_name'] = Variable<String>(guardianName);
    }
    if (!nullToAbsent || location != null) {
      map['location'] = Variable<String>(location);
    }
    map['created_at'] = Variable<DateTime>(createdAt);
    map['updated_at'] = Variable<DateTime>(updatedAt);
    return map;
  }

  ChildrenCompanion toCompanion(bool nullToAbsent) {
    return ChildrenCompanion(
      id: Value(id),
      name: Value(name),
      dateOfBirth: Value(dateOfBirth),
      sex: Value(sex),
      guardianName: guardianName == null && nullToAbsent
          ? const Value.absent()
          : Value(guardianName),
      location: location == null && nullToAbsent
          ? const Value.absent()
          : Value(location),
      createdAt: Value(createdAt),
      updatedAt: Value(updatedAt),
    );
  }

  factory ChildrenData.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return ChildrenData(
      id: serializer.fromJson<int>(json['id']),
      name: serializer.fromJson<String>(json['name']),
      dateOfBirth: serializer.fromJson<String>(json['dateOfBirth']),
      sex: serializer.fromJson<String>(json['sex']),
      guardianName: serializer.fromJson<String?>(json['guardianName']),
      location: serializer.fromJson<String?>(json['location']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
      updatedAt: serializer.fromJson<DateTime>(json['updatedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'name': serializer.toJson<String>(name),
      'dateOfBirth': serializer.toJson<String>(dateOfBirth),
      'sex': serializer.toJson<String>(sex),
      'guardianName': serializer.toJson<String?>(guardianName),
      'location': serializer.toJson<String?>(location),
      'createdAt': serializer.toJson<DateTime>(createdAt),
      'updatedAt': serializer.toJson<DateTime>(updatedAt),
    };
  }

  ChildrenData copyWith(
          {int? id,
          String? name,
          String? dateOfBirth,
          String? sex,
          Value<String?> guardianName = const Value.absent(),
          Value<String?> location = const Value.absent(),
          DateTime? createdAt,
          DateTime? updatedAt}) =>
      ChildrenData(
        id: id ?? this.id,
        name: name ?? this.name,
        dateOfBirth: dateOfBirth ?? this.dateOfBirth,
        sex: sex ?? this.sex,
        guardianName:
            guardianName.present ? guardianName.value : this.guardianName,
        location: location.present ? location.value : this.location,
        createdAt: createdAt ?? this.createdAt,
        updatedAt: updatedAt ?? this.updatedAt,
      );
  ChildrenData copyWithCompanion(ChildrenCompanion data) {
    return ChildrenData(
      id: data.id.present ? data.id.value : this.id,
      name: data.name.present ? data.name.value : this.name,
      dateOfBirth:
          data.dateOfBirth.present ? data.dateOfBirth.value : this.dateOfBirth,
      sex: data.sex.present ? data.sex.value : this.sex,
      guardianName: data.guardianName.present
          ? data.guardianName.value
          : this.guardianName,
      location: data.location.present ? data.location.value : this.location,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
      updatedAt: data.updatedAt.present ? data.updatedAt.value : this.updatedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('ChildrenData(')
          ..write('id: $id, ')
          ..write('name: $name, ')
          ..write('dateOfBirth: $dateOfBirth, ')
          ..write('sex: $sex, ')
          ..write('guardianName: $guardianName, ')
          ..write('location: $location, ')
          ..write('createdAt: $createdAt, ')
          ..write('updatedAt: $updatedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
      id, name, dateOfBirth, sex, guardianName, location, createdAt, updatedAt);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ChildrenData &&
          other.id == this.id &&
          other.name == this.name &&
          other.dateOfBirth == this.dateOfBirth &&
          other.sex == this.sex &&
          other.guardianName == this.guardianName &&
          other.location == this.location &&
          other.createdAt == this.createdAt &&
          other.updatedAt == this.updatedAt);
}

class ChildrenCompanion extends UpdateCompanion<ChildrenData> {
  final Value<int> id;
  final Value<String> name;
  final Value<String> dateOfBirth;
  final Value<String> sex;
  final Value<String?> guardianName;
  final Value<String?> location;
  final Value<DateTime> createdAt;
  final Value<DateTime> updatedAt;
  const ChildrenCompanion({
    this.id = const Value.absent(),
    this.name = const Value.absent(),
    this.dateOfBirth = const Value.absent(),
    this.sex = const Value.absent(),
    this.guardianName = const Value.absent(),
    this.location = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.updatedAt = const Value.absent(),
  });
  ChildrenCompanion.insert({
    this.id = const Value.absent(),
    required String name,
    required String dateOfBirth,
    required String sex,
    this.guardianName = const Value.absent(),
    this.location = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.updatedAt = const Value.absent(),
  })  : name = Value(name),
        dateOfBirth = Value(dateOfBirth),
        sex = Value(sex);
  static Insertable<ChildrenData> custom({
    Expression<int>? id,
    Expression<String>? name,
    Expression<String>? dateOfBirth,
    Expression<String>? sex,
    Expression<String>? guardianName,
    Expression<String>? location,
    Expression<DateTime>? createdAt,
    Expression<DateTime>? updatedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (name != null) 'name': name,
      if (dateOfBirth != null) 'date_of_birth': dateOfBirth,
      if (sex != null) 'sex': sex,
      if (guardianName != null) 'guardian_name': guardianName,
      if (location != null) 'location': location,
      if (createdAt != null) 'created_at': createdAt,
      if (updatedAt != null) 'updated_at': updatedAt,
    });
  }

  ChildrenCompanion copyWith(
      {Value<int>? id,
      Value<String>? name,
      Value<String>? dateOfBirth,
      Value<String>? sex,
      Value<String?>? guardianName,
      Value<String?>? location,
      Value<DateTime>? createdAt,
      Value<DateTime>? updatedAt}) {
    return ChildrenCompanion(
      id: id ?? this.id,
      name: name ?? this.name,
      dateOfBirth: dateOfBirth ?? this.dateOfBirth,
      sex: sex ?? this.sex,
      guardianName: guardianName ?? this.guardianName,
      location: location ?? this.location,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (dateOfBirth.present) {
      map['date_of_birth'] = Variable<String>(dateOfBirth.value);
    }
    if (sex.present) {
      map['sex'] = Variable<String>(sex.value);
    }
    if (guardianName.present) {
      map['guardian_name'] = Variable<String>(guardianName.value);
    }
    if (location.present) {
      map['location'] = Variable<String>(location.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    if (updatedAt.present) {
      map['updated_at'] = Variable<DateTime>(updatedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ChildrenCompanion(')
          ..write('id: $id, ')
          ..write('name: $name, ')
          ..write('dateOfBirth: $dateOfBirth, ')
          ..write('sex: $sex, ')
          ..write('guardianName: $guardianName, ')
          ..write('location: $location, ')
          ..write('createdAt: $createdAt, ')
          ..write('updatedAt: $updatedAt')
          ..write(')'))
        .toString();
  }
}

class $VisitsTable extends Visits with TableInfo<$VisitsTable, Visit> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $VisitsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _childIdMeta =
      const VerificationMeta('childId');
  @override
  late final GeneratedColumn<int> childId = GeneratedColumn<int>(
      'child_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('REFERENCES children (id)'));
  static const VerificationMeta _visitDateMeta =
      const VerificationMeta('visitDate');
  @override
  late final GeneratedColumn<DateTime> visitDate = GeneratedColumn<DateTime>(
      'visit_date', aliasedName, false,
      type: DriftSqlType.dateTime,
      requiredDuringInsert: false,
      defaultValue: currentDateAndTime);
  static const VerificationMeta _ageMonthsMeta =
      const VerificationMeta('ageMonths');
  @override
  late final GeneratedColumn<double> ageMonths = GeneratedColumn<double>(
      'age_months', aliasedName, false,
      type: DriftSqlType.double, requiredDuringInsert: true);
  static const VerificationMeta _imagePathMeta =
      const VerificationMeta('imagePath');
  @override
  late final GeneratedColumn<String> imagePath = GeneratedColumn<String>(
      'image_path', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _sideImagePathMeta =
      const VerificationMeta('sideImagePath');
  @override
  late final GeneratedColumn<String> sideImagePath = GeneratedColumn<String>(
      'side_image_path', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _backImagePathMeta =
      const VerificationMeta('backImagePath');
  @override
  late final GeneratedColumn<String> backImagePath = GeneratedColumn<String>(
      'back_image_path', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _notesMeta = const VerificationMeta('notes');
  @override
  late final GeneratedColumn<String> notes = GeneratedColumn<String>(
      'notes', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        childId,
        visitDate,
        ageMonths,
        imagePath,
        sideImagePath,
        backImagePath,
        notes
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'visits';
  @override
  VerificationContext validateIntegrity(Insertable<Visit> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('child_id')) {
      context.handle(_childIdMeta,
          childId.isAcceptableOrUnknown(data['child_id']!, _childIdMeta));
    } else if (isInserting) {
      context.missing(_childIdMeta);
    }
    if (data.containsKey('visit_date')) {
      context.handle(_visitDateMeta,
          visitDate.isAcceptableOrUnknown(data['visit_date']!, _visitDateMeta));
    }
    if (data.containsKey('age_months')) {
      context.handle(_ageMonthsMeta,
          ageMonths.isAcceptableOrUnknown(data['age_months']!, _ageMonthsMeta));
    } else if (isInserting) {
      context.missing(_ageMonthsMeta);
    }
    if (data.containsKey('image_path')) {
      context.handle(_imagePathMeta,
          imagePath.isAcceptableOrUnknown(data['image_path']!, _imagePathMeta));
    } else if (isInserting) {
      context.missing(_imagePathMeta);
    }
    if (data.containsKey('side_image_path')) {
      context.handle(
          _sideImagePathMeta,
          sideImagePath.isAcceptableOrUnknown(
              data['side_image_path']!, _sideImagePathMeta));
    }
    if (data.containsKey('back_image_path')) {
      context.handle(
          _backImagePathMeta,
          backImagePath.isAcceptableOrUnknown(
              data['back_image_path']!, _backImagePathMeta));
    }
    if (data.containsKey('notes')) {
      context.handle(
          _notesMeta, notes.isAcceptableOrUnknown(data['notes']!, _notesMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Visit map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Visit(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      childId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}child_id'])!,
      visitDate: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}visit_date'])!,
      ageMonths: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}age_months'])!,
      imagePath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}image_path'])!,
      sideImagePath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}side_image_path']),
      backImagePath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}back_image_path']),
      notes: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}notes']),
    );
  }

  @override
  $VisitsTable createAlias(String alias) {
    return $VisitsTable(attachedDatabase, alias);
  }
}

class Visit extends DataClass implements Insertable<Visit> {
  final int id;
  final int childId;
  final DateTime visitDate;
  final double ageMonths;
  final String imagePath;
  final String? sideImagePath;
  final String? backImagePath;
  final String? notes;
  const Visit(
      {required this.id,
      required this.childId,
      required this.visitDate,
      required this.ageMonths,
      required this.imagePath,
      this.sideImagePath,
      this.backImagePath,
      this.notes});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['child_id'] = Variable<int>(childId);
    map['visit_date'] = Variable<DateTime>(visitDate);
    map['age_months'] = Variable<double>(ageMonths);
    map['image_path'] = Variable<String>(imagePath);
    if (!nullToAbsent || sideImagePath != null) {
      map['side_image_path'] = Variable<String>(sideImagePath);
    }
    if (!nullToAbsent || backImagePath != null) {
      map['back_image_path'] = Variable<String>(backImagePath);
    }
    if (!nullToAbsent || notes != null) {
      map['notes'] = Variable<String>(notes);
    }
    return map;
  }

  VisitsCompanion toCompanion(bool nullToAbsent) {
    return VisitsCompanion(
      id: Value(id),
      childId: Value(childId),
      visitDate: Value(visitDate),
      ageMonths: Value(ageMonths),
      imagePath: Value(imagePath),
      sideImagePath: sideImagePath == null && nullToAbsent
          ? const Value.absent()
          : Value(sideImagePath),
      backImagePath: backImagePath == null && nullToAbsent
          ? const Value.absent()
          : Value(backImagePath),
      notes:
          notes == null && nullToAbsent ? const Value.absent() : Value(notes),
    );
  }

  factory Visit.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Visit(
      id: serializer.fromJson<int>(json['id']),
      childId: serializer.fromJson<int>(json['childId']),
      visitDate: serializer.fromJson<DateTime>(json['visitDate']),
      ageMonths: serializer.fromJson<double>(json['ageMonths']),
      imagePath: serializer.fromJson<String>(json['imagePath']),
      sideImagePath: serializer.fromJson<String?>(json['sideImagePath']),
      backImagePath: serializer.fromJson<String?>(json['backImagePath']),
      notes: serializer.fromJson<String?>(json['notes']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'childId': serializer.toJson<int>(childId),
      'visitDate': serializer.toJson<DateTime>(visitDate),
      'ageMonths': serializer.toJson<double>(ageMonths),
      'imagePath': serializer.toJson<String>(imagePath),
      'sideImagePath': serializer.toJson<String?>(sideImagePath),
      'backImagePath': serializer.toJson<String?>(backImagePath),
      'notes': serializer.toJson<String?>(notes),
    };
  }

  Visit copyWith(
          {int? id,
          int? childId,
          DateTime? visitDate,
          double? ageMonths,
          String? imagePath,
          Value<String?> sideImagePath = const Value.absent(),
          Value<String?> backImagePath = const Value.absent(),
          Value<String?> notes = const Value.absent()}) =>
      Visit(
        id: id ?? this.id,
        childId: childId ?? this.childId,
        visitDate: visitDate ?? this.visitDate,
        ageMonths: ageMonths ?? this.ageMonths,
        imagePath: imagePath ?? this.imagePath,
        sideImagePath:
            sideImagePath.present ? sideImagePath.value : this.sideImagePath,
        backImagePath:
            backImagePath.present ? backImagePath.value : this.backImagePath,
        notes: notes.present ? notes.value : this.notes,
      );
  Visit copyWithCompanion(VisitsCompanion data) {
    return Visit(
      id: data.id.present ? data.id.value : this.id,
      childId: data.childId.present ? data.childId.value : this.childId,
      visitDate: data.visitDate.present ? data.visitDate.value : this.visitDate,
      ageMonths: data.ageMonths.present ? data.ageMonths.value : this.ageMonths,
      imagePath: data.imagePath.present ? data.imagePath.value : this.imagePath,
      sideImagePath: data.sideImagePath.present
          ? data.sideImagePath.value
          : this.sideImagePath,
      backImagePath: data.backImagePath.present
          ? data.backImagePath.value
          : this.backImagePath,
      notes: data.notes.present ? data.notes.value : this.notes,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Visit(')
          ..write('id: $id, ')
          ..write('childId: $childId, ')
          ..write('visitDate: $visitDate, ')
          ..write('ageMonths: $ageMonths, ')
          ..write('imagePath: $imagePath, ')
          ..write('sideImagePath: $sideImagePath, ')
          ..write('backImagePath: $backImagePath, ')
          ..write('notes: $notes')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, childId, visitDate, ageMonths, imagePath,
      sideImagePath, backImagePath, notes);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Visit &&
          other.id == this.id &&
          other.childId == this.childId &&
          other.visitDate == this.visitDate &&
          other.ageMonths == this.ageMonths &&
          other.imagePath == this.imagePath &&
          other.sideImagePath == this.sideImagePath &&
          other.backImagePath == this.backImagePath &&
          other.notes == this.notes);
}

class VisitsCompanion extends UpdateCompanion<Visit> {
  final Value<int> id;
  final Value<int> childId;
  final Value<DateTime> visitDate;
  final Value<double> ageMonths;
  final Value<String> imagePath;
  final Value<String?> sideImagePath;
  final Value<String?> backImagePath;
  final Value<String?> notes;
  const VisitsCompanion({
    this.id = const Value.absent(),
    this.childId = const Value.absent(),
    this.visitDate = const Value.absent(),
    this.ageMonths = const Value.absent(),
    this.imagePath = const Value.absent(),
    this.sideImagePath = const Value.absent(),
    this.backImagePath = const Value.absent(),
    this.notes = const Value.absent(),
  });
  VisitsCompanion.insert({
    this.id = const Value.absent(),
    required int childId,
    this.visitDate = const Value.absent(),
    required double ageMonths,
    required String imagePath,
    this.sideImagePath = const Value.absent(),
    this.backImagePath = const Value.absent(),
    this.notes = const Value.absent(),
  })  : childId = Value(childId),
        ageMonths = Value(ageMonths),
        imagePath = Value(imagePath);
  static Insertable<Visit> custom({
    Expression<int>? id,
    Expression<int>? childId,
    Expression<DateTime>? visitDate,
    Expression<double>? ageMonths,
    Expression<String>? imagePath,
    Expression<String>? sideImagePath,
    Expression<String>? backImagePath,
    Expression<String>? notes,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (childId != null) 'child_id': childId,
      if (visitDate != null) 'visit_date': visitDate,
      if (ageMonths != null) 'age_months': ageMonths,
      if (imagePath != null) 'image_path': imagePath,
      if (sideImagePath != null) 'side_image_path': sideImagePath,
      if (backImagePath != null) 'back_image_path': backImagePath,
      if (notes != null) 'notes': notes,
    });
  }

  VisitsCompanion copyWith(
      {Value<int>? id,
      Value<int>? childId,
      Value<DateTime>? visitDate,
      Value<double>? ageMonths,
      Value<String>? imagePath,
      Value<String?>? sideImagePath,
      Value<String?>? backImagePath,
      Value<String?>? notes}) {
    return VisitsCompanion(
      id: id ?? this.id,
      childId: childId ?? this.childId,
      visitDate: visitDate ?? this.visitDate,
      ageMonths: ageMonths ?? this.ageMonths,
      imagePath: imagePath ?? this.imagePath,
      sideImagePath: sideImagePath ?? this.sideImagePath,
      backImagePath: backImagePath ?? this.backImagePath,
      notes: notes ?? this.notes,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (childId.present) {
      map['child_id'] = Variable<int>(childId.value);
    }
    if (visitDate.present) {
      map['visit_date'] = Variable<DateTime>(visitDate.value);
    }
    if (ageMonths.present) {
      map['age_months'] = Variable<double>(ageMonths.value);
    }
    if (imagePath.present) {
      map['image_path'] = Variable<String>(imagePath.value);
    }
    if (sideImagePath.present) {
      map['side_image_path'] = Variable<String>(sideImagePath.value);
    }
    if (backImagePath.present) {
      map['back_image_path'] = Variable<String>(backImagePath.value);
    }
    if (notes.present) {
      map['notes'] = Variable<String>(notes.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('VisitsCompanion(')
          ..write('id: $id, ')
          ..write('childId: $childId, ')
          ..write('visitDate: $visitDate, ')
          ..write('ageMonths: $ageMonths, ')
          ..write('imagePath: $imagePath, ')
          ..write('sideImagePath: $sideImagePath, ')
          ..write('backImagePath: $backImagePath, ')
          ..write('notes: $notes')
          ..write(')'))
        .toString();
  }
}

class $MeasurementsTable extends Measurements
    with TableInfo<$MeasurementsTable, Measurement> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $MeasurementsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _visitIdMeta =
      const VerificationMeta('visitId');
  @override
  late final GeneratedColumn<int> visitId = GeneratedColumn<int>(
      'visit_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('UNIQUE REFERENCES visits (id)'));
  static const VerificationMeta _predictedHeightCmMeta =
      const VerificationMeta('predictedHeightCm');
  @override
  late final GeneratedColumn<double> predictedHeightCm =
      GeneratedColumn<double>('predicted_height_cm', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _predictedWeightKgMeta =
      const VerificationMeta('predictedWeightKg');
  @override
  late final GeneratedColumn<double> predictedWeightKg =
      GeneratedColumn<double>('predicted_weight_kg', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _manualHeightCmMeta =
      const VerificationMeta('manualHeightCm');
  @override
  late final GeneratedColumn<double> manualHeightCm = GeneratedColumn<double>(
      'manual_height_cm', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _manualWeightKgMeta =
      const VerificationMeta('manualWeightKg');
  @override
  late final GeneratedColumn<double> manualWeightKg = GeneratedColumn<double>(
      'manual_weight_kg', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _hazZscoreMeta =
      const VerificationMeta('hazZscore');
  @override
  late final GeneratedColumn<double> hazZscore = GeneratedColumn<double>(
      'haz_zscore', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _whzZscoreMeta =
      const VerificationMeta('whzZscore');
  @override
  late final GeneratedColumn<double> whzZscore = GeneratedColumn<double>(
      'whz_zscore', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _hazStatusMeta =
      const VerificationMeta('hazStatus');
  @override
  late final GeneratedColumn<String> hazStatus = GeneratedColumn<String>(
      'haz_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _whzStatusMeta =
      const VerificationMeta('whzStatus');
  @override
  late final GeneratedColumn<String> whzStatus = GeneratedColumn<String>(
      'whz_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _confidenceScoreMeta =
      const VerificationMeta('confidenceScore');
  @override
  late final GeneratedColumn<double> confidenceScore = GeneratedColumn<double>(
      'confidence_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _bodyBuildMeta =
      const VerificationMeta('bodyBuild');
  @override
  late final GeneratedColumn<String> bodyBuild = GeneratedColumn<String>(
      'body_build', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _estimationMethodMeta =
      const VerificationMeta('estimationMethod');
  @override
  late final GeneratedColumn<String> estimationMethod = GeneratedColumn<String>(
      'estimation_method', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _sideViewUsedMeta =
      const VerificationMeta('sideViewUsed');
  @override
  late final GeneratedColumn<bool> sideViewUsed = GeneratedColumn<bool>(
      'side_view_used', aliasedName, false,
      type: DriftSqlType.bool,
      requiredDuringInsert: false,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'CHECK ("side_view_used" IN (0, 1))'),
      defaultValue: const Constant(false));
  static const VerificationMeta _chestDepthCmMeta =
      const VerificationMeta('chestDepthCm');
  @override
  late final GeneratedColumn<double> chestDepthCm = GeneratedColumn<double>(
      'chest_depth_cm', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _abdDepthCmMeta =
      const VerificationMeta('abdDepthCm');
  @override
  late final GeneratedColumn<double> abdDepthCm = GeneratedColumn<double>(
      'abd_depth_cm', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _mlEstimatedWeightKgMeta =
      const VerificationMeta('mlEstimatedWeightKg');
  @override
  late final GeneratedColumn<double> mlEstimatedWeightKg =
      GeneratedColumn<double>('ml_estimated_weight_kg', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _samProbabilityMeta =
      const VerificationMeta('samProbability');
  @override
  late final GeneratedColumn<double> samProbability = GeneratedColumn<double>(
      'sam_probability', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _mamProbabilityMeta =
      const VerificationMeta('mamProbability');
  @override
  late final GeneratedColumn<double> mamProbability = GeneratedColumn<double>(
      'mam_probability', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _normalProbabilityMeta =
      const VerificationMeta('normalProbability');
  @override
  late final GeneratedColumn<double> normalProbability =
      GeneratedColumn<double>('normal_probability', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _riskOverweightProbabilityMeta =
      const VerificationMeta('riskOverweightProbability');
  @override
  late final GeneratedColumn<double> riskOverweightProbability =
      GeneratedColumn<double>('risk_overweight_probability', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _overweightProbabilityMeta =
      const VerificationMeta('overweightProbability');
  @override
  late final GeneratedColumn<double> overweightProbability =
      GeneratedColumn<double>('overweight_probability', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _wastingStatusMeta =
      const VerificationMeta('wastingStatus');
  @override
  late final GeneratedColumn<String> wastingStatus = GeneratedColumn<String>(
      'wasting_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _muacCmMeta = const VerificationMeta('muacCm');
  @override
  late final GeneratedColumn<double> muacCm = GeneratedColumn<double>(
      'muac_cm', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _muacStatusMeta =
      const VerificationMeta('muacStatus');
  @override
  late final GeneratedColumn<String> muacStatus = GeneratedColumn<String>(
      'muac_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _muacMethodMeta =
      const VerificationMeta('muacMethod');
  @override
  late final GeneratedColumn<String> muacMethod = GeneratedColumn<String>(
      'muac_method', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        visitId,
        predictedHeightCm,
        predictedWeightKg,
        manualHeightCm,
        manualWeightKg,
        hazZscore,
        whzZscore,
        hazStatus,
        whzStatus,
        confidenceScore,
        bodyBuild,
        estimationMethod,
        sideViewUsed,
        chestDepthCm,
        abdDepthCm,
        mlEstimatedWeightKg,
        samProbability,
        mamProbability,
        normalProbability,
        riskOverweightProbability,
        overweightProbability,
        wastingStatus,
        muacCm,
        muacStatus,
        muacMethod
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'measurements';
  @override
  VerificationContext validateIntegrity(Insertable<Measurement> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('visit_id')) {
      context.handle(_visitIdMeta,
          visitId.isAcceptableOrUnknown(data['visit_id']!, _visitIdMeta));
    } else if (isInserting) {
      context.missing(_visitIdMeta);
    }
    if (data.containsKey('predicted_height_cm')) {
      context.handle(
          _predictedHeightCmMeta,
          predictedHeightCm.isAcceptableOrUnknown(
              data['predicted_height_cm']!, _predictedHeightCmMeta));
    }
    if (data.containsKey('predicted_weight_kg')) {
      context.handle(
          _predictedWeightKgMeta,
          predictedWeightKg.isAcceptableOrUnknown(
              data['predicted_weight_kg']!, _predictedWeightKgMeta));
    }
    if (data.containsKey('manual_height_cm')) {
      context.handle(
          _manualHeightCmMeta,
          manualHeightCm.isAcceptableOrUnknown(
              data['manual_height_cm']!, _manualHeightCmMeta));
    }
    if (data.containsKey('manual_weight_kg')) {
      context.handle(
          _manualWeightKgMeta,
          manualWeightKg.isAcceptableOrUnknown(
              data['manual_weight_kg']!, _manualWeightKgMeta));
    }
    if (data.containsKey('haz_zscore')) {
      context.handle(_hazZscoreMeta,
          hazZscore.isAcceptableOrUnknown(data['haz_zscore']!, _hazZscoreMeta));
    }
    if (data.containsKey('whz_zscore')) {
      context.handle(_whzZscoreMeta,
          whzZscore.isAcceptableOrUnknown(data['whz_zscore']!, _whzZscoreMeta));
    }
    if (data.containsKey('haz_status')) {
      context.handle(_hazStatusMeta,
          hazStatus.isAcceptableOrUnknown(data['haz_status']!, _hazStatusMeta));
    }
    if (data.containsKey('whz_status')) {
      context.handle(_whzStatusMeta,
          whzStatus.isAcceptableOrUnknown(data['whz_status']!, _whzStatusMeta));
    }
    if (data.containsKey('confidence_score')) {
      context.handle(
          _confidenceScoreMeta,
          confidenceScore.isAcceptableOrUnknown(
              data['confidence_score']!, _confidenceScoreMeta));
    }
    if (data.containsKey('body_build')) {
      context.handle(_bodyBuildMeta,
          bodyBuild.isAcceptableOrUnknown(data['body_build']!, _bodyBuildMeta));
    }
    if (data.containsKey('estimation_method')) {
      context.handle(
          _estimationMethodMeta,
          estimationMethod.isAcceptableOrUnknown(
              data['estimation_method']!, _estimationMethodMeta));
    }
    if (data.containsKey('side_view_used')) {
      context.handle(
          _sideViewUsedMeta,
          sideViewUsed.isAcceptableOrUnknown(
              data['side_view_used']!, _sideViewUsedMeta));
    }
    if (data.containsKey('chest_depth_cm')) {
      context.handle(
          _chestDepthCmMeta,
          chestDepthCm.isAcceptableOrUnknown(
              data['chest_depth_cm']!, _chestDepthCmMeta));
    }
    if (data.containsKey('abd_depth_cm')) {
      context.handle(
          _abdDepthCmMeta,
          abdDepthCm.isAcceptableOrUnknown(
              data['abd_depth_cm']!, _abdDepthCmMeta));
    }
    if (data.containsKey('ml_estimated_weight_kg')) {
      context.handle(
          _mlEstimatedWeightKgMeta,
          mlEstimatedWeightKg.isAcceptableOrUnknown(
              data['ml_estimated_weight_kg']!, _mlEstimatedWeightKgMeta));
    }
    if (data.containsKey('sam_probability')) {
      context.handle(
          _samProbabilityMeta,
          samProbability.isAcceptableOrUnknown(
              data['sam_probability']!, _samProbabilityMeta));
    }
    if (data.containsKey('mam_probability')) {
      context.handle(
          _mamProbabilityMeta,
          mamProbability.isAcceptableOrUnknown(
              data['mam_probability']!, _mamProbabilityMeta));
    }
    if (data.containsKey('normal_probability')) {
      context.handle(
          _normalProbabilityMeta,
          normalProbability.isAcceptableOrUnknown(
              data['normal_probability']!, _normalProbabilityMeta));
    }
    if (data.containsKey('risk_overweight_probability')) {
      context.handle(
          _riskOverweightProbabilityMeta,
          riskOverweightProbability.isAcceptableOrUnknown(
              data['risk_overweight_probability']!,
              _riskOverweightProbabilityMeta));
    }
    if (data.containsKey('overweight_probability')) {
      context.handle(
          _overweightProbabilityMeta,
          overweightProbability.isAcceptableOrUnknown(
              data['overweight_probability']!, _overweightProbabilityMeta));
    }
    if (data.containsKey('wasting_status')) {
      context.handle(
          _wastingStatusMeta,
          wastingStatus.isAcceptableOrUnknown(
              data['wasting_status']!, _wastingStatusMeta));
    }
    if (data.containsKey('muac_cm')) {
      context.handle(_muacCmMeta,
          muacCm.isAcceptableOrUnknown(data['muac_cm']!, _muacCmMeta));
    }
    if (data.containsKey('muac_status')) {
      context.handle(
          _muacStatusMeta,
          muacStatus.isAcceptableOrUnknown(
              data['muac_status']!, _muacStatusMeta));
    }
    if (data.containsKey('muac_method')) {
      context.handle(
          _muacMethodMeta,
          muacMethod.isAcceptableOrUnknown(
              data['muac_method']!, _muacMethodMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Measurement map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Measurement(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      visitId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}visit_id'])!,
      predictedHeightCm: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}predicted_height_cm']),
      predictedWeightKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}predicted_weight_kg']),
      manualHeightCm: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}manual_height_cm']),
      manualWeightKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}manual_weight_kg']),
      hazZscore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}haz_zscore']),
      whzZscore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}whz_zscore']),
      hazStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}haz_status']),
      whzStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}whz_status']),
      confidenceScore: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}confidence_score']),
      bodyBuild: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}body_build']),
      estimationMethod: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}estimation_method']),
      sideViewUsed: attachedDatabase.typeMapping
          .read(DriftSqlType.bool, data['${effectivePrefix}side_view_used'])!,
      chestDepthCm: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}chest_depth_cm']),
      abdDepthCm: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}abd_depth_cm']),
      mlEstimatedWeightKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}ml_estimated_weight_kg']),
      samProbability: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}sam_probability']),
      mamProbability: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}mam_probability']),
      normalProbability: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}normal_probability']),
      riskOverweightProbability: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}risk_overweight_probability']),
      overweightProbability: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}overweight_probability']),
      wastingStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}wasting_status']),
      muacCm: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}muac_cm']),
      muacStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}muac_status']),
      muacMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}muac_method']),
    );
  }

  @override
  $MeasurementsTable createAlias(String alias) {
    return $MeasurementsTable(attachedDatabase, alias);
  }
}

class Measurement extends DataClass implements Insertable<Measurement> {
  final int id;
  final int visitId;
  final double? predictedHeightCm;
  final double? predictedWeightKg;
  final double? manualHeightCm;
  final double? manualWeightKg;
  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? confidenceScore;
  final String? bodyBuild;
  final String? estimationMethod;
  final bool sideViewUsed;
  final double? chestDepthCm;
  final double? abdDepthCm;
  final double? mlEstimatedWeightKg;
  final double? samProbability;
  final double? mamProbability;
  final double? normalProbability;
  final double? riskOverweightProbability;
  final double? overweightProbability;
  final String? wastingStatus;
  final double? muacCm;
  final String? muacStatus;
  final String? muacMethod;
  const Measurement(
      {required this.id,
      required this.visitId,
      this.predictedHeightCm,
      this.predictedWeightKg,
      this.manualHeightCm,
      this.manualWeightKg,
      this.hazZscore,
      this.whzZscore,
      this.hazStatus,
      this.whzStatus,
      this.confidenceScore,
      this.bodyBuild,
      this.estimationMethod,
      required this.sideViewUsed,
      this.chestDepthCm,
      this.abdDepthCm,
      this.mlEstimatedWeightKg,
      this.samProbability,
      this.mamProbability,
      this.normalProbability,
      this.riskOverweightProbability,
      this.overweightProbability,
      this.wastingStatus,
      this.muacCm,
      this.muacStatus,
      this.muacMethod});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['visit_id'] = Variable<int>(visitId);
    if (!nullToAbsent || predictedHeightCm != null) {
      map['predicted_height_cm'] = Variable<double>(predictedHeightCm);
    }
    if (!nullToAbsent || predictedWeightKg != null) {
      map['predicted_weight_kg'] = Variable<double>(predictedWeightKg);
    }
    if (!nullToAbsent || manualHeightCm != null) {
      map['manual_height_cm'] = Variable<double>(manualHeightCm);
    }
    if (!nullToAbsent || manualWeightKg != null) {
      map['manual_weight_kg'] = Variable<double>(manualWeightKg);
    }
    if (!nullToAbsent || hazZscore != null) {
      map['haz_zscore'] = Variable<double>(hazZscore);
    }
    if (!nullToAbsent || whzZscore != null) {
      map['whz_zscore'] = Variable<double>(whzZscore);
    }
    if (!nullToAbsent || hazStatus != null) {
      map['haz_status'] = Variable<String>(hazStatus);
    }
    if (!nullToAbsent || whzStatus != null) {
      map['whz_status'] = Variable<String>(whzStatus);
    }
    if (!nullToAbsent || confidenceScore != null) {
      map['confidence_score'] = Variable<double>(confidenceScore);
    }
    if (!nullToAbsent || bodyBuild != null) {
      map['body_build'] = Variable<String>(bodyBuild);
    }
    if (!nullToAbsent || estimationMethod != null) {
      map['estimation_method'] = Variable<String>(estimationMethod);
    }
    map['side_view_used'] = Variable<bool>(sideViewUsed);
    if (!nullToAbsent || chestDepthCm != null) {
      map['chest_depth_cm'] = Variable<double>(chestDepthCm);
    }
    if (!nullToAbsent || abdDepthCm != null) {
      map['abd_depth_cm'] = Variable<double>(abdDepthCm);
    }
    if (!nullToAbsent || mlEstimatedWeightKg != null) {
      map['ml_estimated_weight_kg'] = Variable<double>(mlEstimatedWeightKg);
    }
    if (!nullToAbsent || samProbability != null) {
      map['sam_probability'] = Variable<double>(samProbability);
    }
    if (!nullToAbsent || mamProbability != null) {
      map['mam_probability'] = Variable<double>(mamProbability);
    }
    if (!nullToAbsent || normalProbability != null) {
      map['normal_probability'] = Variable<double>(normalProbability);
    }
    if (!nullToAbsent || riskOverweightProbability != null) {
      map['risk_overweight_probability'] =
          Variable<double>(riskOverweightProbability);
    }
    if (!nullToAbsent || overweightProbability != null) {
      map['overweight_probability'] = Variable<double>(overweightProbability);
    }
    if (!nullToAbsent || wastingStatus != null) {
      map['wasting_status'] = Variable<String>(wastingStatus);
    }
    if (!nullToAbsent || muacCm != null) {
      map['muac_cm'] = Variable<double>(muacCm);
    }
    if (!nullToAbsent || muacStatus != null) {
      map['muac_status'] = Variable<String>(muacStatus);
    }
    if (!nullToAbsent || muacMethod != null) {
      map['muac_method'] = Variable<String>(muacMethod);
    }
    return map;
  }

  MeasurementsCompanion toCompanion(bool nullToAbsent) {
    return MeasurementsCompanion(
      id: Value(id),
      visitId: Value(visitId),
      predictedHeightCm: predictedHeightCm == null && nullToAbsent
          ? const Value.absent()
          : Value(predictedHeightCm),
      predictedWeightKg: predictedWeightKg == null && nullToAbsent
          ? const Value.absent()
          : Value(predictedWeightKg),
      manualHeightCm: manualHeightCm == null && nullToAbsent
          ? const Value.absent()
          : Value(manualHeightCm),
      manualWeightKg: manualWeightKg == null && nullToAbsent
          ? const Value.absent()
          : Value(manualWeightKg),
      hazZscore: hazZscore == null && nullToAbsent
          ? const Value.absent()
          : Value(hazZscore),
      whzZscore: whzZscore == null && nullToAbsent
          ? const Value.absent()
          : Value(whzZscore),
      hazStatus: hazStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(hazStatus),
      whzStatus: whzStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(whzStatus),
      confidenceScore: confidenceScore == null && nullToAbsent
          ? const Value.absent()
          : Value(confidenceScore),
      bodyBuild: bodyBuild == null && nullToAbsent
          ? const Value.absent()
          : Value(bodyBuild),
      estimationMethod: estimationMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(estimationMethod),
      sideViewUsed: Value(sideViewUsed),
      chestDepthCm: chestDepthCm == null && nullToAbsent
          ? const Value.absent()
          : Value(chestDepthCm),
      abdDepthCm: abdDepthCm == null && nullToAbsent
          ? const Value.absent()
          : Value(abdDepthCm),
      mlEstimatedWeightKg: mlEstimatedWeightKg == null && nullToAbsent
          ? const Value.absent()
          : Value(mlEstimatedWeightKg),
      samProbability: samProbability == null && nullToAbsent
          ? const Value.absent()
          : Value(samProbability),
      mamProbability: mamProbability == null && nullToAbsent
          ? const Value.absent()
          : Value(mamProbability),
      normalProbability: normalProbability == null && nullToAbsent
          ? const Value.absent()
          : Value(normalProbability),
      riskOverweightProbability:
          riskOverweightProbability == null && nullToAbsent
              ? const Value.absent()
              : Value(riskOverweightProbability),
      overweightProbability: overweightProbability == null && nullToAbsent
          ? const Value.absent()
          : Value(overweightProbability),
      wastingStatus: wastingStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(wastingStatus),
      muacCm:
          muacCm == null && nullToAbsent ? const Value.absent() : Value(muacCm),
      muacStatus: muacStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(muacStatus),
      muacMethod: muacMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(muacMethod),
    );
  }

  factory Measurement.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Measurement(
      id: serializer.fromJson<int>(json['id']),
      visitId: serializer.fromJson<int>(json['visitId']),
      predictedHeightCm:
          serializer.fromJson<double?>(json['predictedHeightCm']),
      predictedWeightKg:
          serializer.fromJson<double?>(json['predictedWeightKg']),
      manualHeightCm: serializer.fromJson<double?>(json['manualHeightCm']),
      manualWeightKg: serializer.fromJson<double?>(json['manualWeightKg']),
      hazZscore: serializer.fromJson<double?>(json['hazZscore']),
      whzZscore: serializer.fromJson<double?>(json['whzZscore']),
      hazStatus: serializer.fromJson<String?>(json['hazStatus']),
      whzStatus: serializer.fromJson<String?>(json['whzStatus']),
      confidenceScore: serializer.fromJson<double?>(json['confidenceScore']),
      bodyBuild: serializer.fromJson<String?>(json['bodyBuild']),
      estimationMethod: serializer.fromJson<String?>(json['estimationMethod']),
      sideViewUsed: serializer.fromJson<bool>(json['sideViewUsed']),
      chestDepthCm: serializer.fromJson<double?>(json['chestDepthCm']),
      abdDepthCm: serializer.fromJson<double?>(json['abdDepthCm']),
      mlEstimatedWeightKg:
          serializer.fromJson<double?>(json['mlEstimatedWeightKg']),
      samProbability: serializer.fromJson<double?>(json['samProbability']),
      mamProbability: serializer.fromJson<double?>(json['mamProbability']),
      normalProbability:
          serializer.fromJson<double?>(json['normalProbability']),
      riskOverweightProbability:
          serializer.fromJson<double?>(json['riskOverweightProbability']),
      overweightProbability:
          serializer.fromJson<double?>(json['overweightProbability']),
      wastingStatus: serializer.fromJson<String?>(json['wastingStatus']),
      muacCm: serializer.fromJson<double?>(json['muacCm']),
      muacStatus: serializer.fromJson<String?>(json['muacStatus']),
      muacMethod: serializer.fromJson<String?>(json['muacMethod']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'visitId': serializer.toJson<int>(visitId),
      'predictedHeightCm': serializer.toJson<double?>(predictedHeightCm),
      'predictedWeightKg': serializer.toJson<double?>(predictedWeightKg),
      'manualHeightCm': serializer.toJson<double?>(manualHeightCm),
      'manualWeightKg': serializer.toJson<double?>(manualWeightKg),
      'hazZscore': serializer.toJson<double?>(hazZscore),
      'whzZscore': serializer.toJson<double?>(whzZscore),
      'hazStatus': serializer.toJson<String?>(hazStatus),
      'whzStatus': serializer.toJson<String?>(whzStatus),
      'confidenceScore': serializer.toJson<double?>(confidenceScore),
      'bodyBuild': serializer.toJson<String?>(bodyBuild),
      'estimationMethod': serializer.toJson<String?>(estimationMethod),
      'sideViewUsed': serializer.toJson<bool>(sideViewUsed),
      'chestDepthCm': serializer.toJson<double?>(chestDepthCm),
      'abdDepthCm': serializer.toJson<double?>(abdDepthCm),
      'mlEstimatedWeightKg': serializer.toJson<double?>(mlEstimatedWeightKg),
      'samProbability': serializer.toJson<double?>(samProbability),
      'mamProbability': serializer.toJson<double?>(mamProbability),
      'normalProbability': serializer.toJson<double?>(normalProbability),
      'riskOverweightProbability':
          serializer.toJson<double?>(riskOverweightProbability),
      'overweightProbability':
          serializer.toJson<double?>(overweightProbability),
      'wastingStatus': serializer.toJson<String?>(wastingStatus),
      'muacCm': serializer.toJson<double?>(muacCm),
      'muacStatus': serializer.toJson<String?>(muacStatus),
      'muacMethod': serializer.toJson<String?>(muacMethod),
    };
  }

  Measurement copyWith(
          {int? id,
          int? visitId,
          Value<double?> predictedHeightCm = const Value.absent(),
          Value<double?> predictedWeightKg = const Value.absent(),
          Value<double?> manualHeightCm = const Value.absent(),
          Value<double?> manualWeightKg = const Value.absent(),
          Value<double?> hazZscore = const Value.absent(),
          Value<double?> whzZscore = const Value.absent(),
          Value<String?> hazStatus = const Value.absent(),
          Value<String?> whzStatus = const Value.absent(),
          Value<double?> confidenceScore = const Value.absent(),
          Value<String?> bodyBuild = const Value.absent(),
          Value<String?> estimationMethod = const Value.absent(),
          bool? sideViewUsed,
          Value<double?> chestDepthCm = const Value.absent(),
          Value<double?> abdDepthCm = const Value.absent(),
          Value<double?> mlEstimatedWeightKg = const Value.absent(),
          Value<double?> samProbability = const Value.absent(),
          Value<double?> mamProbability = const Value.absent(),
          Value<double?> normalProbability = const Value.absent(),
          Value<double?> riskOverweightProbability = const Value.absent(),
          Value<double?> overweightProbability = const Value.absent(),
          Value<String?> wastingStatus = const Value.absent(),
          Value<double?> muacCm = const Value.absent(),
          Value<String?> muacStatus = const Value.absent(),
          Value<String?> muacMethod = const Value.absent()}) =>
      Measurement(
        id: id ?? this.id,
        visitId: visitId ?? this.visitId,
        predictedHeightCm: predictedHeightCm.present
            ? predictedHeightCm.value
            : this.predictedHeightCm,
        predictedWeightKg: predictedWeightKg.present
            ? predictedWeightKg.value
            : this.predictedWeightKg,
        manualHeightCm:
            manualHeightCm.present ? manualHeightCm.value : this.manualHeightCm,
        manualWeightKg:
            manualWeightKg.present ? manualWeightKg.value : this.manualWeightKg,
        hazZscore: hazZscore.present ? hazZscore.value : this.hazZscore,
        whzZscore: whzZscore.present ? whzZscore.value : this.whzZscore,
        hazStatus: hazStatus.present ? hazStatus.value : this.hazStatus,
        whzStatus: whzStatus.present ? whzStatus.value : this.whzStatus,
        confidenceScore: confidenceScore.present
            ? confidenceScore.value
            : this.confidenceScore,
        bodyBuild: bodyBuild.present ? bodyBuild.value : this.bodyBuild,
        estimationMethod: estimationMethod.present
            ? estimationMethod.value
            : this.estimationMethod,
        sideViewUsed: sideViewUsed ?? this.sideViewUsed,
        chestDepthCm:
            chestDepthCm.present ? chestDepthCm.value : this.chestDepthCm,
        abdDepthCm: abdDepthCm.present ? abdDepthCm.value : this.abdDepthCm,
        mlEstimatedWeightKg: mlEstimatedWeightKg.present
            ? mlEstimatedWeightKg.value
            : this.mlEstimatedWeightKg,
        samProbability:
            samProbability.present ? samProbability.value : this.samProbability,
        mamProbability:
            mamProbability.present ? mamProbability.value : this.mamProbability,
        normalProbability: normalProbability.present
            ? normalProbability.value
            : this.normalProbability,
        riskOverweightProbability: riskOverweightProbability.present
            ? riskOverweightProbability.value
            : this.riskOverweightProbability,
        overweightProbability: overweightProbability.present
            ? overweightProbability.value
            : this.overweightProbability,
        wastingStatus:
            wastingStatus.present ? wastingStatus.value : this.wastingStatus,
        muacCm: muacCm.present ? muacCm.value : this.muacCm,
        muacStatus: muacStatus.present ? muacStatus.value : this.muacStatus,
        muacMethod: muacMethod.present ? muacMethod.value : this.muacMethod,
      );
  Measurement copyWithCompanion(MeasurementsCompanion data) {
    return Measurement(
      id: data.id.present ? data.id.value : this.id,
      visitId: data.visitId.present ? data.visitId.value : this.visitId,
      predictedHeightCm: data.predictedHeightCm.present
          ? data.predictedHeightCm.value
          : this.predictedHeightCm,
      predictedWeightKg: data.predictedWeightKg.present
          ? data.predictedWeightKg.value
          : this.predictedWeightKg,
      manualHeightCm: data.manualHeightCm.present
          ? data.manualHeightCm.value
          : this.manualHeightCm,
      manualWeightKg: data.manualWeightKg.present
          ? data.manualWeightKg.value
          : this.manualWeightKg,
      hazZscore: data.hazZscore.present ? data.hazZscore.value : this.hazZscore,
      whzZscore: data.whzZscore.present ? data.whzZscore.value : this.whzZscore,
      hazStatus: data.hazStatus.present ? data.hazStatus.value : this.hazStatus,
      whzStatus: data.whzStatus.present ? data.whzStatus.value : this.whzStatus,
      confidenceScore: data.confidenceScore.present
          ? data.confidenceScore.value
          : this.confidenceScore,
      bodyBuild: data.bodyBuild.present ? data.bodyBuild.value : this.bodyBuild,
      estimationMethod: data.estimationMethod.present
          ? data.estimationMethod.value
          : this.estimationMethod,
      sideViewUsed: data.sideViewUsed.present
          ? data.sideViewUsed.value
          : this.sideViewUsed,
      chestDepthCm: data.chestDepthCm.present
          ? data.chestDepthCm.value
          : this.chestDepthCm,
      abdDepthCm:
          data.abdDepthCm.present ? data.abdDepthCm.value : this.abdDepthCm,
      mlEstimatedWeightKg: data.mlEstimatedWeightKg.present
          ? data.mlEstimatedWeightKg.value
          : this.mlEstimatedWeightKg,
      samProbability: data.samProbability.present
          ? data.samProbability.value
          : this.samProbability,
      mamProbability: data.mamProbability.present
          ? data.mamProbability.value
          : this.mamProbability,
      normalProbability: data.normalProbability.present
          ? data.normalProbability.value
          : this.normalProbability,
      riskOverweightProbability: data.riskOverweightProbability.present
          ? data.riskOverweightProbability.value
          : this.riskOverweightProbability,
      overweightProbability: data.overweightProbability.present
          ? data.overweightProbability.value
          : this.overweightProbability,
      wastingStatus: data.wastingStatus.present
          ? data.wastingStatus.value
          : this.wastingStatus,
      muacCm: data.muacCm.present ? data.muacCm.value : this.muacCm,
      muacStatus:
          data.muacStatus.present ? data.muacStatus.value : this.muacStatus,
      muacMethod:
          data.muacMethod.present ? data.muacMethod.value : this.muacMethod,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Measurement(')
          ..write('id: $id, ')
          ..write('visitId: $visitId, ')
          ..write('predictedHeightCm: $predictedHeightCm, ')
          ..write('predictedWeightKg: $predictedWeightKg, ')
          ..write('manualHeightCm: $manualHeightCm, ')
          ..write('manualWeightKg: $manualWeightKg, ')
          ..write('hazZscore: $hazZscore, ')
          ..write('whzZscore: $whzZscore, ')
          ..write('hazStatus: $hazStatus, ')
          ..write('whzStatus: $whzStatus, ')
          ..write('confidenceScore: $confidenceScore, ')
          ..write('bodyBuild: $bodyBuild, ')
          ..write('estimationMethod: $estimationMethod, ')
          ..write('sideViewUsed: $sideViewUsed, ')
          ..write('chestDepthCm: $chestDepthCm, ')
          ..write('abdDepthCm: $abdDepthCm, ')
          ..write('mlEstimatedWeightKg: $mlEstimatedWeightKg, ')
          ..write('samProbability: $samProbability, ')
          ..write('mamProbability: $mamProbability, ')
          ..write('normalProbability: $normalProbability, ')
          ..write('riskOverweightProbability: $riskOverweightProbability, ')
          ..write('overweightProbability: $overweightProbability, ')
          ..write('wastingStatus: $wastingStatus, ')
          ..write('muacCm: $muacCm, ')
          ..write('muacStatus: $muacStatus, ')
          ..write('muacMethod: $muacMethod')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hashAll([
        id,
        visitId,
        predictedHeightCm,
        predictedWeightKg,
        manualHeightCm,
        manualWeightKg,
        hazZscore,
        whzZscore,
        hazStatus,
        whzStatus,
        confidenceScore,
        bodyBuild,
        estimationMethod,
        sideViewUsed,
        chestDepthCm,
        abdDepthCm,
        mlEstimatedWeightKg,
        samProbability,
        mamProbability,
        normalProbability,
        riskOverweightProbability,
        overweightProbability,
        wastingStatus,
        muacCm,
        muacStatus,
        muacMethod
      ]);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Measurement &&
          other.id == this.id &&
          other.visitId == this.visitId &&
          other.predictedHeightCm == this.predictedHeightCm &&
          other.predictedWeightKg == this.predictedWeightKg &&
          other.manualHeightCm == this.manualHeightCm &&
          other.manualWeightKg == this.manualWeightKg &&
          other.hazZscore == this.hazZscore &&
          other.whzZscore == this.whzZscore &&
          other.hazStatus == this.hazStatus &&
          other.whzStatus == this.whzStatus &&
          other.confidenceScore == this.confidenceScore &&
          other.bodyBuild == this.bodyBuild &&
          other.estimationMethod == this.estimationMethod &&
          other.sideViewUsed == this.sideViewUsed &&
          other.chestDepthCm == this.chestDepthCm &&
          other.abdDepthCm == this.abdDepthCm &&
          other.mlEstimatedWeightKg == this.mlEstimatedWeightKg &&
          other.samProbability == this.samProbability &&
          other.mamProbability == this.mamProbability &&
          other.normalProbability == this.normalProbability &&
          other.riskOverweightProbability == this.riskOverweightProbability &&
          other.overweightProbability == this.overweightProbability &&
          other.wastingStatus == this.wastingStatus &&
          other.muacCm == this.muacCm &&
          other.muacStatus == this.muacStatus &&
          other.muacMethod == this.muacMethod);
}

class MeasurementsCompanion extends UpdateCompanion<Measurement> {
  final Value<int> id;
  final Value<int> visitId;
  final Value<double?> predictedHeightCm;
  final Value<double?> predictedWeightKg;
  final Value<double?> manualHeightCm;
  final Value<double?> manualWeightKg;
  final Value<double?> hazZscore;
  final Value<double?> whzZscore;
  final Value<String?> hazStatus;
  final Value<String?> whzStatus;
  final Value<double?> confidenceScore;
  final Value<String?> bodyBuild;
  final Value<String?> estimationMethod;
  final Value<bool> sideViewUsed;
  final Value<double?> chestDepthCm;
  final Value<double?> abdDepthCm;
  final Value<double?> mlEstimatedWeightKg;
  final Value<double?> samProbability;
  final Value<double?> mamProbability;
  final Value<double?> normalProbability;
  final Value<double?> riskOverweightProbability;
  final Value<double?> overweightProbability;
  final Value<String?> wastingStatus;
  final Value<double?> muacCm;
  final Value<String?> muacStatus;
  final Value<String?> muacMethod;
  const MeasurementsCompanion({
    this.id = const Value.absent(),
    this.visitId = const Value.absent(),
    this.predictedHeightCm = const Value.absent(),
    this.predictedWeightKg = const Value.absent(),
    this.manualHeightCm = const Value.absent(),
    this.manualWeightKg = const Value.absent(),
    this.hazZscore = const Value.absent(),
    this.whzZscore = const Value.absent(),
    this.hazStatus = const Value.absent(),
    this.whzStatus = const Value.absent(),
    this.confidenceScore = const Value.absent(),
    this.bodyBuild = const Value.absent(),
    this.estimationMethod = const Value.absent(),
    this.sideViewUsed = const Value.absent(),
    this.chestDepthCm = const Value.absent(),
    this.abdDepthCm = const Value.absent(),
    this.mlEstimatedWeightKg = const Value.absent(),
    this.samProbability = const Value.absent(),
    this.mamProbability = const Value.absent(),
    this.normalProbability = const Value.absent(),
    this.riskOverweightProbability = const Value.absent(),
    this.overweightProbability = const Value.absent(),
    this.wastingStatus = const Value.absent(),
    this.muacCm = const Value.absent(),
    this.muacStatus = const Value.absent(),
    this.muacMethod = const Value.absent(),
  });
  MeasurementsCompanion.insert({
    this.id = const Value.absent(),
    required int visitId,
    this.predictedHeightCm = const Value.absent(),
    this.predictedWeightKg = const Value.absent(),
    this.manualHeightCm = const Value.absent(),
    this.manualWeightKg = const Value.absent(),
    this.hazZscore = const Value.absent(),
    this.whzZscore = const Value.absent(),
    this.hazStatus = const Value.absent(),
    this.whzStatus = const Value.absent(),
    this.confidenceScore = const Value.absent(),
    this.bodyBuild = const Value.absent(),
    this.estimationMethod = const Value.absent(),
    this.sideViewUsed = const Value.absent(),
    this.chestDepthCm = const Value.absent(),
    this.abdDepthCm = const Value.absent(),
    this.mlEstimatedWeightKg = const Value.absent(),
    this.samProbability = const Value.absent(),
    this.mamProbability = const Value.absent(),
    this.normalProbability = const Value.absent(),
    this.riskOverweightProbability = const Value.absent(),
    this.overweightProbability = const Value.absent(),
    this.wastingStatus = const Value.absent(),
    this.muacCm = const Value.absent(),
    this.muacStatus = const Value.absent(),
    this.muacMethod = const Value.absent(),
  }) : visitId = Value(visitId);
  static Insertable<Measurement> custom({
    Expression<int>? id,
    Expression<int>? visitId,
    Expression<double>? predictedHeightCm,
    Expression<double>? predictedWeightKg,
    Expression<double>? manualHeightCm,
    Expression<double>? manualWeightKg,
    Expression<double>? hazZscore,
    Expression<double>? whzZscore,
    Expression<String>? hazStatus,
    Expression<String>? whzStatus,
    Expression<double>? confidenceScore,
    Expression<String>? bodyBuild,
    Expression<String>? estimationMethod,
    Expression<bool>? sideViewUsed,
    Expression<double>? chestDepthCm,
    Expression<double>? abdDepthCm,
    Expression<double>? mlEstimatedWeightKg,
    Expression<double>? samProbability,
    Expression<double>? mamProbability,
    Expression<double>? normalProbability,
    Expression<double>? riskOverweightProbability,
    Expression<double>? overweightProbability,
    Expression<String>? wastingStatus,
    Expression<double>? muacCm,
    Expression<String>? muacStatus,
    Expression<String>? muacMethod,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (visitId != null) 'visit_id': visitId,
      if (predictedHeightCm != null) 'predicted_height_cm': predictedHeightCm,
      if (predictedWeightKg != null) 'predicted_weight_kg': predictedWeightKg,
      if (manualHeightCm != null) 'manual_height_cm': manualHeightCm,
      if (manualWeightKg != null) 'manual_weight_kg': manualWeightKg,
      if (hazZscore != null) 'haz_zscore': hazZscore,
      if (whzZscore != null) 'whz_zscore': whzZscore,
      if (hazStatus != null) 'haz_status': hazStatus,
      if (whzStatus != null) 'whz_status': whzStatus,
      if (confidenceScore != null) 'confidence_score': confidenceScore,
      if (bodyBuild != null) 'body_build': bodyBuild,
      if (estimationMethod != null) 'estimation_method': estimationMethod,
      if (sideViewUsed != null) 'side_view_used': sideViewUsed,
      if (chestDepthCm != null) 'chest_depth_cm': chestDepthCm,
      if (abdDepthCm != null) 'abd_depth_cm': abdDepthCm,
      if (mlEstimatedWeightKg != null)
        'ml_estimated_weight_kg': mlEstimatedWeightKg,
      if (samProbability != null) 'sam_probability': samProbability,
      if (mamProbability != null) 'mam_probability': mamProbability,
      if (normalProbability != null) 'normal_probability': normalProbability,
      if (riskOverweightProbability != null)
        'risk_overweight_probability': riskOverweightProbability,
      if (overweightProbability != null)
        'overweight_probability': overweightProbability,
      if (wastingStatus != null) 'wasting_status': wastingStatus,
      if (muacCm != null) 'muac_cm': muacCm,
      if (muacStatus != null) 'muac_status': muacStatus,
      if (muacMethod != null) 'muac_method': muacMethod,
    });
  }

  MeasurementsCompanion copyWith(
      {Value<int>? id,
      Value<int>? visitId,
      Value<double?>? predictedHeightCm,
      Value<double?>? predictedWeightKg,
      Value<double?>? manualHeightCm,
      Value<double?>? manualWeightKg,
      Value<double?>? hazZscore,
      Value<double?>? whzZscore,
      Value<String?>? hazStatus,
      Value<String?>? whzStatus,
      Value<double?>? confidenceScore,
      Value<String?>? bodyBuild,
      Value<String?>? estimationMethod,
      Value<bool>? sideViewUsed,
      Value<double?>? chestDepthCm,
      Value<double?>? abdDepthCm,
      Value<double?>? mlEstimatedWeightKg,
      Value<double?>? samProbability,
      Value<double?>? mamProbability,
      Value<double?>? normalProbability,
      Value<double?>? riskOverweightProbability,
      Value<double?>? overweightProbability,
      Value<String?>? wastingStatus,
      Value<double?>? muacCm,
      Value<String?>? muacStatus,
      Value<String?>? muacMethod}) {
    return MeasurementsCompanion(
      id: id ?? this.id,
      visitId: visitId ?? this.visitId,
      predictedHeightCm: predictedHeightCm ?? this.predictedHeightCm,
      predictedWeightKg: predictedWeightKg ?? this.predictedWeightKg,
      manualHeightCm: manualHeightCm ?? this.manualHeightCm,
      manualWeightKg: manualWeightKg ?? this.manualWeightKg,
      hazZscore: hazZscore ?? this.hazZscore,
      whzZscore: whzZscore ?? this.whzZscore,
      hazStatus: hazStatus ?? this.hazStatus,
      whzStatus: whzStatus ?? this.whzStatus,
      confidenceScore: confidenceScore ?? this.confidenceScore,
      bodyBuild: bodyBuild ?? this.bodyBuild,
      estimationMethod: estimationMethod ?? this.estimationMethod,
      sideViewUsed: sideViewUsed ?? this.sideViewUsed,
      chestDepthCm: chestDepthCm ?? this.chestDepthCm,
      abdDepthCm: abdDepthCm ?? this.abdDepthCm,
      mlEstimatedWeightKg: mlEstimatedWeightKg ?? this.mlEstimatedWeightKg,
      samProbability: samProbability ?? this.samProbability,
      mamProbability: mamProbability ?? this.mamProbability,
      normalProbability: normalProbability ?? this.normalProbability,
      riskOverweightProbability:
          riskOverweightProbability ?? this.riskOverweightProbability,
      overweightProbability:
          overweightProbability ?? this.overweightProbability,
      wastingStatus: wastingStatus ?? this.wastingStatus,
      muacCm: muacCm ?? this.muacCm,
      muacStatus: muacStatus ?? this.muacStatus,
      muacMethod: muacMethod ?? this.muacMethod,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (visitId.present) {
      map['visit_id'] = Variable<int>(visitId.value);
    }
    if (predictedHeightCm.present) {
      map['predicted_height_cm'] = Variable<double>(predictedHeightCm.value);
    }
    if (predictedWeightKg.present) {
      map['predicted_weight_kg'] = Variable<double>(predictedWeightKg.value);
    }
    if (manualHeightCm.present) {
      map['manual_height_cm'] = Variable<double>(manualHeightCm.value);
    }
    if (manualWeightKg.present) {
      map['manual_weight_kg'] = Variable<double>(manualWeightKg.value);
    }
    if (hazZscore.present) {
      map['haz_zscore'] = Variable<double>(hazZscore.value);
    }
    if (whzZscore.present) {
      map['whz_zscore'] = Variable<double>(whzZscore.value);
    }
    if (hazStatus.present) {
      map['haz_status'] = Variable<String>(hazStatus.value);
    }
    if (whzStatus.present) {
      map['whz_status'] = Variable<String>(whzStatus.value);
    }
    if (confidenceScore.present) {
      map['confidence_score'] = Variable<double>(confidenceScore.value);
    }
    if (bodyBuild.present) {
      map['body_build'] = Variable<String>(bodyBuild.value);
    }
    if (estimationMethod.present) {
      map['estimation_method'] = Variable<String>(estimationMethod.value);
    }
    if (sideViewUsed.present) {
      map['side_view_used'] = Variable<bool>(sideViewUsed.value);
    }
    if (chestDepthCm.present) {
      map['chest_depth_cm'] = Variable<double>(chestDepthCm.value);
    }
    if (abdDepthCm.present) {
      map['abd_depth_cm'] = Variable<double>(abdDepthCm.value);
    }
    if (mlEstimatedWeightKg.present) {
      map['ml_estimated_weight_kg'] =
          Variable<double>(mlEstimatedWeightKg.value);
    }
    if (samProbability.present) {
      map['sam_probability'] = Variable<double>(samProbability.value);
    }
    if (mamProbability.present) {
      map['mam_probability'] = Variable<double>(mamProbability.value);
    }
    if (normalProbability.present) {
      map['normal_probability'] = Variable<double>(normalProbability.value);
    }
    if (riskOverweightProbability.present) {
      map['risk_overweight_probability'] =
          Variable<double>(riskOverweightProbability.value);
    }
    if (overweightProbability.present) {
      map['overweight_probability'] =
          Variable<double>(overweightProbability.value);
    }
    if (wastingStatus.present) {
      map['wasting_status'] = Variable<String>(wastingStatus.value);
    }
    if (muacCm.present) {
      map['muac_cm'] = Variable<double>(muacCm.value);
    }
    if (muacStatus.present) {
      map['muac_status'] = Variable<String>(muacStatus.value);
    }
    if (muacMethod.present) {
      map['muac_method'] = Variable<String>(muacMethod.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('MeasurementsCompanion(')
          ..write('id: $id, ')
          ..write('visitId: $visitId, ')
          ..write('predictedHeightCm: $predictedHeightCm, ')
          ..write('predictedWeightKg: $predictedWeightKg, ')
          ..write('manualHeightCm: $manualHeightCm, ')
          ..write('manualWeightKg: $manualWeightKg, ')
          ..write('hazZscore: $hazZscore, ')
          ..write('whzZscore: $whzZscore, ')
          ..write('hazStatus: $hazStatus, ')
          ..write('whzStatus: $whzStatus, ')
          ..write('confidenceScore: $confidenceScore, ')
          ..write('bodyBuild: $bodyBuild, ')
          ..write('estimationMethod: $estimationMethod, ')
          ..write('sideViewUsed: $sideViewUsed, ')
          ..write('chestDepthCm: $chestDepthCm, ')
          ..write('abdDepthCm: $abdDepthCm, ')
          ..write('mlEstimatedWeightKg: $mlEstimatedWeightKg, ')
          ..write('samProbability: $samProbability, ')
          ..write('mamProbability: $mamProbability, ')
          ..write('normalProbability: $normalProbability, ')
          ..write('riskOverweightProbability: $riskOverweightProbability, ')
          ..write('overweightProbability: $overweightProbability, ')
          ..write('wastingStatus: $wastingStatus, ')
          ..write('muacCm: $muacCm, ')
          ..write('muacStatus: $muacStatus, ')
          ..write('muacMethod: $muacMethod')
          ..write(')'))
        .toString();
  }
}

class $SyncQueueTable extends SyncQueue
    with TableInfo<$SyncQueueTable, SyncQueueData> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $SyncQueueTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _visitIdMeta =
      const VerificationMeta('visitId');
  @override
  late final GeneratedColumn<int> visitId = GeneratedColumn<int>(
      'visit_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('REFERENCES visits (id)'));
  static const VerificationMeta _statusMeta = const VerificationMeta('status');
  @override
  late final GeneratedColumn<String> status = GeneratedColumn<String>(
      'status', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant('pending'));
  static const VerificationMeta _retryCountMeta =
      const VerificationMeta('retryCount');
  @override
  late final GeneratedColumn<int> retryCount = GeneratedColumn<int>(
      'retry_count', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultValue: const Constant(0));
  static const VerificationMeta _createdAtMeta =
      const VerificationMeta('createdAt');
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
      'created_at', aliasedName, false,
      type: DriftSqlType.dateTime,
      requiredDuringInsert: false,
      defaultValue: currentDateAndTime);
  static const VerificationMeta _lastAttemptAtMeta =
      const VerificationMeta('lastAttemptAt');
  @override
  late final GeneratedColumn<DateTime> lastAttemptAt =
      GeneratedColumn<DateTime>('last_attempt_at', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _serverVisitIdMeta =
      const VerificationMeta('serverVisitId');
  @override
  late final GeneratedColumn<int> serverVisitId = GeneratedColumn<int>(
      'server_visit_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _errorMessageMeta =
      const VerificationMeta('errorMessage');
  @override
  late final GeneratedColumn<String> errorMessage = GeneratedColumn<String>(
      'error_message', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        visitId,
        status,
        retryCount,
        createdAt,
        lastAttemptAt,
        serverVisitId,
        errorMessage
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'sync_queue';
  @override
  VerificationContext validateIntegrity(Insertable<SyncQueueData> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('visit_id')) {
      context.handle(_visitIdMeta,
          visitId.isAcceptableOrUnknown(data['visit_id']!, _visitIdMeta));
    } else if (isInserting) {
      context.missing(_visitIdMeta);
    }
    if (data.containsKey('status')) {
      context.handle(_statusMeta,
          status.isAcceptableOrUnknown(data['status']!, _statusMeta));
    }
    if (data.containsKey('retry_count')) {
      context.handle(
          _retryCountMeta,
          retryCount.isAcceptableOrUnknown(
              data['retry_count']!, _retryCountMeta));
    }
    if (data.containsKey('created_at')) {
      context.handle(_createdAtMeta,
          createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta));
    }
    if (data.containsKey('last_attempt_at')) {
      context.handle(
          _lastAttemptAtMeta,
          lastAttemptAt.isAcceptableOrUnknown(
              data['last_attempt_at']!, _lastAttemptAtMeta));
    }
    if (data.containsKey('server_visit_id')) {
      context.handle(
          _serverVisitIdMeta,
          serverVisitId.isAcceptableOrUnknown(
              data['server_visit_id']!, _serverVisitIdMeta));
    }
    if (data.containsKey('error_message')) {
      context.handle(
          _errorMessageMeta,
          errorMessage.isAcceptableOrUnknown(
              data['error_message']!, _errorMessageMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  SyncQueueData map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return SyncQueueData(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      visitId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}visit_id'])!,
      status: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}status'])!,
      retryCount: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}retry_count'])!,
      createdAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}created_at'])!,
      lastAttemptAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime, data['${effectivePrefix}last_attempt_at']),
      serverVisitId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}server_visit_id']),
      errorMessage: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}error_message']),
    );
  }

  @override
  $SyncQueueTable createAlias(String alias) {
    return $SyncQueueTable(attachedDatabase, alias);
  }
}

class SyncQueueData extends DataClass implements Insertable<SyncQueueData> {
  final int id;
  final int visitId;
  final String status;
  final int retryCount;
  final DateTime createdAt;
  final DateTime? lastAttemptAt;
  final int? serverVisitId;
  final String? errorMessage;
  const SyncQueueData(
      {required this.id,
      required this.visitId,
      required this.status,
      required this.retryCount,
      required this.createdAt,
      this.lastAttemptAt,
      this.serverVisitId,
      this.errorMessage});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['visit_id'] = Variable<int>(visitId);
    map['status'] = Variable<String>(status);
    map['retry_count'] = Variable<int>(retryCount);
    map['created_at'] = Variable<DateTime>(createdAt);
    if (!nullToAbsent || lastAttemptAt != null) {
      map['last_attempt_at'] = Variable<DateTime>(lastAttemptAt);
    }
    if (!nullToAbsent || serverVisitId != null) {
      map['server_visit_id'] = Variable<int>(serverVisitId);
    }
    if (!nullToAbsent || errorMessage != null) {
      map['error_message'] = Variable<String>(errorMessage);
    }
    return map;
  }

  SyncQueueCompanion toCompanion(bool nullToAbsent) {
    return SyncQueueCompanion(
      id: Value(id),
      visitId: Value(visitId),
      status: Value(status),
      retryCount: Value(retryCount),
      createdAt: Value(createdAt),
      lastAttemptAt: lastAttemptAt == null && nullToAbsent
          ? const Value.absent()
          : Value(lastAttemptAt),
      serverVisitId: serverVisitId == null && nullToAbsent
          ? const Value.absent()
          : Value(serverVisitId),
      errorMessage: errorMessage == null && nullToAbsent
          ? const Value.absent()
          : Value(errorMessage),
    );
  }

  factory SyncQueueData.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return SyncQueueData(
      id: serializer.fromJson<int>(json['id']),
      visitId: serializer.fromJson<int>(json['visitId']),
      status: serializer.fromJson<String>(json['status']),
      retryCount: serializer.fromJson<int>(json['retryCount']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
      lastAttemptAt: serializer.fromJson<DateTime?>(json['lastAttemptAt']),
      serverVisitId: serializer.fromJson<int?>(json['serverVisitId']),
      errorMessage: serializer.fromJson<String?>(json['errorMessage']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'visitId': serializer.toJson<int>(visitId),
      'status': serializer.toJson<String>(status),
      'retryCount': serializer.toJson<int>(retryCount),
      'createdAt': serializer.toJson<DateTime>(createdAt),
      'lastAttemptAt': serializer.toJson<DateTime?>(lastAttemptAt),
      'serverVisitId': serializer.toJson<int?>(serverVisitId),
      'errorMessage': serializer.toJson<String?>(errorMessage),
    };
  }

  SyncQueueData copyWith(
          {int? id,
          int? visitId,
          String? status,
          int? retryCount,
          DateTime? createdAt,
          Value<DateTime?> lastAttemptAt = const Value.absent(),
          Value<int?> serverVisitId = const Value.absent(),
          Value<String?> errorMessage = const Value.absent()}) =>
      SyncQueueData(
        id: id ?? this.id,
        visitId: visitId ?? this.visitId,
        status: status ?? this.status,
        retryCount: retryCount ?? this.retryCount,
        createdAt: createdAt ?? this.createdAt,
        lastAttemptAt:
            lastAttemptAt.present ? lastAttemptAt.value : this.lastAttemptAt,
        serverVisitId:
            serverVisitId.present ? serverVisitId.value : this.serverVisitId,
        errorMessage:
            errorMessage.present ? errorMessage.value : this.errorMessage,
      );
  SyncQueueData copyWithCompanion(SyncQueueCompanion data) {
    return SyncQueueData(
      id: data.id.present ? data.id.value : this.id,
      visitId: data.visitId.present ? data.visitId.value : this.visitId,
      status: data.status.present ? data.status.value : this.status,
      retryCount:
          data.retryCount.present ? data.retryCount.value : this.retryCount,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
      lastAttemptAt: data.lastAttemptAt.present
          ? data.lastAttemptAt.value
          : this.lastAttemptAt,
      serverVisitId: data.serverVisitId.present
          ? data.serverVisitId.value
          : this.serverVisitId,
      errorMessage: data.errorMessage.present
          ? data.errorMessage.value
          : this.errorMessage,
    );
  }

  @override
  String toString() {
    return (StringBuffer('SyncQueueData(')
          ..write('id: $id, ')
          ..write('visitId: $visitId, ')
          ..write('status: $status, ')
          ..write('retryCount: $retryCount, ')
          ..write('createdAt: $createdAt, ')
          ..write('lastAttemptAt: $lastAttemptAt, ')
          ..write('serverVisitId: $serverVisitId, ')
          ..write('errorMessage: $errorMessage')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, visitId, status, retryCount, createdAt,
      lastAttemptAt, serverVisitId, errorMessage);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is SyncQueueData &&
          other.id == this.id &&
          other.visitId == this.visitId &&
          other.status == this.status &&
          other.retryCount == this.retryCount &&
          other.createdAt == this.createdAt &&
          other.lastAttemptAt == this.lastAttemptAt &&
          other.serverVisitId == this.serverVisitId &&
          other.errorMessage == this.errorMessage);
}

class SyncQueueCompanion extends UpdateCompanion<SyncQueueData> {
  final Value<int> id;
  final Value<int> visitId;
  final Value<String> status;
  final Value<int> retryCount;
  final Value<DateTime> createdAt;
  final Value<DateTime?> lastAttemptAt;
  final Value<int?> serverVisitId;
  final Value<String?> errorMessage;
  const SyncQueueCompanion({
    this.id = const Value.absent(),
    this.visitId = const Value.absent(),
    this.status = const Value.absent(),
    this.retryCount = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.lastAttemptAt = const Value.absent(),
    this.serverVisitId = const Value.absent(),
    this.errorMessage = const Value.absent(),
  });
  SyncQueueCompanion.insert({
    this.id = const Value.absent(),
    required int visitId,
    this.status = const Value.absent(),
    this.retryCount = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.lastAttemptAt = const Value.absent(),
    this.serverVisitId = const Value.absent(),
    this.errorMessage = const Value.absent(),
  }) : visitId = Value(visitId);
  static Insertable<SyncQueueData> custom({
    Expression<int>? id,
    Expression<int>? visitId,
    Expression<String>? status,
    Expression<int>? retryCount,
    Expression<DateTime>? createdAt,
    Expression<DateTime>? lastAttemptAt,
    Expression<int>? serverVisitId,
    Expression<String>? errorMessage,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (visitId != null) 'visit_id': visitId,
      if (status != null) 'status': status,
      if (retryCount != null) 'retry_count': retryCount,
      if (createdAt != null) 'created_at': createdAt,
      if (lastAttemptAt != null) 'last_attempt_at': lastAttemptAt,
      if (serverVisitId != null) 'server_visit_id': serverVisitId,
      if (errorMessage != null) 'error_message': errorMessage,
    });
  }

  SyncQueueCompanion copyWith(
      {Value<int>? id,
      Value<int>? visitId,
      Value<String>? status,
      Value<int>? retryCount,
      Value<DateTime>? createdAt,
      Value<DateTime?>? lastAttemptAt,
      Value<int?>? serverVisitId,
      Value<String?>? errorMessage}) {
    return SyncQueueCompanion(
      id: id ?? this.id,
      visitId: visitId ?? this.visitId,
      status: status ?? this.status,
      retryCount: retryCount ?? this.retryCount,
      createdAt: createdAt ?? this.createdAt,
      lastAttemptAt: lastAttemptAt ?? this.lastAttemptAt,
      serverVisitId: serverVisitId ?? this.serverVisitId,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (visitId.present) {
      map['visit_id'] = Variable<int>(visitId.value);
    }
    if (status.present) {
      map['status'] = Variable<String>(status.value);
    }
    if (retryCount.present) {
      map['retry_count'] = Variable<int>(retryCount.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    if (lastAttemptAt.present) {
      map['last_attempt_at'] = Variable<DateTime>(lastAttemptAt.value);
    }
    if (serverVisitId.present) {
      map['server_visit_id'] = Variable<int>(serverVisitId.value);
    }
    if (errorMessage.present) {
      map['error_message'] = Variable<String>(errorMessage.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('SyncQueueCompanion(')
          ..write('id: $id, ')
          ..write('visitId: $visitId, ')
          ..write('status: $status, ')
          ..write('retryCount: $retryCount, ')
          ..write('createdAt: $createdAt, ')
          ..write('lastAttemptAt: $lastAttemptAt, ')
          ..write('serverVisitId: $serverVisitId, ')
          ..write('errorMessage: $errorMessage')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $ChildrenTable children = $ChildrenTable(this);
  late final $VisitsTable visits = $VisitsTable(this);
  late final $MeasurementsTable measurements = $MeasurementsTable(this);
  late final $SyncQueueTable syncQueue = $SyncQueueTable(this);
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities =>
      [children, visits, measurements, syncQueue];
}

typedef $$ChildrenTableCreateCompanionBuilder = ChildrenCompanion Function({
  Value<int> id,
  required String name,
  required String dateOfBirth,
  required String sex,
  Value<String?> guardianName,
  Value<String?> location,
  Value<DateTime> createdAt,
  Value<DateTime> updatedAt,
});
typedef $$ChildrenTableUpdateCompanionBuilder = ChildrenCompanion Function({
  Value<int> id,
  Value<String> name,
  Value<String> dateOfBirth,
  Value<String> sex,
  Value<String?> guardianName,
  Value<String?> location,
  Value<DateTime> createdAt,
  Value<DateTime> updatedAt,
});

final class $$ChildrenTableReferences
    extends BaseReferences<_$AppDatabase, $ChildrenTable, ChildrenData> {
  $$ChildrenTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static MultiTypedResultKey<$VisitsTable, List<Visit>> _visitsRefsTable(
          _$AppDatabase db) =>
      MultiTypedResultKey.fromTable(db.visits,
          aliasName: $_aliasNameGenerator(db.children.id, db.visits.childId));

  $$VisitsTableProcessedTableManager get visitsRefs {
    final manager = $$VisitsTableTableManager($_db, $_db.visits)
        .filter((f) => f.childId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_visitsRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }
}

class $$ChildrenTableFilterComposer
    extends Composer<_$AppDatabase, $ChildrenTable> {
  $$ChildrenTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get name => $composableBuilder(
      column: $table.name, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get dateOfBirth => $composableBuilder(
      column: $table.dateOfBirth, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get sex => $composableBuilder(
      column: $table.sex, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get guardianName => $composableBuilder(
      column: $table.guardianName, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get location => $composableBuilder(
      column: $table.location, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get updatedAt => $composableBuilder(
      column: $table.updatedAt, builder: (column) => ColumnFilters(column));

  Expression<bool> visitsRefs(
      Expression<bool> Function($$VisitsTableFilterComposer f) f) {
    final $$VisitsTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.childId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableFilterComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }
}

class $$ChildrenTableOrderingComposer
    extends Composer<_$AppDatabase, $ChildrenTable> {
  $$ChildrenTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get name => $composableBuilder(
      column: $table.name, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get dateOfBirth => $composableBuilder(
      column: $table.dateOfBirth, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get sex => $composableBuilder(
      column: $table.sex, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get guardianName => $composableBuilder(
      column: $table.guardianName,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get location => $composableBuilder(
      column: $table.location, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get updatedAt => $composableBuilder(
      column: $table.updatedAt, builder: (column) => ColumnOrderings(column));
}

class $$ChildrenTableAnnotationComposer
    extends Composer<_$AppDatabase, $ChildrenTable> {
  $$ChildrenTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<String> get dateOfBirth => $composableBuilder(
      column: $table.dateOfBirth, builder: (column) => column);

  GeneratedColumn<String> get sex =>
      $composableBuilder(column: $table.sex, builder: (column) => column);

  GeneratedColumn<String> get guardianName => $composableBuilder(
      column: $table.guardianName, builder: (column) => column);

  GeneratedColumn<String> get location =>
      $composableBuilder(column: $table.location, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  GeneratedColumn<DateTime> get updatedAt =>
      $composableBuilder(column: $table.updatedAt, builder: (column) => column);

  Expression<T> visitsRefs<T extends Object>(
      Expression<T> Function($$VisitsTableAnnotationComposer a) f) {
    final $$VisitsTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.childId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableAnnotationComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }
}

class $$ChildrenTableTableManager extends RootTableManager<
    _$AppDatabase,
    $ChildrenTable,
    ChildrenData,
    $$ChildrenTableFilterComposer,
    $$ChildrenTableOrderingComposer,
    $$ChildrenTableAnnotationComposer,
    $$ChildrenTableCreateCompanionBuilder,
    $$ChildrenTableUpdateCompanionBuilder,
    (ChildrenData, $$ChildrenTableReferences),
    ChildrenData,
    PrefetchHooks Function({bool visitsRefs})> {
  $$ChildrenTableTableManager(_$AppDatabase db, $ChildrenTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ChildrenTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ChildrenTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ChildrenTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<String> name = const Value.absent(),
            Value<String> dateOfBirth = const Value.absent(),
            Value<String> sex = const Value.absent(),
            Value<String?> guardianName = const Value.absent(),
            Value<String?> location = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<DateTime> updatedAt = const Value.absent(),
          }) =>
              ChildrenCompanion(
            id: id,
            name: name,
            dateOfBirth: dateOfBirth,
            sex: sex,
            guardianName: guardianName,
            location: location,
            createdAt: createdAt,
            updatedAt: updatedAt,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required String name,
            required String dateOfBirth,
            required String sex,
            Value<String?> guardianName = const Value.absent(),
            Value<String?> location = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<DateTime> updatedAt = const Value.absent(),
          }) =>
              ChildrenCompanion.insert(
            id: id,
            name: name,
            dateOfBirth: dateOfBirth,
            sex: sex,
            guardianName: guardianName,
            location: location,
            createdAt: createdAt,
            updatedAt: updatedAt,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) =>
                  (e.readTable(table), $$ChildrenTableReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: ({visitsRefs = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [if (visitsRefs) db.visits],
              addJoins: null,
              getPrefetchedDataCallback: (items) async {
                return [
                  if (visitsRefs)
                    await $_getPrefetchedData<ChildrenData, $ChildrenTable,
                            Visit>(
                        currentTable: table,
                        referencedTable:
                            $$ChildrenTableReferences._visitsRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$ChildrenTableReferences(db, table, p0).visitsRefs,
                        referencedItemsForCurrentItem: (item,
                                referencedItems) =>
                            referencedItems.where((e) => e.childId == item.id),
                        typedResults: items)
                ];
              },
            );
          },
        ));
}

typedef $$ChildrenTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $ChildrenTable,
    ChildrenData,
    $$ChildrenTableFilterComposer,
    $$ChildrenTableOrderingComposer,
    $$ChildrenTableAnnotationComposer,
    $$ChildrenTableCreateCompanionBuilder,
    $$ChildrenTableUpdateCompanionBuilder,
    (ChildrenData, $$ChildrenTableReferences),
    ChildrenData,
    PrefetchHooks Function({bool visitsRefs})>;
typedef $$VisitsTableCreateCompanionBuilder = VisitsCompanion Function({
  Value<int> id,
  required int childId,
  Value<DateTime> visitDate,
  required double ageMonths,
  required String imagePath,
  Value<String?> sideImagePath,
  Value<String?> backImagePath,
  Value<String?> notes,
});
typedef $$VisitsTableUpdateCompanionBuilder = VisitsCompanion Function({
  Value<int> id,
  Value<int> childId,
  Value<DateTime> visitDate,
  Value<double> ageMonths,
  Value<String> imagePath,
  Value<String?> sideImagePath,
  Value<String?> backImagePath,
  Value<String?> notes,
});

final class $$VisitsTableReferences
    extends BaseReferences<_$AppDatabase, $VisitsTable, Visit> {
  $$VisitsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $ChildrenTable _childIdTable(_$AppDatabase db) => db.children
      .createAlias($_aliasNameGenerator(db.visits.childId, db.children.id));

  $$ChildrenTableProcessedTableManager get childId {
    final $_column = $_itemColumn<int>('child_id')!;

    final manager = $$ChildrenTableTableManager($_db, $_db.children)
        .filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_childIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: [item]));
  }

  static MultiTypedResultKey<$MeasurementsTable, List<Measurement>>
      _measurementsRefsTable(_$AppDatabase db) =>
          MultiTypedResultKey.fromTable(db.measurements,
              aliasName:
                  $_aliasNameGenerator(db.visits.id, db.measurements.visitId));

  $$MeasurementsTableProcessedTableManager get measurementsRefs {
    final manager = $$MeasurementsTableTableManager($_db, $_db.measurements)
        .filter((f) => f.visitId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_measurementsRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }

  static MultiTypedResultKey<$SyncQueueTable, List<SyncQueueData>>
      _syncQueueRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
          db.syncQueue,
          aliasName: $_aliasNameGenerator(db.visits.id, db.syncQueue.visitId));

  $$SyncQueueTableProcessedTableManager get syncQueueRefs {
    final manager = $$SyncQueueTableTableManager($_db, $_db.syncQueue)
        .filter((f) => f.visitId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_syncQueueRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }
}

class $$VisitsTableFilterComposer
    extends Composer<_$AppDatabase, $VisitsTable> {
  $$VisitsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get visitDate => $composableBuilder(
      column: $table.visitDate, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get ageMonths => $composableBuilder(
      column: $table.ageMonths, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get imagePath => $composableBuilder(
      column: $table.imagePath, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get sideImagePath => $composableBuilder(
      column: $table.sideImagePath, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get backImagePath => $composableBuilder(
      column: $table.backImagePath, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get notes => $composableBuilder(
      column: $table.notes, builder: (column) => ColumnFilters(column));

  $$ChildrenTableFilterComposer get childId {
    final $$ChildrenTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.childId,
        referencedTable: $db.children,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$ChildrenTableFilterComposer(
              $db: $db,
              $table: $db.children,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }

  Expression<bool> measurementsRefs(
      Expression<bool> Function($$MeasurementsTableFilterComposer f) f) {
    final $$MeasurementsTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.measurements,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$MeasurementsTableFilterComposer(
              $db: $db,
              $table: $db.measurements,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }

  Expression<bool> syncQueueRefs(
      Expression<bool> Function($$SyncQueueTableFilterComposer f) f) {
    final $$SyncQueueTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.syncQueue,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$SyncQueueTableFilterComposer(
              $db: $db,
              $table: $db.syncQueue,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }
}

class $$VisitsTableOrderingComposer
    extends Composer<_$AppDatabase, $VisitsTable> {
  $$VisitsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get visitDate => $composableBuilder(
      column: $table.visitDate, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get ageMonths => $composableBuilder(
      column: $table.ageMonths, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get imagePath => $composableBuilder(
      column: $table.imagePath, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get sideImagePath => $composableBuilder(
      column: $table.sideImagePath,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get backImagePath => $composableBuilder(
      column: $table.backImagePath,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get notes => $composableBuilder(
      column: $table.notes, builder: (column) => ColumnOrderings(column));

  $$ChildrenTableOrderingComposer get childId {
    final $$ChildrenTableOrderingComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.childId,
        referencedTable: $db.children,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$ChildrenTableOrderingComposer(
              $db: $db,
              $table: $db.children,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$VisitsTableAnnotationComposer
    extends Composer<_$AppDatabase, $VisitsTable> {
  $$VisitsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<DateTime> get visitDate =>
      $composableBuilder(column: $table.visitDate, builder: (column) => column);

  GeneratedColumn<double> get ageMonths =>
      $composableBuilder(column: $table.ageMonths, builder: (column) => column);

  GeneratedColumn<String> get imagePath =>
      $composableBuilder(column: $table.imagePath, builder: (column) => column);

  GeneratedColumn<String> get sideImagePath => $composableBuilder(
      column: $table.sideImagePath, builder: (column) => column);

  GeneratedColumn<String> get backImagePath => $composableBuilder(
      column: $table.backImagePath, builder: (column) => column);

  GeneratedColumn<String> get notes =>
      $composableBuilder(column: $table.notes, builder: (column) => column);

  $$ChildrenTableAnnotationComposer get childId {
    final $$ChildrenTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.childId,
        referencedTable: $db.children,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$ChildrenTableAnnotationComposer(
              $db: $db,
              $table: $db.children,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }

  Expression<T> measurementsRefs<T extends Object>(
      Expression<T> Function($$MeasurementsTableAnnotationComposer a) f) {
    final $$MeasurementsTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.measurements,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$MeasurementsTableAnnotationComposer(
              $db: $db,
              $table: $db.measurements,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }

  Expression<T> syncQueueRefs<T extends Object>(
      Expression<T> Function($$SyncQueueTableAnnotationComposer a) f) {
    final $$SyncQueueTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.syncQueue,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$SyncQueueTableAnnotationComposer(
              $db: $db,
              $table: $db.syncQueue,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }
}

class $$VisitsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $VisitsTable,
    Visit,
    $$VisitsTableFilterComposer,
    $$VisitsTableOrderingComposer,
    $$VisitsTableAnnotationComposer,
    $$VisitsTableCreateCompanionBuilder,
    $$VisitsTableUpdateCompanionBuilder,
    (Visit, $$VisitsTableReferences),
    Visit,
    PrefetchHooks Function(
        {bool childId, bool measurementsRefs, bool syncQueueRefs})> {
  $$VisitsTableTableManager(_$AppDatabase db, $VisitsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$VisitsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$VisitsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$VisitsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> childId = const Value.absent(),
            Value<DateTime> visitDate = const Value.absent(),
            Value<double> ageMonths = const Value.absent(),
            Value<String> imagePath = const Value.absent(),
            Value<String?> sideImagePath = const Value.absent(),
            Value<String?> backImagePath = const Value.absent(),
            Value<String?> notes = const Value.absent(),
          }) =>
              VisitsCompanion(
            id: id,
            childId: childId,
            visitDate: visitDate,
            ageMonths: ageMonths,
            imagePath: imagePath,
            sideImagePath: sideImagePath,
            backImagePath: backImagePath,
            notes: notes,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int childId,
            Value<DateTime> visitDate = const Value.absent(),
            required double ageMonths,
            required String imagePath,
            Value<String?> sideImagePath = const Value.absent(),
            Value<String?> backImagePath = const Value.absent(),
            Value<String?> notes = const Value.absent(),
          }) =>
              VisitsCompanion.insert(
            id: id,
            childId: childId,
            visitDate: visitDate,
            ageMonths: ageMonths,
            imagePath: imagePath,
            sideImagePath: sideImagePath,
            backImagePath: backImagePath,
            notes: notes,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) =>
                  (e.readTable(table), $$VisitsTableReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: (
              {childId = false,
              measurementsRefs = false,
              syncQueueRefs = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [
                if (measurementsRefs) db.measurements,
                if (syncQueueRefs) db.syncQueue
              ],
              addJoins: <
                  T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic>>(state) {
                if (childId) {
                  state = state.withJoin(
                    currentTable: table,
                    currentColumn: table.childId,
                    referencedTable: $$VisitsTableReferences._childIdTable(db),
                    referencedColumn:
                        $$VisitsTableReferences._childIdTable(db).id,
                  ) as T;
                }

                return state;
              },
              getPrefetchedDataCallback: (items) async {
                return [
                  if (measurementsRefs)
                    await $_getPrefetchedData<Visit, $VisitsTable, Measurement>(
                        currentTable: table,
                        referencedTable:
                            $$VisitsTableReferences._measurementsRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$VisitsTableReferences(db, table, p0)
                                .measurementsRefs,
                        referencedItemsForCurrentItem: (item,
                                referencedItems) =>
                            referencedItems.where((e) => e.visitId == item.id),
                        typedResults: items),
                  if (syncQueueRefs)
                    await $_getPrefetchedData<Visit, $VisitsTable,
                            SyncQueueData>(
                        currentTable: table,
                        referencedTable:
                            $$VisitsTableReferences._syncQueueRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$VisitsTableReferences(db, table, p0)
                                .syncQueueRefs,
                        referencedItemsForCurrentItem: (item,
                                referencedItems) =>
                            referencedItems.where((e) => e.visitId == item.id),
                        typedResults: items)
                ];
              },
            );
          },
        ));
}

typedef $$VisitsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $VisitsTable,
    Visit,
    $$VisitsTableFilterComposer,
    $$VisitsTableOrderingComposer,
    $$VisitsTableAnnotationComposer,
    $$VisitsTableCreateCompanionBuilder,
    $$VisitsTableUpdateCompanionBuilder,
    (Visit, $$VisitsTableReferences),
    Visit,
    PrefetchHooks Function(
        {bool childId, bool measurementsRefs, bool syncQueueRefs})>;
typedef $$MeasurementsTableCreateCompanionBuilder = MeasurementsCompanion
    Function({
  Value<int> id,
  required int visitId,
  Value<double?> predictedHeightCm,
  Value<double?> predictedWeightKg,
  Value<double?> manualHeightCm,
  Value<double?> manualWeightKg,
  Value<double?> hazZscore,
  Value<double?> whzZscore,
  Value<String?> hazStatus,
  Value<String?> whzStatus,
  Value<double?> confidenceScore,
  Value<String?> bodyBuild,
  Value<String?> estimationMethod,
  Value<bool> sideViewUsed,
  Value<double?> chestDepthCm,
  Value<double?> abdDepthCm,
  Value<double?> mlEstimatedWeightKg,
  Value<double?> samProbability,
  Value<double?> mamProbability,
  Value<double?> normalProbability,
  Value<double?> riskOverweightProbability,
  Value<double?> overweightProbability,
  Value<String?> wastingStatus,
  Value<double?> muacCm,
  Value<String?> muacStatus,
  Value<String?> muacMethod,
});
typedef $$MeasurementsTableUpdateCompanionBuilder = MeasurementsCompanion
    Function({
  Value<int> id,
  Value<int> visitId,
  Value<double?> predictedHeightCm,
  Value<double?> predictedWeightKg,
  Value<double?> manualHeightCm,
  Value<double?> manualWeightKg,
  Value<double?> hazZscore,
  Value<double?> whzZscore,
  Value<String?> hazStatus,
  Value<String?> whzStatus,
  Value<double?> confidenceScore,
  Value<String?> bodyBuild,
  Value<String?> estimationMethod,
  Value<bool> sideViewUsed,
  Value<double?> chestDepthCm,
  Value<double?> abdDepthCm,
  Value<double?> mlEstimatedWeightKg,
  Value<double?> samProbability,
  Value<double?> mamProbability,
  Value<double?> normalProbability,
  Value<double?> riskOverweightProbability,
  Value<double?> overweightProbability,
  Value<String?> wastingStatus,
  Value<double?> muacCm,
  Value<String?> muacStatus,
  Value<String?> muacMethod,
});

final class $$MeasurementsTableReferences
    extends BaseReferences<_$AppDatabase, $MeasurementsTable, Measurement> {
  $$MeasurementsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $VisitsTable _visitIdTable(_$AppDatabase db) => db.visits
      .createAlias($_aliasNameGenerator(db.measurements.visitId, db.visits.id));

  $$VisitsTableProcessedTableManager get visitId {
    final $_column = $_itemColumn<int>('visit_id')!;

    final manager = $$VisitsTableTableManager($_db, $_db.visits)
        .filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_visitIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: [item]));
  }
}

class $$MeasurementsTableFilterComposer
    extends Composer<_$AppDatabase, $MeasurementsTable> {
  $$MeasurementsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get predictedHeightCm => $composableBuilder(
      column: $table.predictedHeightCm,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get predictedWeightKg => $composableBuilder(
      column: $table.predictedWeightKg,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get manualHeightCm => $composableBuilder(
      column: $table.manualHeightCm,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get manualWeightKg => $composableBuilder(
      column: $table.manualWeightKg,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get hazZscore => $composableBuilder(
      column: $table.hazZscore, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get whzZscore => $composableBuilder(
      column: $table.whzZscore, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get hazStatus => $composableBuilder(
      column: $table.hazStatus, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get whzStatus => $composableBuilder(
      column: $table.whzStatus, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get confidenceScore => $composableBuilder(
      column: $table.confidenceScore,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get bodyBuild => $composableBuilder(
      column: $table.bodyBuild, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get estimationMethod => $composableBuilder(
      column: $table.estimationMethod,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get sideViewUsed => $composableBuilder(
      column: $table.sideViewUsed, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get chestDepthCm => $composableBuilder(
      column: $table.chestDepthCm, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get abdDepthCm => $composableBuilder(
      column: $table.abdDepthCm, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get mlEstimatedWeightKg => $composableBuilder(
      column: $table.mlEstimatedWeightKg,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get samProbability => $composableBuilder(
      column: $table.samProbability,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get mamProbability => $composableBuilder(
      column: $table.mamProbability,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get normalProbability => $composableBuilder(
      column: $table.normalProbability,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get riskOverweightProbability => $composableBuilder(
      column: $table.riskOverweightProbability,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get overweightProbability => $composableBuilder(
      column: $table.overweightProbability,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get wastingStatus => $composableBuilder(
      column: $table.wastingStatus, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get muacCm => $composableBuilder(
      column: $table.muacCm, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacStatus => $composableBuilder(
      column: $table.muacStatus, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacMethod => $composableBuilder(
      column: $table.muacMethod, builder: (column) => ColumnFilters(column));

  $$VisitsTableFilterComposer get visitId {
    final $$VisitsTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.visitId,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableFilterComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$MeasurementsTableOrderingComposer
    extends Composer<_$AppDatabase, $MeasurementsTable> {
  $$MeasurementsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get predictedHeightCm => $composableBuilder(
      column: $table.predictedHeightCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get predictedWeightKg => $composableBuilder(
      column: $table.predictedWeightKg,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get manualHeightCm => $composableBuilder(
      column: $table.manualHeightCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get manualWeightKg => $composableBuilder(
      column: $table.manualWeightKg,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get hazZscore => $composableBuilder(
      column: $table.hazZscore, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get whzZscore => $composableBuilder(
      column: $table.whzZscore, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get hazStatus => $composableBuilder(
      column: $table.hazStatus, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get whzStatus => $composableBuilder(
      column: $table.whzStatus, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get confidenceScore => $composableBuilder(
      column: $table.confidenceScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get bodyBuild => $composableBuilder(
      column: $table.bodyBuild, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get estimationMethod => $composableBuilder(
      column: $table.estimationMethod,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get sideViewUsed => $composableBuilder(
      column: $table.sideViewUsed,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get chestDepthCm => $composableBuilder(
      column: $table.chestDepthCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get abdDepthCm => $composableBuilder(
      column: $table.abdDepthCm, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get mlEstimatedWeightKg => $composableBuilder(
      column: $table.mlEstimatedWeightKg,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get samProbability => $composableBuilder(
      column: $table.samProbability,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get mamProbability => $composableBuilder(
      column: $table.mamProbability,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get normalProbability => $composableBuilder(
      column: $table.normalProbability,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get riskOverweightProbability => $composableBuilder(
      column: $table.riskOverweightProbability,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get overweightProbability => $composableBuilder(
      column: $table.overweightProbability,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get wastingStatus => $composableBuilder(
      column: $table.wastingStatus,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get muacCm => $composableBuilder(
      column: $table.muacCm, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacStatus => $composableBuilder(
      column: $table.muacStatus, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacMethod => $composableBuilder(
      column: $table.muacMethod, builder: (column) => ColumnOrderings(column));

  $$VisitsTableOrderingComposer get visitId {
    final $$VisitsTableOrderingComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.visitId,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableOrderingComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$MeasurementsTableAnnotationComposer
    extends Composer<_$AppDatabase, $MeasurementsTable> {
  $$MeasurementsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<double> get predictedHeightCm => $composableBuilder(
      column: $table.predictedHeightCm, builder: (column) => column);

  GeneratedColumn<double> get predictedWeightKg => $composableBuilder(
      column: $table.predictedWeightKg, builder: (column) => column);

  GeneratedColumn<double> get manualHeightCm => $composableBuilder(
      column: $table.manualHeightCm, builder: (column) => column);

  GeneratedColumn<double> get manualWeightKg => $composableBuilder(
      column: $table.manualWeightKg, builder: (column) => column);

  GeneratedColumn<double> get hazZscore =>
      $composableBuilder(column: $table.hazZscore, builder: (column) => column);

  GeneratedColumn<double> get whzZscore =>
      $composableBuilder(column: $table.whzZscore, builder: (column) => column);

  GeneratedColumn<String> get hazStatus =>
      $composableBuilder(column: $table.hazStatus, builder: (column) => column);

  GeneratedColumn<String> get whzStatus =>
      $composableBuilder(column: $table.whzStatus, builder: (column) => column);

  GeneratedColumn<double> get confidenceScore => $composableBuilder(
      column: $table.confidenceScore, builder: (column) => column);

  GeneratedColumn<String> get bodyBuild =>
      $composableBuilder(column: $table.bodyBuild, builder: (column) => column);

  GeneratedColumn<String> get estimationMethod => $composableBuilder(
      column: $table.estimationMethod, builder: (column) => column);

  GeneratedColumn<bool> get sideViewUsed => $composableBuilder(
      column: $table.sideViewUsed, builder: (column) => column);

  GeneratedColumn<double> get chestDepthCm => $composableBuilder(
      column: $table.chestDepthCm, builder: (column) => column);

  GeneratedColumn<double> get abdDepthCm => $composableBuilder(
      column: $table.abdDepthCm, builder: (column) => column);

  GeneratedColumn<double> get mlEstimatedWeightKg => $composableBuilder(
      column: $table.mlEstimatedWeightKg, builder: (column) => column);

  GeneratedColumn<double> get samProbability => $composableBuilder(
      column: $table.samProbability, builder: (column) => column);

  GeneratedColumn<double> get mamProbability => $composableBuilder(
      column: $table.mamProbability, builder: (column) => column);

  GeneratedColumn<double> get normalProbability => $composableBuilder(
      column: $table.normalProbability, builder: (column) => column);

  GeneratedColumn<double> get riskOverweightProbability => $composableBuilder(
      column: $table.riskOverweightProbability, builder: (column) => column);

  GeneratedColumn<double> get overweightProbability => $composableBuilder(
      column: $table.overweightProbability, builder: (column) => column);

  GeneratedColumn<String> get wastingStatus => $composableBuilder(
      column: $table.wastingStatus, builder: (column) => column);

  GeneratedColumn<double> get muacCm =>
      $composableBuilder(column: $table.muacCm, builder: (column) => column);

  GeneratedColumn<String> get muacStatus => $composableBuilder(
      column: $table.muacStatus, builder: (column) => column);

  GeneratedColumn<String> get muacMethod => $composableBuilder(
      column: $table.muacMethod, builder: (column) => column);

  $$VisitsTableAnnotationComposer get visitId {
    final $$VisitsTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.visitId,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableAnnotationComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$MeasurementsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $MeasurementsTable,
    Measurement,
    $$MeasurementsTableFilterComposer,
    $$MeasurementsTableOrderingComposer,
    $$MeasurementsTableAnnotationComposer,
    $$MeasurementsTableCreateCompanionBuilder,
    $$MeasurementsTableUpdateCompanionBuilder,
    (Measurement, $$MeasurementsTableReferences),
    Measurement,
    PrefetchHooks Function({bool visitId})> {
  $$MeasurementsTableTableManager(_$AppDatabase db, $MeasurementsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$MeasurementsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$MeasurementsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$MeasurementsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> visitId = const Value.absent(),
            Value<double?> predictedHeightCm = const Value.absent(),
            Value<double?> predictedWeightKg = const Value.absent(),
            Value<double?> manualHeightCm = const Value.absent(),
            Value<double?> manualWeightKg = const Value.absent(),
            Value<double?> hazZscore = const Value.absent(),
            Value<double?> whzZscore = const Value.absent(),
            Value<String?> hazStatus = const Value.absent(),
            Value<String?> whzStatus = const Value.absent(),
            Value<double?> confidenceScore = const Value.absent(),
            Value<String?> bodyBuild = const Value.absent(),
            Value<String?> estimationMethod = const Value.absent(),
            Value<bool> sideViewUsed = const Value.absent(),
            Value<double?> chestDepthCm = const Value.absent(),
            Value<double?> abdDepthCm = const Value.absent(),
            Value<double?> mlEstimatedWeightKg = const Value.absent(),
            Value<double?> samProbability = const Value.absent(),
            Value<double?> mamProbability = const Value.absent(),
            Value<double?> normalProbability = const Value.absent(),
            Value<double?> riskOverweightProbability = const Value.absent(),
            Value<double?> overweightProbability = const Value.absent(),
            Value<String?> wastingStatus = const Value.absent(),
            Value<double?> muacCm = const Value.absent(),
            Value<String?> muacStatus = const Value.absent(),
            Value<String?> muacMethod = const Value.absent(),
          }) =>
              MeasurementsCompanion(
            id: id,
            visitId: visitId,
            predictedHeightCm: predictedHeightCm,
            predictedWeightKg: predictedWeightKg,
            manualHeightCm: manualHeightCm,
            manualWeightKg: manualWeightKg,
            hazZscore: hazZscore,
            whzZscore: whzZscore,
            hazStatus: hazStatus,
            whzStatus: whzStatus,
            confidenceScore: confidenceScore,
            bodyBuild: bodyBuild,
            estimationMethod: estimationMethod,
            sideViewUsed: sideViewUsed,
            chestDepthCm: chestDepthCm,
            abdDepthCm: abdDepthCm,
            mlEstimatedWeightKg: mlEstimatedWeightKg,
            samProbability: samProbability,
            mamProbability: mamProbability,
            normalProbability: normalProbability,
            riskOverweightProbability: riskOverweightProbability,
            overweightProbability: overweightProbability,
            wastingStatus: wastingStatus,
            muacCm: muacCm,
            muacStatus: muacStatus,
            muacMethod: muacMethod,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int visitId,
            Value<double?> predictedHeightCm = const Value.absent(),
            Value<double?> predictedWeightKg = const Value.absent(),
            Value<double?> manualHeightCm = const Value.absent(),
            Value<double?> manualWeightKg = const Value.absent(),
            Value<double?> hazZscore = const Value.absent(),
            Value<double?> whzZscore = const Value.absent(),
            Value<String?> hazStatus = const Value.absent(),
            Value<String?> whzStatus = const Value.absent(),
            Value<double?> confidenceScore = const Value.absent(),
            Value<String?> bodyBuild = const Value.absent(),
            Value<String?> estimationMethod = const Value.absent(),
            Value<bool> sideViewUsed = const Value.absent(),
            Value<double?> chestDepthCm = const Value.absent(),
            Value<double?> abdDepthCm = const Value.absent(),
            Value<double?> mlEstimatedWeightKg = const Value.absent(),
            Value<double?> samProbability = const Value.absent(),
            Value<double?> mamProbability = const Value.absent(),
            Value<double?> normalProbability = const Value.absent(),
            Value<double?> riskOverweightProbability = const Value.absent(),
            Value<double?> overweightProbability = const Value.absent(),
            Value<String?> wastingStatus = const Value.absent(),
            Value<double?> muacCm = const Value.absent(),
            Value<String?> muacStatus = const Value.absent(),
            Value<String?> muacMethod = const Value.absent(),
          }) =>
              MeasurementsCompanion.insert(
            id: id,
            visitId: visitId,
            predictedHeightCm: predictedHeightCm,
            predictedWeightKg: predictedWeightKg,
            manualHeightCm: manualHeightCm,
            manualWeightKg: manualWeightKg,
            hazZscore: hazZscore,
            whzZscore: whzZscore,
            hazStatus: hazStatus,
            whzStatus: whzStatus,
            confidenceScore: confidenceScore,
            bodyBuild: bodyBuild,
            estimationMethod: estimationMethod,
            sideViewUsed: sideViewUsed,
            chestDepthCm: chestDepthCm,
            abdDepthCm: abdDepthCm,
            mlEstimatedWeightKg: mlEstimatedWeightKg,
            samProbability: samProbability,
            mamProbability: mamProbability,
            normalProbability: normalProbability,
            riskOverweightProbability: riskOverweightProbability,
            overweightProbability: overweightProbability,
            wastingStatus: wastingStatus,
            muacCm: muacCm,
            muacStatus: muacStatus,
            muacMethod: muacMethod,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$MeasurementsTableReferences(db, table, e)
                  ))
              .toList(),
          prefetchHooksCallback: ({visitId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins: <
                  T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic>>(state) {
                if (visitId) {
                  state = state.withJoin(
                    currentTable: table,
                    currentColumn: table.visitId,
                    referencedTable:
                        $$MeasurementsTableReferences._visitIdTable(db),
                    referencedColumn:
                        $$MeasurementsTableReferences._visitIdTable(db).id,
                  ) as T;
                }

                return state;
              },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ));
}

typedef $$MeasurementsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $MeasurementsTable,
    Measurement,
    $$MeasurementsTableFilterComposer,
    $$MeasurementsTableOrderingComposer,
    $$MeasurementsTableAnnotationComposer,
    $$MeasurementsTableCreateCompanionBuilder,
    $$MeasurementsTableUpdateCompanionBuilder,
    (Measurement, $$MeasurementsTableReferences),
    Measurement,
    PrefetchHooks Function({bool visitId})>;
typedef $$SyncQueueTableCreateCompanionBuilder = SyncQueueCompanion Function({
  Value<int> id,
  required int visitId,
  Value<String> status,
  Value<int> retryCount,
  Value<DateTime> createdAt,
  Value<DateTime?> lastAttemptAt,
  Value<int?> serverVisitId,
  Value<String?> errorMessage,
});
typedef $$SyncQueueTableUpdateCompanionBuilder = SyncQueueCompanion Function({
  Value<int> id,
  Value<int> visitId,
  Value<String> status,
  Value<int> retryCount,
  Value<DateTime> createdAt,
  Value<DateTime?> lastAttemptAt,
  Value<int?> serverVisitId,
  Value<String?> errorMessage,
});

final class $$SyncQueueTableReferences
    extends BaseReferences<_$AppDatabase, $SyncQueueTable, SyncQueueData> {
  $$SyncQueueTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $VisitsTable _visitIdTable(_$AppDatabase db) => db.visits
      .createAlias($_aliasNameGenerator(db.syncQueue.visitId, db.visits.id));

  $$VisitsTableProcessedTableManager get visitId {
    final $_column = $_itemColumn<int>('visit_id')!;

    final manager = $$VisitsTableTableManager($_db, $_db.visits)
        .filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_visitIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: [item]));
  }
}

class $$SyncQueueTableFilterComposer
    extends Composer<_$AppDatabase, $SyncQueueTable> {
  $$SyncQueueTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get status => $composableBuilder(
      column: $table.status, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get lastAttemptAt => $composableBuilder(
      column: $table.lastAttemptAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get serverVisitId => $composableBuilder(
      column: $table.serverVisitId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get errorMessage => $composableBuilder(
      column: $table.errorMessage, builder: (column) => ColumnFilters(column));

  $$VisitsTableFilterComposer get visitId {
    final $$VisitsTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.visitId,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableFilterComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$SyncQueueTableOrderingComposer
    extends Composer<_$AppDatabase, $SyncQueueTable> {
  $$SyncQueueTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get status => $composableBuilder(
      column: $table.status, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get lastAttemptAt => $composableBuilder(
      column: $table.lastAttemptAt,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get serverVisitId => $composableBuilder(
      column: $table.serverVisitId,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get errorMessage => $composableBuilder(
      column: $table.errorMessage,
      builder: (column) => ColumnOrderings(column));

  $$VisitsTableOrderingComposer get visitId {
    final $$VisitsTableOrderingComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.visitId,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableOrderingComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$SyncQueueTableAnnotationComposer
    extends Composer<_$AppDatabase, $SyncQueueTable> {
  $$SyncQueueTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get status =>
      $composableBuilder(column: $table.status, builder: (column) => column);

  GeneratedColumn<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  GeneratedColumn<DateTime> get lastAttemptAt => $composableBuilder(
      column: $table.lastAttemptAt, builder: (column) => column);

  GeneratedColumn<int> get serverVisitId => $composableBuilder(
      column: $table.serverVisitId, builder: (column) => column);

  GeneratedColumn<String> get errorMessage => $composableBuilder(
      column: $table.errorMessage, builder: (column) => column);

  $$VisitsTableAnnotationComposer get visitId {
    final $$VisitsTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.visitId,
        referencedTable: $db.visits,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$VisitsTableAnnotationComposer(
              $db: $db,
              $table: $db.visits,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return composer;
  }
}

class $$SyncQueueTableTableManager extends RootTableManager<
    _$AppDatabase,
    $SyncQueueTable,
    SyncQueueData,
    $$SyncQueueTableFilterComposer,
    $$SyncQueueTableOrderingComposer,
    $$SyncQueueTableAnnotationComposer,
    $$SyncQueueTableCreateCompanionBuilder,
    $$SyncQueueTableUpdateCompanionBuilder,
    (SyncQueueData, $$SyncQueueTableReferences),
    SyncQueueData,
    PrefetchHooks Function({bool visitId})> {
  $$SyncQueueTableTableManager(_$AppDatabase db, $SyncQueueTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$SyncQueueTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$SyncQueueTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$SyncQueueTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> visitId = const Value.absent(),
            Value<String> status = const Value.absent(),
            Value<int> retryCount = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<DateTime?> lastAttemptAt = const Value.absent(),
            Value<int?> serverVisitId = const Value.absent(),
            Value<String?> errorMessage = const Value.absent(),
          }) =>
              SyncQueueCompanion(
            id: id,
            visitId: visitId,
            status: status,
            retryCount: retryCount,
            createdAt: createdAt,
            lastAttemptAt: lastAttemptAt,
            serverVisitId: serverVisitId,
            errorMessage: errorMessage,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int visitId,
            Value<String> status = const Value.absent(),
            Value<int> retryCount = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<DateTime?> lastAttemptAt = const Value.absent(),
            Value<int?> serverVisitId = const Value.absent(),
            Value<String?> errorMessage = const Value.absent(),
          }) =>
              SyncQueueCompanion.insert(
            id: id,
            visitId: visitId,
            status: status,
            retryCount: retryCount,
            createdAt: createdAt,
            lastAttemptAt: lastAttemptAt,
            serverVisitId: serverVisitId,
            errorMessage: errorMessage,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$SyncQueueTableReferences(db, table, e)
                  ))
              .toList(),
          prefetchHooksCallback: ({visitId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins: <
                  T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic>>(state) {
                if (visitId) {
                  state = state.withJoin(
                    currentTable: table,
                    currentColumn: table.visitId,
                    referencedTable:
                        $$SyncQueueTableReferences._visitIdTable(db),
                    referencedColumn:
                        $$SyncQueueTableReferences._visitIdTable(db).id,
                  ) as T;
                }

                return state;
              },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ));
}

typedef $$SyncQueueTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $SyncQueueTable,
    SyncQueueData,
    $$SyncQueueTableFilterComposer,
    $$SyncQueueTableOrderingComposer,
    $$SyncQueueTableAnnotationComposer,
    $$SyncQueueTableCreateCompanionBuilder,
    $$SyncQueueTableUpdateCompanionBuilder,
    (SyncQueueData, $$SyncQueueTableReferences),
    SyncQueueData,
    PrefetchHooks Function({bool visitId})>;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$ChildrenTableTableManager get children =>
      $$ChildrenTableTableManager(_db, _db.children);
  $$VisitsTableTableManager get visits =>
      $$VisitsTableTableManager(_db, _db.visits);
  $$MeasurementsTableTableManager get measurements =>
      $$MeasurementsTableTableManager(_db, _db.measurements);
  $$SyncQueueTableTableManager get syncQueue =>
      $$SyncQueueTableTableManager(_db, _db.syncQueue);
}
