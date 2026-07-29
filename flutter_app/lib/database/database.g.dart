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
  static const VerificationMeta _ownerUserIdMeta =
      const VerificationMeta('ownerUserId');
  @override
  late final GeneratedColumn<int> ownerUserId = GeneratedColumn<int>(
      'owner_user_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _photoPathMeta =
      const VerificationMeta('photoPath');
  @override
  late final GeneratedColumn<String> photoPath = GeneratedColumn<String>(
      'photo_path', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _isArchivedMeta =
      const VerificationMeta('isArchived');
  @override
  late final GeneratedColumn<bool> isArchived = GeneratedColumn<bool>(
      'is_archived', aliasedName, false,
      type: DriftSqlType.bool,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('CHECK ("is_archived" IN (0, 1))'),
      defaultValue: const Constant(false));
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
        ownerUserId,
        photoPath,
        isArchived,
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
    if (data.containsKey('owner_user_id')) {
      context.handle(
          _ownerUserIdMeta,
          ownerUserId.isAcceptableOrUnknown(
              data['owner_user_id']!, _ownerUserIdMeta));
    }
    if (data.containsKey('photo_path')) {
      context.handle(_photoPathMeta,
          photoPath.isAcceptableOrUnknown(data['photo_path']!, _photoPathMeta));
    }
    if (data.containsKey('is_archived')) {
      context.handle(
          _isArchivedMeta,
          isArchived.isAcceptableOrUnknown(
              data['is_archived']!, _isArchivedMeta));
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
      ownerUserId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}owner_user_id']),
      photoPath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}photo_path']),
      isArchived: attachedDatabase.typeMapping
          .read(DriftSqlType.bool, data['${effectivePrefix}is_archived'])!,
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
  final int? ownerUserId;
  final String? photoPath;
  final bool isArchived;
  final DateTime createdAt;
  final DateTime updatedAt;
  const ChildrenData(
      {required this.id,
      required this.name,
      required this.dateOfBirth,
      required this.sex,
      this.guardianName,
      this.location,
      this.ownerUserId,
      this.photoPath,
      required this.isArchived,
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
    if (!nullToAbsent || ownerUserId != null) {
      map['owner_user_id'] = Variable<int>(ownerUserId);
    }
    if (!nullToAbsent || photoPath != null) {
      map['photo_path'] = Variable<String>(photoPath);
    }
    map['is_archived'] = Variable<bool>(isArchived);
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
      ownerUserId: ownerUserId == null && nullToAbsent
          ? const Value.absent()
          : Value(ownerUserId),
      photoPath: photoPath == null && nullToAbsent
          ? const Value.absent()
          : Value(photoPath),
      isArchived: Value(isArchived),
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
      ownerUserId: serializer.fromJson<int?>(json['ownerUserId']),
      photoPath: serializer.fromJson<String?>(json['photoPath']),
      isArchived: serializer.fromJson<bool>(json['isArchived']),
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
      'ownerUserId': serializer.toJson<int?>(ownerUserId),
      'photoPath': serializer.toJson<String?>(photoPath),
      'isArchived': serializer.toJson<bool>(isArchived),
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
          Value<int?> ownerUserId = const Value.absent(),
          Value<String?> photoPath = const Value.absent(),
          bool? isArchived,
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
        ownerUserId: ownerUserId.present ? ownerUserId.value : this.ownerUserId,
        photoPath: photoPath.present ? photoPath.value : this.photoPath,
        isArchived: isArchived ?? this.isArchived,
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
      ownerUserId:
          data.ownerUserId.present ? data.ownerUserId.value : this.ownerUserId,
      photoPath: data.photoPath.present ? data.photoPath.value : this.photoPath,
      isArchived:
          data.isArchived.present ? data.isArchived.value : this.isArchived,
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
          ..write('ownerUserId: $ownerUserId, ')
          ..write('photoPath: $photoPath, ')
          ..write('isArchived: $isArchived, ')
          ..write('createdAt: $createdAt, ')
          ..write('updatedAt: $updatedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, name, dateOfBirth, sex, guardianName,
      location, ownerUserId, photoPath, isArchived, createdAt, updatedAt);
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
          other.ownerUserId == this.ownerUserId &&
          other.photoPath == this.photoPath &&
          other.isArchived == this.isArchived &&
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
  final Value<int?> ownerUserId;
  final Value<String?> photoPath;
  final Value<bool> isArchived;
  final Value<DateTime> createdAt;
  final Value<DateTime> updatedAt;
  const ChildrenCompanion({
    this.id = const Value.absent(),
    this.name = const Value.absent(),
    this.dateOfBirth = const Value.absent(),
    this.sex = const Value.absent(),
    this.guardianName = const Value.absent(),
    this.location = const Value.absent(),
    this.ownerUserId = const Value.absent(),
    this.photoPath = const Value.absent(),
    this.isArchived = const Value.absent(),
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
    this.ownerUserId = const Value.absent(),
    this.photoPath = const Value.absent(),
    this.isArchived = const Value.absent(),
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
    Expression<int>? ownerUserId,
    Expression<String>? photoPath,
    Expression<bool>? isArchived,
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
      if (ownerUserId != null) 'owner_user_id': ownerUserId,
      if (photoPath != null) 'photo_path': photoPath,
      if (isArchived != null) 'is_archived': isArchived,
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
      Value<int?>? ownerUserId,
      Value<String?>? photoPath,
      Value<bool>? isArchived,
      Value<DateTime>? createdAt,
      Value<DateTime>? updatedAt}) {
    return ChildrenCompanion(
      id: id ?? this.id,
      name: name ?? this.name,
      dateOfBirth: dateOfBirth ?? this.dateOfBirth,
      sex: sex ?? this.sex,
      guardianName: guardianName ?? this.guardianName,
      location: location ?? this.location,
      ownerUserId: ownerUserId ?? this.ownerUserId,
      photoPath: photoPath ?? this.photoPath,
      isArchived: isArchived ?? this.isArchived,
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
    if (ownerUserId.present) {
      map['owner_user_id'] = Variable<int>(ownerUserId.value);
    }
    if (photoPath.present) {
      map['photo_path'] = Variable<String>(photoPath.value);
    }
    if (isArchived.present) {
      map['is_archived'] = Variable<bool>(isArchived.value);
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
          ..write('ownerUserId: $ownerUserId, ')
          ..write('photoPath: $photoPath, ')
          ..write('isArchived: $isArchived, ')
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
  static const VerificationMeta _localUuidMeta =
      const VerificationMeta('localUuid');
  @override
  late final GeneratedColumn<String> localUuid = GeneratedColumn<String>(
      'local_uuid', aliasedName, false,
      additionalChecks:
          GeneratedColumn.checkTextLength(minTextLength: 36, maxTextLength: 36),
      type: DriftSqlType.string,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways('UNIQUE'));
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
      'image_path', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
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
  static const VerificationMeta _ownerUserIdMeta =
      const VerificationMeta('ownerUserId');
  @override
  late final GeneratedColumn<int> ownerUserId = GeneratedColumn<int>(
      'owner_user_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _serverIdMeta =
      const VerificationMeta('serverId');
  @override
  late final GeneratedColumn<int> serverId = GeneratedColumn<int>(
      'server_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _entryMethodMeta =
      const VerificationMeta('entryMethod');
  @override
  late final GeneratedColumn<String> entryMethod = GeneratedColumn<String>(
      'entry_method', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant('assessment'));
  static const VerificationMeta _captureStateMeta =
      const VerificationMeta('captureState');
  @override
  late final GeneratedColumn<String> captureState = GeneratedColumn<String>(
      'capture_state', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _captureStartedAtMeta =
      const VerificationMeta('captureStartedAt');
  @override
  late final GeneratedColumn<DateTime> captureStartedAt =
      GeneratedColumn<DateTime>('capture_started_at', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _captureCompletedAtMeta =
      const VerificationMeta('captureCompletedAt');
  @override
  late final GeneratedColumn<DateTime> captureCompletedAt =
      GeneratedColumn<DateTime>('capture_completed_at', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _deviceMetadataJsonMeta =
      const VerificationMeta('deviceMetadataJson');
  @override
  late final GeneratedColumn<String> deviceMetadataJson =
      GeneratedColumn<String>('device_metadata_json', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _consentVersionMeta =
      const VerificationMeta('consentVersion');
  @override
  late final GeneratedColumn<String> consentVersion = GeneratedColumn<String>(
      'consent_version', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _consentTimestampMeta =
      const VerificationMeta('consentTimestamp');
  @override
  late final GeneratedColumn<DateTime> consentTimestamp =
      GeneratedColumn<DateTime>('consent_timestamp', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _consentOperatorIdentifierMeta =
      const VerificationMeta('consentOperatorIdentifier');
  @override
  late final GeneratedColumn<String> consentOperatorIdentifier =
      GeneratedColumn<String>('consent_operator_identifier', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _mediaDeletedAtMeta =
      const VerificationMeta('mediaDeletedAt');
  @override
  late final GeneratedColumn<DateTime> mediaDeletedAt =
      GeneratedColumn<DateTime>('media_deleted_at', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        childId,
        localUuid,
        visitDate,
        ageMonths,
        imagePath,
        sideImagePath,
        backImagePath,
        notes,
        ownerUserId,
        serverId,
        entryMethod,
        captureState,
        captureStartedAt,
        captureCompletedAt,
        deviceMetadataJson,
        consentVersion,
        consentTimestamp,
        consentOperatorIdentifier,
        mediaDeletedAt
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
    if (data.containsKey('local_uuid')) {
      context.handle(_localUuidMeta,
          localUuid.isAcceptableOrUnknown(data['local_uuid']!, _localUuidMeta));
    } else if (isInserting) {
      context.missing(_localUuidMeta);
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
    if (data.containsKey('owner_user_id')) {
      context.handle(
          _ownerUserIdMeta,
          ownerUserId.isAcceptableOrUnknown(
              data['owner_user_id']!, _ownerUserIdMeta));
    }
    if (data.containsKey('server_id')) {
      context.handle(_serverIdMeta,
          serverId.isAcceptableOrUnknown(data['server_id']!, _serverIdMeta));
    }
    if (data.containsKey('entry_method')) {
      context.handle(
          _entryMethodMeta,
          entryMethod.isAcceptableOrUnknown(
              data['entry_method']!, _entryMethodMeta));
    }
    if (data.containsKey('capture_state')) {
      context.handle(
          _captureStateMeta,
          captureState.isAcceptableOrUnknown(
              data['capture_state']!, _captureStateMeta));
    }
    if (data.containsKey('capture_started_at')) {
      context.handle(
          _captureStartedAtMeta,
          captureStartedAt.isAcceptableOrUnknown(
              data['capture_started_at']!, _captureStartedAtMeta));
    }
    if (data.containsKey('capture_completed_at')) {
      context.handle(
          _captureCompletedAtMeta,
          captureCompletedAt.isAcceptableOrUnknown(
              data['capture_completed_at']!, _captureCompletedAtMeta));
    }
    if (data.containsKey('device_metadata_json')) {
      context.handle(
          _deviceMetadataJsonMeta,
          deviceMetadataJson.isAcceptableOrUnknown(
              data['device_metadata_json']!, _deviceMetadataJsonMeta));
    }
    if (data.containsKey('consent_version')) {
      context.handle(
          _consentVersionMeta,
          consentVersion.isAcceptableOrUnknown(
              data['consent_version']!, _consentVersionMeta));
    }
    if (data.containsKey('consent_timestamp')) {
      context.handle(
          _consentTimestampMeta,
          consentTimestamp.isAcceptableOrUnknown(
              data['consent_timestamp']!, _consentTimestampMeta));
    }
    if (data.containsKey('consent_operator_identifier')) {
      context.handle(
          _consentOperatorIdentifierMeta,
          consentOperatorIdentifier.isAcceptableOrUnknown(
              data['consent_operator_identifier']!,
              _consentOperatorIdentifierMeta));
    }
    if (data.containsKey('media_deleted_at')) {
      context.handle(
          _mediaDeletedAtMeta,
          mediaDeletedAt.isAcceptableOrUnknown(
              data['media_deleted_at']!, _mediaDeletedAtMeta));
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
      localUuid: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}local_uuid'])!,
      visitDate: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}visit_date'])!,
      ageMonths: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}age_months'])!,
      imagePath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}image_path']),
      sideImagePath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}side_image_path']),
      backImagePath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}back_image_path']),
      notes: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}notes']),
      ownerUserId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}owner_user_id']),
      serverId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}server_id']),
      entryMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}entry_method'])!,
      captureState: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}capture_state']),
      captureStartedAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime, data['${effectivePrefix}capture_started_at']),
      captureCompletedAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime,
          data['${effectivePrefix}capture_completed_at']),
      deviceMetadataJson: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}device_metadata_json']),
      consentVersion: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}consent_version']),
      consentTimestamp: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime, data['${effectivePrefix}consent_timestamp']),
      consentOperatorIdentifier: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}consent_operator_identifier']),
      mediaDeletedAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime, data['${effectivePrefix}media_deleted_at']),
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
  final String localUuid;
  final DateTime visitDate;
  final double ageMonths;
  final String? imagePath;
  final String? sideImagePath;
  final String? backImagePath;
  final String? notes;
  final int? ownerUserId;
  final int? serverId;
  final String entryMethod;
  final String? captureState;
  final DateTime? captureStartedAt;
  final DateTime? captureCompletedAt;
  final String? deviceMetadataJson;
  final String? consentVersion;
  final DateTime? consentTimestamp;
  final String? consentOperatorIdentifier;
  final DateTime? mediaDeletedAt;
  const Visit(
      {required this.id,
      required this.childId,
      required this.localUuid,
      required this.visitDate,
      required this.ageMonths,
      this.imagePath,
      this.sideImagePath,
      this.backImagePath,
      this.notes,
      this.ownerUserId,
      this.serverId,
      required this.entryMethod,
      this.captureState,
      this.captureStartedAt,
      this.captureCompletedAt,
      this.deviceMetadataJson,
      this.consentVersion,
      this.consentTimestamp,
      this.consentOperatorIdentifier,
      this.mediaDeletedAt});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['child_id'] = Variable<int>(childId);
    map['local_uuid'] = Variable<String>(localUuid);
    map['visit_date'] = Variable<DateTime>(visitDate);
    map['age_months'] = Variable<double>(ageMonths);
    if (!nullToAbsent || imagePath != null) {
      map['image_path'] = Variable<String>(imagePath);
    }
    if (!nullToAbsent || sideImagePath != null) {
      map['side_image_path'] = Variable<String>(sideImagePath);
    }
    if (!nullToAbsent || backImagePath != null) {
      map['back_image_path'] = Variable<String>(backImagePath);
    }
    if (!nullToAbsent || notes != null) {
      map['notes'] = Variable<String>(notes);
    }
    if (!nullToAbsent || ownerUserId != null) {
      map['owner_user_id'] = Variable<int>(ownerUserId);
    }
    if (!nullToAbsent || serverId != null) {
      map['server_id'] = Variable<int>(serverId);
    }
    map['entry_method'] = Variable<String>(entryMethod);
    if (!nullToAbsent || captureState != null) {
      map['capture_state'] = Variable<String>(captureState);
    }
    if (!nullToAbsent || captureStartedAt != null) {
      map['capture_started_at'] = Variable<DateTime>(captureStartedAt);
    }
    if (!nullToAbsent || captureCompletedAt != null) {
      map['capture_completed_at'] = Variable<DateTime>(captureCompletedAt);
    }
    if (!nullToAbsent || deviceMetadataJson != null) {
      map['device_metadata_json'] = Variable<String>(deviceMetadataJson);
    }
    if (!nullToAbsent || consentVersion != null) {
      map['consent_version'] = Variable<String>(consentVersion);
    }
    if (!nullToAbsent || consentTimestamp != null) {
      map['consent_timestamp'] = Variable<DateTime>(consentTimestamp);
    }
    if (!nullToAbsent || consentOperatorIdentifier != null) {
      map['consent_operator_identifier'] =
          Variable<String>(consentOperatorIdentifier);
    }
    if (!nullToAbsent || mediaDeletedAt != null) {
      map['media_deleted_at'] = Variable<DateTime>(mediaDeletedAt);
    }
    return map;
  }

  VisitsCompanion toCompanion(bool nullToAbsent) {
    return VisitsCompanion(
      id: Value(id),
      childId: Value(childId),
      localUuid: Value(localUuid),
      visitDate: Value(visitDate),
      ageMonths: Value(ageMonths),
      imagePath: imagePath == null && nullToAbsent
          ? const Value.absent()
          : Value(imagePath),
      sideImagePath: sideImagePath == null && nullToAbsent
          ? const Value.absent()
          : Value(sideImagePath),
      backImagePath: backImagePath == null && nullToAbsent
          ? const Value.absent()
          : Value(backImagePath),
      notes:
          notes == null && nullToAbsent ? const Value.absent() : Value(notes),
      ownerUserId: ownerUserId == null && nullToAbsent
          ? const Value.absent()
          : Value(ownerUserId),
      serverId: serverId == null && nullToAbsent
          ? const Value.absent()
          : Value(serverId),
      entryMethod: Value(entryMethod),
      captureState: captureState == null && nullToAbsent
          ? const Value.absent()
          : Value(captureState),
      captureStartedAt: captureStartedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(captureStartedAt),
      captureCompletedAt: captureCompletedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(captureCompletedAt),
      deviceMetadataJson: deviceMetadataJson == null && nullToAbsent
          ? const Value.absent()
          : Value(deviceMetadataJson),
      consentVersion: consentVersion == null && nullToAbsent
          ? const Value.absent()
          : Value(consentVersion),
      consentTimestamp: consentTimestamp == null && nullToAbsent
          ? const Value.absent()
          : Value(consentTimestamp),
      consentOperatorIdentifier:
          consentOperatorIdentifier == null && nullToAbsent
              ? const Value.absent()
              : Value(consentOperatorIdentifier),
      mediaDeletedAt: mediaDeletedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(mediaDeletedAt),
    );
  }

  factory Visit.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Visit(
      id: serializer.fromJson<int>(json['id']),
      childId: serializer.fromJson<int>(json['childId']),
      localUuid: serializer.fromJson<String>(json['localUuid']),
      visitDate: serializer.fromJson<DateTime>(json['visitDate']),
      ageMonths: serializer.fromJson<double>(json['ageMonths']),
      imagePath: serializer.fromJson<String?>(json['imagePath']),
      sideImagePath: serializer.fromJson<String?>(json['sideImagePath']),
      backImagePath: serializer.fromJson<String?>(json['backImagePath']),
      notes: serializer.fromJson<String?>(json['notes']),
      ownerUserId: serializer.fromJson<int?>(json['ownerUserId']),
      serverId: serializer.fromJson<int?>(json['serverId']),
      entryMethod: serializer.fromJson<String>(json['entryMethod']),
      captureState: serializer.fromJson<String?>(json['captureState']),
      captureStartedAt:
          serializer.fromJson<DateTime?>(json['captureStartedAt']),
      captureCompletedAt:
          serializer.fromJson<DateTime?>(json['captureCompletedAt']),
      deviceMetadataJson:
          serializer.fromJson<String?>(json['deviceMetadataJson']),
      consentVersion: serializer.fromJson<String?>(json['consentVersion']),
      consentTimestamp:
          serializer.fromJson<DateTime?>(json['consentTimestamp']),
      consentOperatorIdentifier:
          serializer.fromJson<String?>(json['consentOperatorIdentifier']),
      mediaDeletedAt: serializer.fromJson<DateTime?>(json['mediaDeletedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'childId': serializer.toJson<int>(childId),
      'localUuid': serializer.toJson<String>(localUuid),
      'visitDate': serializer.toJson<DateTime>(visitDate),
      'ageMonths': serializer.toJson<double>(ageMonths),
      'imagePath': serializer.toJson<String?>(imagePath),
      'sideImagePath': serializer.toJson<String?>(sideImagePath),
      'backImagePath': serializer.toJson<String?>(backImagePath),
      'notes': serializer.toJson<String?>(notes),
      'ownerUserId': serializer.toJson<int?>(ownerUserId),
      'serverId': serializer.toJson<int?>(serverId),
      'entryMethod': serializer.toJson<String>(entryMethod),
      'captureState': serializer.toJson<String?>(captureState),
      'captureStartedAt': serializer.toJson<DateTime?>(captureStartedAt),
      'captureCompletedAt': serializer.toJson<DateTime?>(captureCompletedAt),
      'deviceMetadataJson': serializer.toJson<String?>(deviceMetadataJson),
      'consentVersion': serializer.toJson<String?>(consentVersion),
      'consentTimestamp': serializer.toJson<DateTime?>(consentTimestamp),
      'consentOperatorIdentifier':
          serializer.toJson<String?>(consentOperatorIdentifier),
      'mediaDeletedAt': serializer.toJson<DateTime?>(mediaDeletedAt),
    };
  }

  Visit copyWith(
          {int? id,
          int? childId,
          String? localUuid,
          DateTime? visitDate,
          double? ageMonths,
          Value<String?> imagePath = const Value.absent(),
          Value<String?> sideImagePath = const Value.absent(),
          Value<String?> backImagePath = const Value.absent(),
          Value<String?> notes = const Value.absent(),
          Value<int?> ownerUserId = const Value.absent(),
          Value<int?> serverId = const Value.absent(),
          String? entryMethod,
          Value<String?> captureState = const Value.absent(),
          Value<DateTime?> captureStartedAt = const Value.absent(),
          Value<DateTime?> captureCompletedAt = const Value.absent(),
          Value<String?> deviceMetadataJson = const Value.absent(),
          Value<String?> consentVersion = const Value.absent(),
          Value<DateTime?> consentTimestamp = const Value.absent(),
          Value<String?> consentOperatorIdentifier = const Value.absent(),
          Value<DateTime?> mediaDeletedAt = const Value.absent()}) =>
      Visit(
        id: id ?? this.id,
        childId: childId ?? this.childId,
        localUuid: localUuid ?? this.localUuid,
        visitDate: visitDate ?? this.visitDate,
        ageMonths: ageMonths ?? this.ageMonths,
        imagePath: imagePath.present ? imagePath.value : this.imagePath,
        sideImagePath:
            sideImagePath.present ? sideImagePath.value : this.sideImagePath,
        backImagePath:
            backImagePath.present ? backImagePath.value : this.backImagePath,
        notes: notes.present ? notes.value : this.notes,
        ownerUserId: ownerUserId.present ? ownerUserId.value : this.ownerUserId,
        serverId: serverId.present ? serverId.value : this.serverId,
        entryMethod: entryMethod ?? this.entryMethod,
        captureState:
            captureState.present ? captureState.value : this.captureState,
        captureStartedAt: captureStartedAt.present
            ? captureStartedAt.value
            : this.captureStartedAt,
        captureCompletedAt: captureCompletedAt.present
            ? captureCompletedAt.value
            : this.captureCompletedAt,
        deviceMetadataJson: deviceMetadataJson.present
            ? deviceMetadataJson.value
            : this.deviceMetadataJson,
        consentVersion:
            consentVersion.present ? consentVersion.value : this.consentVersion,
        consentTimestamp: consentTimestamp.present
            ? consentTimestamp.value
            : this.consentTimestamp,
        consentOperatorIdentifier: consentOperatorIdentifier.present
            ? consentOperatorIdentifier.value
            : this.consentOperatorIdentifier,
        mediaDeletedAt:
            mediaDeletedAt.present ? mediaDeletedAt.value : this.mediaDeletedAt,
      );
  Visit copyWithCompanion(VisitsCompanion data) {
    return Visit(
      id: data.id.present ? data.id.value : this.id,
      childId: data.childId.present ? data.childId.value : this.childId,
      localUuid: data.localUuid.present ? data.localUuid.value : this.localUuid,
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
      ownerUserId:
          data.ownerUserId.present ? data.ownerUserId.value : this.ownerUserId,
      serverId: data.serverId.present ? data.serverId.value : this.serverId,
      entryMethod:
          data.entryMethod.present ? data.entryMethod.value : this.entryMethod,
      captureState: data.captureState.present
          ? data.captureState.value
          : this.captureState,
      captureStartedAt: data.captureStartedAt.present
          ? data.captureStartedAt.value
          : this.captureStartedAt,
      captureCompletedAt: data.captureCompletedAt.present
          ? data.captureCompletedAt.value
          : this.captureCompletedAt,
      deviceMetadataJson: data.deviceMetadataJson.present
          ? data.deviceMetadataJson.value
          : this.deviceMetadataJson,
      consentVersion: data.consentVersion.present
          ? data.consentVersion.value
          : this.consentVersion,
      consentTimestamp: data.consentTimestamp.present
          ? data.consentTimestamp.value
          : this.consentTimestamp,
      consentOperatorIdentifier: data.consentOperatorIdentifier.present
          ? data.consentOperatorIdentifier.value
          : this.consentOperatorIdentifier,
      mediaDeletedAt: data.mediaDeletedAt.present
          ? data.mediaDeletedAt.value
          : this.mediaDeletedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Visit(')
          ..write('id: $id, ')
          ..write('childId: $childId, ')
          ..write('localUuid: $localUuid, ')
          ..write('visitDate: $visitDate, ')
          ..write('ageMonths: $ageMonths, ')
          ..write('imagePath: $imagePath, ')
          ..write('sideImagePath: $sideImagePath, ')
          ..write('backImagePath: $backImagePath, ')
          ..write('notes: $notes, ')
          ..write('ownerUserId: $ownerUserId, ')
          ..write('serverId: $serverId, ')
          ..write('entryMethod: $entryMethod, ')
          ..write('captureState: $captureState, ')
          ..write('captureStartedAt: $captureStartedAt, ')
          ..write('captureCompletedAt: $captureCompletedAt, ')
          ..write('deviceMetadataJson: $deviceMetadataJson, ')
          ..write('consentVersion: $consentVersion, ')
          ..write('consentTimestamp: $consentTimestamp, ')
          ..write('consentOperatorIdentifier: $consentOperatorIdentifier, ')
          ..write('mediaDeletedAt: $mediaDeletedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
      id,
      childId,
      localUuid,
      visitDate,
      ageMonths,
      imagePath,
      sideImagePath,
      backImagePath,
      notes,
      ownerUserId,
      serverId,
      entryMethod,
      captureState,
      captureStartedAt,
      captureCompletedAt,
      deviceMetadataJson,
      consentVersion,
      consentTimestamp,
      consentOperatorIdentifier,
      mediaDeletedAt);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Visit &&
          other.id == this.id &&
          other.childId == this.childId &&
          other.localUuid == this.localUuid &&
          other.visitDate == this.visitDate &&
          other.ageMonths == this.ageMonths &&
          other.imagePath == this.imagePath &&
          other.sideImagePath == this.sideImagePath &&
          other.backImagePath == this.backImagePath &&
          other.notes == this.notes &&
          other.ownerUserId == this.ownerUserId &&
          other.serverId == this.serverId &&
          other.entryMethod == this.entryMethod &&
          other.captureState == this.captureState &&
          other.captureStartedAt == this.captureStartedAt &&
          other.captureCompletedAt == this.captureCompletedAt &&
          other.deviceMetadataJson == this.deviceMetadataJson &&
          other.consentVersion == this.consentVersion &&
          other.consentTimestamp == this.consentTimestamp &&
          other.consentOperatorIdentifier == this.consentOperatorIdentifier &&
          other.mediaDeletedAt == this.mediaDeletedAt);
}

class VisitsCompanion extends UpdateCompanion<Visit> {
  final Value<int> id;
  final Value<int> childId;
  final Value<String> localUuid;
  final Value<DateTime> visitDate;
  final Value<double> ageMonths;
  final Value<String?> imagePath;
  final Value<String?> sideImagePath;
  final Value<String?> backImagePath;
  final Value<String?> notes;
  final Value<int?> ownerUserId;
  final Value<int?> serverId;
  final Value<String> entryMethod;
  final Value<String?> captureState;
  final Value<DateTime?> captureStartedAt;
  final Value<DateTime?> captureCompletedAt;
  final Value<String?> deviceMetadataJson;
  final Value<String?> consentVersion;
  final Value<DateTime?> consentTimestamp;
  final Value<String?> consentOperatorIdentifier;
  final Value<DateTime?> mediaDeletedAt;
  const VisitsCompanion({
    this.id = const Value.absent(),
    this.childId = const Value.absent(),
    this.localUuid = const Value.absent(),
    this.visitDate = const Value.absent(),
    this.ageMonths = const Value.absent(),
    this.imagePath = const Value.absent(),
    this.sideImagePath = const Value.absent(),
    this.backImagePath = const Value.absent(),
    this.notes = const Value.absent(),
    this.ownerUserId = const Value.absent(),
    this.serverId = const Value.absent(),
    this.entryMethod = const Value.absent(),
    this.captureState = const Value.absent(),
    this.captureStartedAt = const Value.absent(),
    this.captureCompletedAt = const Value.absent(),
    this.deviceMetadataJson = const Value.absent(),
    this.consentVersion = const Value.absent(),
    this.consentTimestamp = const Value.absent(),
    this.consentOperatorIdentifier = const Value.absent(),
    this.mediaDeletedAt = const Value.absent(),
  });
  VisitsCompanion.insert({
    this.id = const Value.absent(),
    required int childId,
    required String localUuid,
    this.visitDate = const Value.absent(),
    required double ageMonths,
    this.imagePath = const Value.absent(),
    this.sideImagePath = const Value.absent(),
    this.backImagePath = const Value.absent(),
    this.notes = const Value.absent(),
    this.ownerUserId = const Value.absent(),
    this.serverId = const Value.absent(),
    this.entryMethod = const Value.absent(),
    this.captureState = const Value.absent(),
    this.captureStartedAt = const Value.absent(),
    this.captureCompletedAt = const Value.absent(),
    this.deviceMetadataJson = const Value.absent(),
    this.consentVersion = const Value.absent(),
    this.consentTimestamp = const Value.absent(),
    this.consentOperatorIdentifier = const Value.absent(),
    this.mediaDeletedAt = const Value.absent(),
  })  : childId = Value(childId),
        localUuid = Value(localUuid),
        ageMonths = Value(ageMonths);
  static Insertable<Visit> custom({
    Expression<int>? id,
    Expression<int>? childId,
    Expression<String>? localUuid,
    Expression<DateTime>? visitDate,
    Expression<double>? ageMonths,
    Expression<String>? imagePath,
    Expression<String>? sideImagePath,
    Expression<String>? backImagePath,
    Expression<String>? notes,
    Expression<int>? ownerUserId,
    Expression<int>? serverId,
    Expression<String>? entryMethod,
    Expression<String>? captureState,
    Expression<DateTime>? captureStartedAt,
    Expression<DateTime>? captureCompletedAt,
    Expression<String>? deviceMetadataJson,
    Expression<String>? consentVersion,
    Expression<DateTime>? consentTimestamp,
    Expression<String>? consentOperatorIdentifier,
    Expression<DateTime>? mediaDeletedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (childId != null) 'child_id': childId,
      if (localUuid != null) 'local_uuid': localUuid,
      if (visitDate != null) 'visit_date': visitDate,
      if (ageMonths != null) 'age_months': ageMonths,
      if (imagePath != null) 'image_path': imagePath,
      if (sideImagePath != null) 'side_image_path': sideImagePath,
      if (backImagePath != null) 'back_image_path': backImagePath,
      if (notes != null) 'notes': notes,
      if (ownerUserId != null) 'owner_user_id': ownerUserId,
      if (serverId != null) 'server_id': serverId,
      if (entryMethod != null) 'entry_method': entryMethod,
      if (captureState != null) 'capture_state': captureState,
      if (captureStartedAt != null) 'capture_started_at': captureStartedAt,
      if (captureCompletedAt != null)
        'capture_completed_at': captureCompletedAt,
      if (deviceMetadataJson != null)
        'device_metadata_json': deviceMetadataJson,
      if (consentVersion != null) 'consent_version': consentVersion,
      if (consentTimestamp != null) 'consent_timestamp': consentTimestamp,
      if (consentOperatorIdentifier != null)
        'consent_operator_identifier': consentOperatorIdentifier,
      if (mediaDeletedAt != null) 'media_deleted_at': mediaDeletedAt,
    });
  }

  VisitsCompanion copyWith(
      {Value<int>? id,
      Value<int>? childId,
      Value<String>? localUuid,
      Value<DateTime>? visitDate,
      Value<double>? ageMonths,
      Value<String?>? imagePath,
      Value<String?>? sideImagePath,
      Value<String?>? backImagePath,
      Value<String?>? notes,
      Value<int?>? ownerUserId,
      Value<int?>? serverId,
      Value<String>? entryMethod,
      Value<String?>? captureState,
      Value<DateTime?>? captureStartedAt,
      Value<DateTime?>? captureCompletedAt,
      Value<String?>? deviceMetadataJson,
      Value<String?>? consentVersion,
      Value<DateTime?>? consentTimestamp,
      Value<String?>? consentOperatorIdentifier,
      Value<DateTime?>? mediaDeletedAt}) {
    return VisitsCompanion(
      id: id ?? this.id,
      childId: childId ?? this.childId,
      localUuid: localUuid ?? this.localUuid,
      visitDate: visitDate ?? this.visitDate,
      ageMonths: ageMonths ?? this.ageMonths,
      imagePath: imagePath ?? this.imagePath,
      sideImagePath: sideImagePath ?? this.sideImagePath,
      backImagePath: backImagePath ?? this.backImagePath,
      notes: notes ?? this.notes,
      ownerUserId: ownerUserId ?? this.ownerUserId,
      serverId: serverId ?? this.serverId,
      entryMethod: entryMethod ?? this.entryMethod,
      captureState: captureState ?? this.captureState,
      captureStartedAt: captureStartedAt ?? this.captureStartedAt,
      captureCompletedAt: captureCompletedAt ?? this.captureCompletedAt,
      deviceMetadataJson: deviceMetadataJson ?? this.deviceMetadataJson,
      consentVersion: consentVersion ?? this.consentVersion,
      consentTimestamp: consentTimestamp ?? this.consentTimestamp,
      consentOperatorIdentifier:
          consentOperatorIdentifier ?? this.consentOperatorIdentifier,
      mediaDeletedAt: mediaDeletedAt ?? this.mediaDeletedAt,
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
    if (localUuid.present) {
      map['local_uuid'] = Variable<String>(localUuid.value);
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
    if (ownerUserId.present) {
      map['owner_user_id'] = Variable<int>(ownerUserId.value);
    }
    if (serverId.present) {
      map['server_id'] = Variable<int>(serverId.value);
    }
    if (entryMethod.present) {
      map['entry_method'] = Variable<String>(entryMethod.value);
    }
    if (captureState.present) {
      map['capture_state'] = Variable<String>(captureState.value);
    }
    if (captureStartedAt.present) {
      map['capture_started_at'] = Variable<DateTime>(captureStartedAt.value);
    }
    if (captureCompletedAt.present) {
      map['capture_completed_at'] =
          Variable<DateTime>(captureCompletedAt.value);
    }
    if (deviceMetadataJson.present) {
      map['device_metadata_json'] = Variable<String>(deviceMetadataJson.value);
    }
    if (consentVersion.present) {
      map['consent_version'] = Variable<String>(consentVersion.value);
    }
    if (consentTimestamp.present) {
      map['consent_timestamp'] = Variable<DateTime>(consentTimestamp.value);
    }
    if (consentOperatorIdentifier.present) {
      map['consent_operator_identifier'] =
          Variable<String>(consentOperatorIdentifier.value);
    }
    if (mediaDeletedAt.present) {
      map['media_deleted_at'] = Variable<DateTime>(mediaDeletedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('VisitsCompanion(')
          ..write('id: $id, ')
          ..write('childId: $childId, ')
          ..write('localUuid: $localUuid, ')
          ..write('visitDate: $visitDate, ')
          ..write('ageMonths: $ageMonths, ')
          ..write('imagePath: $imagePath, ')
          ..write('sideImagePath: $sideImagePath, ')
          ..write('backImagePath: $backImagePath, ')
          ..write('notes: $notes, ')
          ..write('ownerUserId: $ownerUserId, ')
          ..write('serverId: $serverId, ')
          ..write('entryMethod: $entryMethod, ')
          ..write('captureState: $captureState, ')
          ..write('captureStartedAt: $captureStartedAt, ')
          ..write('captureCompletedAt: $captureCompletedAt, ')
          ..write('deviceMetadataJson: $deviceMetadataJson, ')
          ..write('consentVersion: $consentVersion, ')
          ..write('consentTimestamp: $consentTimestamp, ')
          ..write('consentOperatorIdentifier: $consentOperatorIdentifier, ')
          ..write('mediaDeletedAt: $mediaDeletedAt')
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
  static const VerificationMeta _effectiveHeightCmMeta =
      const VerificationMeta('effectiveHeightCm');
  @override
  late final GeneratedColumn<double> effectiveHeightCm =
      GeneratedColumn<double>('effective_height_cm', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _effectiveWeightKgMeta =
      const VerificationMeta('effectiveWeightKg');
  @override
  late final GeneratedColumn<double> effectiveWeightKg =
      GeneratedColumn<double>('effective_weight_kg', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _heightMethodMeta =
      const VerificationMeta('heightMethod');
  @override
  late final GeneratedColumn<String> heightMethod = GeneratedColumn<String>(
      'height_method', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _weightMethodMeta =
      const VerificationMeta('weightMethod');
  @override
  late final GeneratedColumn<String> weightMethod = GeneratedColumn<String>(
      'weight_method', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _bmiMeta = const VerificationMeta('bmi');
  @override
  late final GeneratedColumn<double> bmi = GeneratedColumn<double>(
      'bmi', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _bmiStatusMeta =
      const VerificationMeta('bmiStatus');
  @override
  late final GeneratedColumn<String> bmiStatus = GeneratedColumn<String>(
      'bmi_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
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
  static const VerificationMeta _heightConfidenceMeta =
      const VerificationMeta('heightConfidence');
  @override
  late final GeneratedColumn<double> heightConfidence = GeneratedColumn<double>(
      'height_confidence', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _weightConfidenceMeta =
      const VerificationMeta('weightConfidence');
  @override
  late final GeneratedColumn<double> weightConfidence = GeneratedColumn<double>(
      'weight_confidence', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _classificationConfidenceMeta =
      const VerificationMeta('classificationConfidence');
  @override
  late final GeneratedColumn<double> classificationConfidence =
      GeneratedColumn<double>('classification_confidence', aliasedName, true,
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
  static const VerificationMeta _wastingMethodMeta =
      const VerificationMeta('wastingMethod');
  @override
  late final GeneratedColumn<String> wastingMethod = GeneratedColumn<String>(
      'wasting_method', aliasedName, true,
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
  static const VerificationMeta _muacAgeInRangeMeta =
      const VerificationMeta('muacAgeInRange');
  @override
  late final GeneratedColumn<bool> muacAgeInRange = GeneratedColumn<bool>(
      'muac_age_in_range', aliasedName, true,
      type: DriftSqlType.bool,
      requiredDuringInsert: false,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'CHECK ("muac_age_in_range" IN (0, 1))'));
  static const VerificationMeta _muacConfidenceMeta =
      const VerificationMeta('muacConfidence');
  @override
  late final GeneratedColumn<double> muacConfidence = GeneratedColumn<double>(
      'muac_confidence', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _muacUncertaintyLowerCmMeta =
      const VerificationMeta('muacUncertaintyLowerCm');
  @override
  late final GeneratedColumn<double> muacUncertaintyLowerCm =
      GeneratedColumn<double>('muac_uncertainty_lower_cm', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _muacUncertaintyUpperCmMeta =
      const VerificationMeta('muacUncertaintyUpperCm');
  @override
  late final GeneratedColumn<double> muacUncertaintyUpperCm =
      GeneratedColumn<double>('muac_uncertainty_upper_cm', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _muacModelVersionMeta =
      const VerificationMeta('muacModelVersion');
  @override
  late final GeneratedColumn<String> muacModelVersion = GeneratedColumn<String>(
      'muac_model_version', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _muacCalibrationVersionMeta =
      const VerificationMeta('muacCalibrationVersion');
  @override
  late final GeneratedColumn<String> muacCalibrationVersion =
      GeneratedColumn<String>('muac_calibration_version', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _muacIsDirectMeasurementMeta =
      const VerificationMeta('muacIsDirectMeasurement');
  @override
  late final GeneratedColumn<bool> muacIsDirectMeasurement =
      GeneratedColumn<bool>('muac_is_direct_measurement', aliasedName, true,
          type: DriftSqlType.bool,
          requiredDuringInsert: false,
          defaultConstraints: GeneratedColumn.constraintIsAlways(
              'CHECK ("muac_is_direct_measurement" IN (0, 1))'));
  static const VerificationMeta _muacRequiresConfirmationMeta =
      const VerificationMeta('muacRequiresConfirmation');
  @override
  late final GeneratedColumn<bool> muacRequiresConfirmation =
      GeneratedColumn<bool>(
          'muac_requires_confirmation', aliasedName, true,
          type: DriftSqlType.bool,
          requiredDuringInsert: false,
          defaultConstraints: GeneratedColumn.constraintIsAlways(
              'CHECK ("muac_requires_confirmation" IN (0, 1))'));
  static const VerificationMeta _muacReferralGuidanceMeta =
      const VerificationMeta('muacReferralGuidance');
  @override
  late final GeneratedColumn<String> muacReferralGuidance =
      GeneratedColumn<String>('muac_referral_guidance', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _combinedStatusMeta =
      const VerificationMeta('combinedStatus');
  @override
  late final GeneratedColumn<String> combinedStatus = GeneratedColumn<String>(
      'combined_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _combinedTriggeredByMeta =
      const VerificationMeta('combinedTriggeredBy');
  @override
  late final GeneratedColumn<String> combinedTriggeredBy =
      GeneratedColumn<String>('combined_triggered_by', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _combinedRationaleMeta =
      const VerificationMeta('combinedRationale');
  @override
  late final GeneratedColumn<String> combinedRationale =
      GeneratedColumn<String>('combined_rationale', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _combinedMethodMeta =
      const VerificationMeta('combinedMethod');
  @override
  late final GeneratedColumn<String> combinedMethod = GeneratedColumn<String>(
      'combined_method', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _combinedConfidenceScoreMeta =
      const VerificationMeta('combinedConfidenceScore');
  @override
  late final GeneratedColumn<double> combinedConfidenceScore =
      GeneratedColumn<double>('combined_confidence_score', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _combinedProtocolVersionMeta =
      const VerificationMeta('combinedProtocolVersion');
  @override
  late final GeneratedColumn<String> combinedProtocolVersion =
      GeneratedColumn<String>('combined_protocol_version', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _poshanStatusMeta =
      const VerificationMeta('poshanStatus');
  @override
  late final GeneratedColumn<String> poshanStatus = GeneratedColumn<String>(
      'poshan_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _poshanTriggeredByMeta =
      const VerificationMeta('poshanTriggeredBy');
  @override
  late final GeneratedColumn<String> poshanTriggeredBy =
      GeneratedColumn<String>('poshan_triggered_by', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _classificationMethodMeta =
      const VerificationMeta('classificationMethod');
  @override
  late final GeneratedColumn<String> classificationMethod =
      GeneratedColumn<String>('classification_method', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _classificationRationaleMeta =
      const VerificationMeta('classificationRationale');
  @override
  late final GeneratedColumn<String> classificationRationale =
      GeneratedColumn<String>('classification_rationale', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _poshanCompleteMeta =
      const VerificationMeta('poshanComplete');
  @override
  late final GeneratedColumn<bool> poshanComplete = GeneratedColumn<bool>(
      'poshan_complete', aliasedName, true,
      type: DriftSqlType.bool,
      requiredDuringInsert: false,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'CHECK ("poshan_complete" IN (0, 1))'));
  static const VerificationMeta _measurementModeMeta =
      const VerificationMeta('measurementMode');
  @override
  late final GeneratedColumn<String> measurementMode = GeneratedColumn<String>(
      'measurement_mode', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _oedemaMeta = const VerificationMeta('oedema');
  @override
  late final GeneratedColumn<String> oedema = GeneratedColumn<String>(
      'oedema', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _measuredAtMeta =
      const VerificationMeta('measuredAt');
  @override
  late final GeneratedColumn<DateTime> measuredAt = GeneratedColumn<DateTime>(
      'measured_at', aliasedName, true,
      type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _editorUserIdMeta =
      const VerificationMeta('editorUserId');
  @override
  late final GeneratedColumn<int> editorUserId = GeneratedColumn<int>(
      'editor_user_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _measuredNotesMeta =
      const VerificationMeta('measuredNotes');
  @override
  late final GeneratedColumn<String> measuredNotes = GeneratedColumn<String>(
      'measured_notes', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _whoAcuteStatusMeta =
      const VerificationMeta('whoAcuteStatus');
  @override
  late final GeneratedColumn<String> whoAcuteStatus = GeneratedColumn<String>(
      'who_acute_status', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _whoAcuteTriggeredByMeta =
      const VerificationMeta('whoAcuteTriggeredBy');
  @override
  late final GeneratedColumn<String> whoAcuteTriggeredBy =
      GeneratedColumn<String>('who_acute_triggered_by', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _whoAcuteRationaleMeta =
      const VerificationMeta('whoAcuteRationale');
  @override
  late final GeneratedColumn<String> whoAcuteRationale =
      GeneratedColumn<String>('who_acute_rationale', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        visitId,
        predictedHeightCm,
        predictedWeightKg,
        manualHeightCm,
        manualWeightKg,
        effectiveHeightCm,
        effectiveWeightKg,
        heightMethod,
        weightMethod,
        bmi,
        bmiStatus,
        hazZscore,
        whzZscore,
        hazStatus,
        whzStatus,
        confidenceScore,
        heightConfidence,
        weightConfidence,
        classificationConfidence,
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
        wastingMethod,
        muacCm,
        muacStatus,
        muacMethod,
        muacAgeInRange,
        muacConfidence,
        muacUncertaintyLowerCm,
        muacUncertaintyUpperCm,
        muacModelVersion,
        muacCalibrationVersion,
        muacIsDirectMeasurement,
        muacRequiresConfirmation,
        muacReferralGuidance,
        combinedStatus,
        combinedTriggeredBy,
        combinedRationale,
        combinedMethod,
        combinedConfidenceScore,
        combinedProtocolVersion,
        poshanStatus,
        poshanTriggeredBy,
        classificationMethod,
        classificationRationale,
        poshanComplete,
        measurementMode,
        oedema,
        measuredAt,
        editorUserId,
        measuredNotes,
        whoAcuteStatus,
        whoAcuteTriggeredBy,
        whoAcuteRationale
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
    if (data.containsKey('effective_height_cm')) {
      context.handle(
          _effectiveHeightCmMeta,
          effectiveHeightCm.isAcceptableOrUnknown(
              data['effective_height_cm']!, _effectiveHeightCmMeta));
    }
    if (data.containsKey('effective_weight_kg')) {
      context.handle(
          _effectiveWeightKgMeta,
          effectiveWeightKg.isAcceptableOrUnknown(
              data['effective_weight_kg']!, _effectiveWeightKgMeta));
    }
    if (data.containsKey('height_method')) {
      context.handle(
          _heightMethodMeta,
          heightMethod.isAcceptableOrUnknown(
              data['height_method']!, _heightMethodMeta));
    }
    if (data.containsKey('weight_method')) {
      context.handle(
          _weightMethodMeta,
          weightMethod.isAcceptableOrUnknown(
              data['weight_method']!, _weightMethodMeta));
    }
    if (data.containsKey('bmi')) {
      context.handle(
          _bmiMeta, bmi.isAcceptableOrUnknown(data['bmi']!, _bmiMeta));
    }
    if (data.containsKey('bmi_status')) {
      context.handle(_bmiStatusMeta,
          bmiStatus.isAcceptableOrUnknown(data['bmi_status']!, _bmiStatusMeta));
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
    if (data.containsKey('height_confidence')) {
      context.handle(
          _heightConfidenceMeta,
          heightConfidence.isAcceptableOrUnknown(
              data['height_confidence']!, _heightConfidenceMeta));
    }
    if (data.containsKey('weight_confidence')) {
      context.handle(
          _weightConfidenceMeta,
          weightConfidence.isAcceptableOrUnknown(
              data['weight_confidence']!, _weightConfidenceMeta));
    }
    if (data.containsKey('classification_confidence')) {
      context.handle(
          _classificationConfidenceMeta,
          classificationConfidence.isAcceptableOrUnknown(
              data['classification_confidence']!,
              _classificationConfidenceMeta));
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
    if (data.containsKey('wasting_method')) {
      context.handle(
          _wastingMethodMeta,
          wastingMethod.isAcceptableOrUnknown(
              data['wasting_method']!, _wastingMethodMeta));
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
    if (data.containsKey('muac_age_in_range')) {
      context.handle(
          _muacAgeInRangeMeta,
          muacAgeInRange.isAcceptableOrUnknown(
              data['muac_age_in_range']!, _muacAgeInRangeMeta));
    }
    if (data.containsKey('muac_confidence')) {
      context.handle(
          _muacConfidenceMeta,
          muacConfidence.isAcceptableOrUnknown(
              data['muac_confidence']!, _muacConfidenceMeta));
    }
    if (data.containsKey('muac_uncertainty_lower_cm')) {
      context.handle(
          _muacUncertaintyLowerCmMeta,
          muacUncertaintyLowerCm.isAcceptableOrUnknown(
              data['muac_uncertainty_lower_cm']!, _muacUncertaintyLowerCmMeta));
    }
    if (data.containsKey('muac_uncertainty_upper_cm')) {
      context.handle(
          _muacUncertaintyUpperCmMeta,
          muacUncertaintyUpperCm.isAcceptableOrUnknown(
              data['muac_uncertainty_upper_cm']!, _muacUncertaintyUpperCmMeta));
    }
    if (data.containsKey('muac_model_version')) {
      context.handle(
          _muacModelVersionMeta,
          muacModelVersion.isAcceptableOrUnknown(
              data['muac_model_version']!, _muacModelVersionMeta));
    }
    if (data.containsKey('muac_calibration_version')) {
      context.handle(
          _muacCalibrationVersionMeta,
          muacCalibrationVersion.isAcceptableOrUnknown(
              data['muac_calibration_version']!, _muacCalibrationVersionMeta));
    }
    if (data.containsKey('muac_is_direct_measurement')) {
      context.handle(
          _muacIsDirectMeasurementMeta,
          muacIsDirectMeasurement.isAcceptableOrUnknown(
              data['muac_is_direct_measurement']!,
              _muacIsDirectMeasurementMeta));
    }
    if (data.containsKey('muac_requires_confirmation')) {
      context.handle(
          _muacRequiresConfirmationMeta,
          muacRequiresConfirmation.isAcceptableOrUnknown(
              data['muac_requires_confirmation']!,
              _muacRequiresConfirmationMeta));
    }
    if (data.containsKey('muac_referral_guidance')) {
      context.handle(
          _muacReferralGuidanceMeta,
          muacReferralGuidance.isAcceptableOrUnknown(
              data['muac_referral_guidance']!, _muacReferralGuidanceMeta));
    }
    if (data.containsKey('combined_status')) {
      context.handle(
          _combinedStatusMeta,
          combinedStatus.isAcceptableOrUnknown(
              data['combined_status']!, _combinedStatusMeta));
    }
    if (data.containsKey('combined_triggered_by')) {
      context.handle(
          _combinedTriggeredByMeta,
          combinedTriggeredBy.isAcceptableOrUnknown(
              data['combined_triggered_by']!, _combinedTriggeredByMeta));
    }
    if (data.containsKey('combined_rationale')) {
      context.handle(
          _combinedRationaleMeta,
          combinedRationale.isAcceptableOrUnknown(
              data['combined_rationale']!, _combinedRationaleMeta));
    }
    if (data.containsKey('combined_method')) {
      context.handle(
          _combinedMethodMeta,
          combinedMethod.isAcceptableOrUnknown(
              data['combined_method']!, _combinedMethodMeta));
    }
    if (data.containsKey('combined_confidence_score')) {
      context.handle(
          _combinedConfidenceScoreMeta,
          combinedConfidenceScore.isAcceptableOrUnknown(
              data['combined_confidence_score']!,
              _combinedConfidenceScoreMeta));
    }
    if (data.containsKey('combined_protocol_version')) {
      context.handle(
          _combinedProtocolVersionMeta,
          combinedProtocolVersion.isAcceptableOrUnknown(
              data['combined_protocol_version']!,
              _combinedProtocolVersionMeta));
    }
    if (data.containsKey('poshan_status')) {
      context.handle(
          _poshanStatusMeta,
          poshanStatus.isAcceptableOrUnknown(
              data['poshan_status']!, _poshanStatusMeta));
    }
    if (data.containsKey('poshan_triggered_by')) {
      context.handle(
          _poshanTriggeredByMeta,
          poshanTriggeredBy.isAcceptableOrUnknown(
              data['poshan_triggered_by']!, _poshanTriggeredByMeta));
    }
    if (data.containsKey('classification_method')) {
      context.handle(
          _classificationMethodMeta,
          classificationMethod.isAcceptableOrUnknown(
              data['classification_method']!, _classificationMethodMeta));
    }
    if (data.containsKey('classification_rationale')) {
      context.handle(
          _classificationRationaleMeta,
          classificationRationale.isAcceptableOrUnknown(
              data['classification_rationale']!, _classificationRationaleMeta));
    }
    if (data.containsKey('poshan_complete')) {
      context.handle(
          _poshanCompleteMeta,
          poshanComplete.isAcceptableOrUnknown(
              data['poshan_complete']!, _poshanCompleteMeta));
    }
    if (data.containsKey('measurement_mode')) {
      context.handle(
          _measurementModeMeta,
          measurementMode.isAcceptableOrUnknown(
              data['measurement_mode']!, _measurementModeMeta));
    }
    if (data.containsKey('oedema')) {
      context.handle(_oedemaMeta,
          oedema.isAcceptableOrUnknown(data['oedema']!, _oedemaMeta));
    }
    if (data.containsKey('measured_at')) {
      context.handle(
          _measuredAtMeta,
          measuredAt.isAcceptableOrUnknown(
              data['measured_at']!, _measuredAtMeta));
    }
    if (data.containsKey('editor_user_id')) {
      context.handle(
          _editorUserIdMeta,
          editorUserId.isAcceptableOrUnknown(
              data['editor_user_id']!, _editorUserIdMeta));
    }
    if (data.containsKey('measured_notes')) {
      context.handle(
          _measuredNotesMeta,
          measuredNotes.isAcceptableOrUnknown(
              data['measured_notes']!, _measuredNotesMeta));
    }
    if (data.containsKey('who_acute_status')) {
      context.handle(
          _whoAcuteStatusMeta,
          whoAcuteStatus.isAcceptableOrUnknown(
              data['who_acute_status']!, _whoAcuteStatusMeta));
    }
    if (data.containsKey('who_acute_triggered_by')) {
      context.handle(
          _whoAcuteTriggeredByMeta,
          whoAcuteTriggeredBy.isAcceptableOrUnknown(
              data['who_acute_triggered_by']!, _whoAcuteTriggeredByMeta));
    }
    if (data.containsKey('who_acute_rationale')) {
      context.handle(
          _whoAcuteRationaleMeta,
          whoAcuteRationale.isAcceptableOrUnknown(
              data['who_acute_rationale']!, _whoAcuteRationaleMeta));
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
      effectiveHeightCm: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}effective_height_cm']),
      effectiveWeightKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}effective_weight_kg']),
      heightMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}height_method']),
      weightMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}weight_method']),
      bmi: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}bmi']),
      bmiStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}bmi_status']),
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
      heightConfidence: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}height_confidence']),
      weightConfidence: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}weight_confidence']),
      classificationConfidence: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}classification_confidence']),
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
      wastingMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}wasting_method']),
      muacCm: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}muac_cm']),
      muacStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}muac_status']),
      muacMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}muac_method']),
      muacAgeInRange: attachedDatabase.typeMapping
          .read(DriftSqlType.bool, data['${effectivePrefix}muac_age_in_range']),
      muacConfidence: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}muac_confidence']),
      muacUncertaintyLowerCm: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}muac_uncertainty_lower_cm']),
      muacUncertaintyUpperCm: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}muac_uncertainty_upper_cm']),
      muacModelVersion: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}muac_model_version']),
      muacCalibrationVersion: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}muac_calibration_version']),
      muacIsDirectMeasurement: attachedDatabase.typeMapping.read(
          DriftSqlType.bool,
          data['${effectivePrefix}muac_is_direct_measurement']),
      muacRequiresConfirmation: attachedDatabase.typeMapping.read(
          DriftSqlType.bool,
          data['${effectivePrefix}muac_requires_confirmation']),
      muacReferralGuidance: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}muac_referral_guidance']),
      combinedStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}combined_status']),
      combinedTriggeredBy: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}combined_triggered_by']),
      combinedRationale: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}combined_rationale']),
      combinedMethod: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}combined_method']),
      combinedConfidenceScore: attachedDatabase.typeMapping.read(
          DriftSqlType.double,
          data['${effectivePrefix}combined_confidence_score']),
      combinedProtocolVersion: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}combined_protocol_version']),
      poshanStatus: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}poshan_status']),
      poshanTriggeredBy: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}poshan_triggered_by']),
      classificationMethod: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}classification_method']),
      classificationRationale: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}classification_rationale']),
      poshanComplete: attachedDatabase.typeMapping
          .read(DriftSqlType.bool, data['${effectivePrefix}poshan_complete']),
      measurementMode: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}measurement_mode']),
      oedema: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}oedema']),
      measuredAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}measured_at']),
      editorUserId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}editor_user_id']),
      measuredNotes: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}measured_notes']),
      whoAcuteStatus: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}who_acute_status']),
      whoAcuteTriggeredBy: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}who_acute_triggered_by']),
      whoAcuteRationale: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}who_acute_rationale']),
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
  final double? effectiveHeightCm;
  final double? effectiveWeightKg;
  final String? heightMethod;
  final String? weightMethod;
  final double? bmi;
  final String? bmiStatus;
  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? confidenceScore;
  final double? heightConfidence;
  final double? weightConfidence;
  final double? classificationConfidence;
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
  final String? wastingMethod;
  final double? muacCm;
  final String? muacStatus;
  final String? muacMethod;
  final bool? muacAgeInRange;
  final double? muacConfidence;
  final double? muacUncertaintyLowerCm;
  final double? muacUncertaintyUpperCm;
  final String? muacModelVersion;
  final String? muacCalibrationVersion;
  final bool? muacIsDirectMeasurement;
  final bool? muacRequiresConfirmation;
  final String? muacReferralGuidance;
  final String? combinedStatus;
  final String? combinedTriggeredBy;
  final String? combinedRationale;
  final String? combinedMethod;
  final double? combinedConfidenceScore;
  final String? combinedProtocolVersion;
  final String? poshanStatus;
  final String? poshanTriggeredBy;
  final String? classificationMethod;
  final String? classificationRationale;
  final bool? poshanComplete;
  final String? measurementMode;
  final String? oedema;
  final DateTime? measuredAt;
  final int? editorUserId;
  final String? measuredNotes;
  final String? whoAcuteStatus;
  final String? whoAcuteTriggeredBy;
  final String? whoAcuteRationale;
  const Measurement(
      {required this.id,
      required this.visitId,
      this.predictedHeightCm,
      this.predictedWeightKg,
      this.manualHeightCm,
      this.manualWeightKg,
      this.effectiveHeightCm,
      this.effectiveWeightKg,
      this.heightMethod,
      this.weightMethod,
      this.bmi,
      this.bmiStatus,
      this.hazZscore,
      this.whzZscore,
      this.hazStatus,
      this.whzStatus,
      this.confidenceScore,
      this.heightConfidence,
      this.weightConfidence,
      this.classificationConfidence,
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
      this.wastingMethod,
      this.muacCm,
      this.muacStatus,
      this.muacMethod,
      this.muacAgeInRange,
      this.muacConfidence,
      this.muacUncertaintyLowerCm,
      this.muacUncertaintyUpperCm,
      this.muacModelVersion,
      this.muacCalibrationVersion,
      this.muacIsDirectMeasurement,
      this.muacRequiresConfirmation,
      this.muacReferralGuidance,
      this.combinedStatus,
      this.combinedTriggeredBy,
      this.combinedRationale,
      this.combinedMethod,
      this.combinedConfidenceScore,
      this.combinedProtocolVersion,
      this.poshanStatus,
      this.poshanTriggeredBy,
      this.classificationMethod,
      this.classificationRationale,
      this.poshanComplete,
      this.measurementMode,
      this.oedema,
      this.measuredAt,
      this.editorUserId,
      this.measuredNotes,
      this.whoAcuteStatus,
      this.whoAcuteTriggeredBy,
      this.whoAcuteRationale});
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
    if (!nullToAbsent || effectiveHeightCm != null) {
      map['effective_height_cm'] = Variable<double>(effectiveHeightCm);
    }
    if (!nullToAbsent || effectiveWeightKg != null) {
      map['effective_weight_kg'] = Variable<double>(effectiveWeightKg);
    }
    if (!nullToAbsent || heightMethod != null) {
      map['height_method'] = Variable<String>(heightMethod);
    }
    if (!nullToAbsent || weightMethod != null) {
      map['weight_method'] = Variable<String>(weightMethod);
    }
    if (!nullToAbsent || bmi != null) {
      map['bmi'] = Variable<double>(bmi);
    }
    if (!nullToAbsent || bmiStatus != null) {
      map['bmi_status'] = Variable<String>(bmiStatus);
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
    if (!nullToAbsent || heightConfidence != null) {
      map['height_confidence'] = Variable<double>(heightConfidence);
    }
    if (!nullToAbsent || weightConfidence != null) {
      map['weight_confidence'] = Variable<double>(weightConfidence);
    }
    if (!nullToAbsent || classificationConfidence != null) {
      map['classification_confidence'] =
          Variable<double>(classificationConfidence);
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
    if (!nullToAbsent || wastingMethod != null) {
      map['wasting_method'] = Variable<String>(wastingMethod);
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
    if (!nullToAbsent || muacAgeInRange != null) {
      map['muac_age_in_range'] = Variable<bool>(muacAgeInRange);
    }
    if (!nullToAbsent || muacConfidence != null) {
      map['muac_confidence'] = Variable<double>(muacConfidence);
    }
    if (!nullToAbsent || muacUncertaintyLowerCm != null) {
      map['muac_uncertainty_lower_cm'] =
          Variable<double>(muacUncertaintyLowerCm);
    }
    if (!nullToAbsent || muacUncertaintyUpperCm != null) {
      map['muac_uncertainty_upper_cm'] =
          Variable<double>(muacUncertaintyUpperCm);
    }
    if (!nullToAbsent || muacModelVersion != null) {
      map['muac_model_version'] = Variable<String>(muacModelVersion);
    }
    if (!nullToAbsent || muacCalibrationVersion != null) {
      map['muac_calibration_version'] =
          Variable<String>(muacCalibrationVersion);
    }
    if (!nullToAbsent || muacIsDirectMeasurement != null) {
      map['muac_is_direct_measurement'] =
          Variable<bool>(muacIsDirectMeasurement);
    }
    if (!nullToAbsent || muacRequiresConfirmation != null) {
      map['muac_requires_confirmation'] =
          Variable<bool>(muacRequiresConfirmation);
    }
    if (!nullToAbsent || muacReferralGuidance != null) {
      map['muac_referral_guidance'] = Variable<String>(muacReferralGuidance);
    }
    if (!nullToAbsent || combinedStatus != null) {
      map['combined_status'] = Variable<String>(combinedStatus);
    }
    if (!nullToAbsent || combinedTriggeredBy != null) {
      map['combined_triggered_by'] = Variable<String>(combinedTriggeredBy);
    }
    if (!nullToAbsent || combinedRationale != null) {
      map['combined_rationale'] = Variable<String>(combinedRationale);
    }
    if (!nullToAbsent || combinedMethod != null) {
      map['combined_method'] = Variable<String>(combinedMethod);
    }
    if (!nullToAbsent || combinedConfidenceScore != null) {
      map['combined_confidence_score'] =
          Variable<double>(combinedConfidenceScore);
    }
    if (!nullToAbsent || combinedProtocolVersion != null) {
      map['combined_protocol_version'] =
          Variable<String>(combinedProtocolVersion);
    }
    if (!nullToAbsent || poshanStatus != null) {
      map['poshan_status'] = Variable<String>(poshanStatus);
    }
    if (!nullToAbsent || poshanTriggeredBy != null) {
      map['poshan_triggered_by'] = Variable<String>(poshanTriggeredBy);
    }
    if (!nullToAbsent || classificationMethod != null) {
      map['classification_method'] = Variable<String>(classificationMethod);
    }
    if (!nullToAbsent || classificationRationale != null) {
      map['classification_rationale'] =
          Variable<String>(classificationRationale);
    }
    if (!nullToAbsent || poshanComplete != null) {
      map['poshan_complete'] = Variable<bool>(poshanComplete);
    }
    if (!nullToAbsent || measurementMode != null) {
      map['measurement_mode'] = Variable<String>(measurementMode);
    }
    if (!nullToAbsent || oedema != null) {
      map['oedema'] = Variable<String>(oedema);
    }
    if (!nullToAbsent || measuredAt != null) {
      map['measured_at'] = Variable<DateTime>(measuredAt);
    }
    if (!nullToAbsent || editorUserId != null) {
      map['editor_user_id'] = Variable<int>(editorUserId);
    }
    if (!nullToAbsent || measuredNotes != null) {
      map['measured_notes'] = Variable<String>(measuredNotes);
    }
    if (!nullToAbsent || whoAcuteStatus != null) {
      map['who_acute_status'] = Variable<String>(whoAcuteStatus);
    }
    if (!nullToAbsent || whoAcuteTriggeredBy != null) {
      map['who_acute_triggered_by'] = Variable<String>(whoAcuteTriggeredBy);
    }
    if (!nullToAbsent || whoAcuteRationale != null) {
      map['who_acute_rationale'] = Variable<String>(whoAcuteRationale);
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
      effectiveHeightCm: effectiveHeightCm == null && nullToAbsent
          ? const Value.absent()
          : Value(effectiveHeightCm),
      effectiveWeightKg: effectiveWeightKg == null && nullToAbsent
          ? const Value.absent()
          : Value(effectiveWeightKg),
      heightMethod: heightMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(heightMethod),
      weightMethod: weightMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(weightMethod),
      bmi: bmi == null && nullToAbsent ? const Value.absent() : Value(bmi),
      bmiStatus: bmiStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(bmiStatus),
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
      heightConfidence: heightConfidence == null && nullToAbsent
          ? const Value.absent()
          : Value(heightConfidence),
      weightConfidence: weightConfidence == null && nullToAbsent
          ? const Value.absent()
          : Value(weightConfidence),
      classificationConfidence: classificationConfidence == null && nullToAbsent
          ? const Value.absent()
          : Value(classificationConfidence),
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
      wastingMethod: wastingMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(wastingMethod),
      muacCm:
          muacCm == null && nullToAbsent ? const Value.absent() : Value(muacCm),
      muacStatus: muacStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(muacStatus),
      muacMethod: muacMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(muacMethod),
      muacAgeInRange: muacAgeInRange == null && nullToAbsent
          ? const Value.absent()
          : Value(muacAgeInRange),
      muacConfidence: muacConfidence == null && nullToAbsent
          ? const Value.absent()
          : Value(muacConfidence),
      muacUncertaintyLowerCm: muacUncertaintyLowerCm == null && nullToAbsent
          ? const Value.absent()
          : Value(muacUncertaintyLowerCm),
      muacUncertaintyUpperCm: muacUncertaintyUpperCm == null && nullToAbsent
          ? const Value.absent()
          : Value(muacUncertaintyUpperCm),
      muacModelVersion: muacModelVersion == null && nullToAbsent
          ? const Value.absent()
          : Value(muacModelVersion),
      muacCalibrationVersion: muacCalibrationVersion == null && nullToAbsent
          ? const Value.absent()
          : Value(muacCalibrationVersion),
      muacIsDirectMeasurement: muacIsDirectMeasurement == null && nullToAbsent
          ? const Value.absent()
          : Value(muacIsDirectMeasurement),
      muacRequiresConfirmation: muacRequiresConfirmation == null && nullToAbsent
          ? const Value.absent()
          : Value(muacRequiresConfirmation),
      muacReferralGuidance: muacReferralGuidance == null && nullToAbsent
          ? const Value.absent()
          : Value(muacReferralGuidance),
      combinedStatus: combinedStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(combinedStatus),
      combinedTriggeredBy: combinedTriggeredBy == null && nullToAbsent
          ? const Value.absent()
          : Value(combinedTriggeredBy),
      combinedRationale: combinedRationale == null && nullToAbsent
          ? const Value.absent()
          : Value(combinedRationale),
      combinedMethod: combinedMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(combinedMethod),
      combinedConfidenceScore: combinedConfidenceScore == null && nullToAbsent
          ? const Value.absent()
          : Value(combinedConfidenceScore),
      combinedProtocolVersion: combinedProtocolVersion == null && nullToAbsent
          ? const Value.absent()
          : Value(combinedProtocolVersion),
      poshanStatus: poshanStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(poshanStatus),
      poshanTriggeredBy: poshanTriggeredBy == null && nullToAbsent
          ? const Value.absent()
          : Value(poshanTriggeredBy),
      classificationMethod: classificationMethod == null && nullToAbsent
          ? const Value.absent()
          : Value(classificationMethod),
      classificationRationale: classificationRationale == null && nullToAbsent
          ? const Value.absent()
          : Value(classificationRationale),
      poshanComplete: poshanComplete == null && nullToAbsent
          ? const Value.absent()
          : Value(poshanComplete),
      measurementMode: measurementMode == null && nullToAbsent
          ? const Value.absent()
          : Value(measurementMode),
      oedema:
          oedema == null && nullToAbsent ? const Value.absent() : Value(oedema),
      measuredAt: measuredAt == null && nullToAbsent
          ? const Value.absent()
          : Value(measuredAt),
      editorUserId: editorUserId == null && nullToAbsent
          ? const Value.absent()
          : Value(editorUserId),
      measuredNotes: measuredNotes == null && nullToAbsent
          ? const Value.absent()
          : Value(measuredNotes),
      whoAcuteStatus: whoAcuteStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(whoAcuteStatus),
      whoAcuteTriggeredBy: whoAcuteTriggeredBy == null && nullToAbsent
          ? const Value.absent()
          : Value(whoAcuteTriggeredBy),
      whoAcuteRationale: whoAcuteRationale == null && nullToAbsent
          ? const Value.absent()
          : Value(whoAcuteRationale),
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
      effectiveHeightCm:
          serializer.fromJson<double?>(json['effectiveHeightCm']),
      effectiveWeightKg:
          serializer.fromJson<double?>(json['effectiveWeightKg']),
      heightMethod: serializer.fromJson<String?>(json['heightMethod']),
      weightMethod: serializer.fromJson<String?>(json['weightMethod']),
      bmi: serializer.fromJson<double?>(json['bmi']),
      bmiStatus: serializer.fromJson<String?>(json['bmiStatus']),
      hazZscore: serializer.fromJson<double?>(json['hazZscore']),
      whzZscore: serializer.fromJson<double?>(json['whzZscore']),
      hazStatus: serializer.fromJson<String?>(json['hazStatus']),
      whzStatus: serializer.fromJson<String?>(json['whzStatus']),
      confidenceScore: serializer.fromJson<double?>(json['confidenceScore']),
      heightConfidence: serializer.fromJson<double?>(json['heightConfidence']),
      weightConfidence: serializer.fromJson<double?>(json['weightConfidence']),
      classificationConfidence:
          serializer.fromJson<double?>(json['classificationConfidence']),
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
      wastingMethod: serializer.fromJson<String?>(json['wastingMethod']),
      muacCm: serializer.fromJson<double?>(json['muacCm']),
      muacStatus: serializer.fromJson<String?>(json['muacStatus']),
      muacMethod: serializer.fromJson<String?>(json['muacMethod']),
      muacAgeInRange: serializer.fromJson<bool?>(json['muacAgeInRange']),
      muacConfidence: serializer.fromJson<double?>(json['muacConfidence']),
      muacUncertaintyLowerCm:
          serializer.fromJson<double?>(json['muacUncertaintyLowerCm']),
      muacUncertaintyUpperCm:
          serializer.fromJson<double?>(json['muacUncertaintyUpperCm']),
      muacModelVersion: serializer.fromJson<String?>(json['muacModelVersion']),
      muacCalibrationVersion:
          serializer.fromJson<String?>(json['muacCalibrationVersion']),
      muacIsDirectMeasurement:
          serializer.fromJson<bool?>(json['muacIsDirectMeasurement']),
      muacRequiresConfirmation:
          serializer.fromJson<bool?>(json['muacRequiresConfirmation']),
      muacReferralGuidance:
          serializer.fromJson<String?>(json['muacReferralGuidance']),
      combinedStatus: serializer.fromJson<String?>(json['combinedStatus']),
      combinedTriggeredBy:
          serializer.fromJson<String?>(json['combinedTriggeredBy']),
      combinedRationale:
          serializer.fromJson<String?>(json['combinedRationale']),
      combinedMethod: serializer.fromJson<String?>(json['combinedMethod']),
      combinedConfidenceScore:
          serializer.fromJson<double?>(json['combinedConfidenceScore']),
      combinedProtocolVersion:
          serializer.fromJson<String?>(json['combinedProtocolVersion']),
      poshanStatus: serializer.fromJson<String?>(json['poshanStatus']),
      poshanTriggeredBy:
          serializer.fromJson<String?>(json['poshanTriggeredBy']),
      classificationMethod:
          serializer.fromJson<String?>(json['classificationMethod']),
      classificationRationale:
          serializer.fromJson<String?>(json['classificationRationale']),
      poshanComplete: serializer.fromJson<bool?>(json['poshanComplete']),
      measurementMode: serializer.fromJson<String?>(json['measurementMode']),
      oedema: serializer.fromJson<String?>(json['oedema']),
      measuredAt: serializer.fromJson<DateTime?>(json['measuredAt']),
      editorUserId: serializer.fromJson<int?>(json['editorUserId']),
      measuredNotes: serializer.fromJson<String?>(json['measuredNotes']),
      whoAcuteStatus: serializer.fromJson<String?>(json['whoAcuteStatus']),
      whoAcuteTriggeredBy:
          serializer.fromJson<String?>(json['whoAcuteTriggeredBy']),
      whoAcuteRationale:
          serializer.fromJson<String?>(json['whoAcuteRationale']),
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
      'effectiveHeightCm': serializer.toJson<double?>(effectiveHeightCm),
      'effectiveWeightKg': serializer.toJson<double?>(effectiveWeightKg),
      'heightMethod': serializer.toJson<String?>(heightMethod),
      'weightMethod': serializer.toJson<String?>(weightMethod),
      'bmi': serializer.toJson<double?>(bmi),
      'bmiStatus': serializer.toJson<String?>(bmiStatus),
      'hazZscore': serializer.toJson<double?>(hazZscore),
      'whzZscore': serializer.toJson<double?>(whzZscore),
      'hazStatus': serializer.toJson<String?>(hazStatus),
      'whzStatus': serializer.toJson<String?>(whzStatus),
      'confidenceScore': serializer.toJson<double?>(confidenceScore),
      'heightConfidence': serializer.toJson<double?>(heightConfidence),
      'weightConfidence': serializer.toJson<double?>(weightConfidence),
      'classificationConfidence':
          serializer.toJson<double?>(classificationConfidence),
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
      'wastingMethod': serializer.toJson<String?>(wastingMethod),
      'muacCm': serializer.toJson<double?>(muacCm),
      'muacStatus': serializer.toJson<String?>(muacStatus),
      'muacMethod': serializer.toJson<String?>(muacMethod),
      'muacAgeInRange': serializer.toJson<bool?>(muacAgeInRange),
      'muacConfidence': serializer.toJson<double?>(muacConfidence),
      'muacUncertaintyLowerCm':
          serializer.toJson<double?>(muacUncertaintyLowerCm),
      'muacUncertaintyUpperCm':
          serializer.toJson<double?>(muacUncertaintyUpperCm),
      'muacModelVersion': serializer.toJson<String?>(muacModelVersion),
      'muacCalibrationVersion':
          serializer.toJson<String?>(muacCalibrationVersion),
      'muacIsDirectMeasurement':
          serializer.toJson<bool?>(muacIsDirectMeasurement),
      'muacRequiresConfirmation':
          serializer.toJson<bool?>(muacRequiresConfirmation),
      'muacReferralGuidance': serializer.toJson<String?>(muacReferralGuidance),
      'combinedStatus': serializer.toJson<String?>(combinedStatus),
      'combinedTriggeredBy': serializer.toJson<String?>(combinedTriggeredBy),
      'combinedRationale': serializer.toJson<String?>(combinedRationale),
      'combinedMethod': serializer.toJson<String?>(combinedMethod),
      'combinedConfidenceScore':
          serializer.toJson<double?>(combinedConfidenceScore),
      'combinedProtocolVersion':
          serializer.toJson<String?>(combinedProtocolVersion),
      'poshanStatus': serializer.toJson<String?>(poshanStatus),
      'poshanTriggeredBy': serializer.toJson<String?>(poshanTriggeredBy),
      'classificationMethod': serializer.toJson<String?>(classificationMethod),
      'classificationRationale':
          serializer.toJson<String?>(classificationRationale),
      'poshanComplete': serializer.toJson<bool?>(poshanComplete),
      'measurementMode': serializer.toJson<String?>(measurementMode),
      'oedema': serializer.toJson<String?>(oedema),
      'measuredAt': serializer.toJson<DateTime?>(measuredAt),
      'editorUserId': serializer.toJson<int?>(editorUserId),
      'measuredNotes': serializer.toJson<String?>(measuredNotes),
      'whoAcuteStatus': serializer.toJson<String?>(whoAcuteStatus),
      'whoAcuteTriggeredBy': serializer.toJson<String?>(whoAcuteTriggeredBy),
      'whoAcuteRationale': serializer.toJson<String?>(whoAcuteRationale),
    };
  }

  Measurement copyWith(
          {int? id,
          int? visitId,
          Value<double?> predictedHeightCm = const Value.absent(),
          Value<double?> predictedWeightKg = const Value.absent(),
          Value<double?> manualHeightCm = const Value.absent(),
          Value<double?> manualWeightKg = const Value.absent(),
          Value<double?> effectiveHeightCm = const Value.absent(),
          Value<double?> effectiveWeightKg = const Value.absent(),
          Value<String?> heightMethod = const Value.absent(),
          Value<String?> weightMethod = const Value.absent(),
          Value<double?> bmi = const Value.absent(),
          Value<String?> bmiStatus = const Value.absent(),
          Value<double?> hazZscore = const Value.absent(),
          Value<double?> whzZscore = const Value.absent(),
          Value<String?> hazStatus = const Value.absent(),
          Value<String?> whzStatus = const Value.absent(),
          Value<double?> confidenceScore = const Value.absent(),
          Value<double?> heightConfidence = const Value.absent(),
          Value<double?> weightConfidence = const Value.absent(),
          Value<double?> classificationConfidence = const Value.absent(),
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
          Value<String?> wastingMethod = const Value.absent(),
          Value<double?> muacCm = const Value.absent(),
          Value<String?> muacStatus = const Value.absent(),
          Value<String?> muacMethod = const Value.absent(),
          Value<bool?> muacAgeInRange = const Value.absent(),
          Value<double?> muacConfidence = const Value.absent(),
          Value<double?> muacUncertaintyLowerCm = const Value.absent(),
          Value<double?> muacUncertaintyUpperCm = const Value.absent(),
          Value<String?> muacModelVersion = const Value.absent(),
          Value<String?> muacCalibrationVersion = const Value.absent(),
          Value<bool?> muacIsDirectMeasurement = const Value.absent(),
          Value<bool?> muacRequiresConfirmation = const Value.absent(),
          Value<String?> muacReferralGuidance = const Value.absent(),
          Value<String?> combinedStatus = const Value.absent(),
          Value<String?> combinedTriggeredBy = const Value.absent(),
          Value<String?> combinedRationale = const Value.absent(),
          Value<String?> combinedMethod = const Value.absent(),
          Value<double?> combinedConfidenceScore = const Value.absent(),
          Value<String?> combinedProtocolVersion = const Value.absent(),
          Value<String?> poshanStatus = const Value.absent(),
          Value<String?> poshanTriggeredBy = const Value.absent(),
          Value<String?> classificationMethod = const Value.absent(),
          Value<String?> classificationRationale = const Value.absent(),
          Value<bool?> poshanComplete = const Value.absent(),
          Value<String?> measurementMode = const Value.absent(),
          Value<String?> oedema = const Value.absent(),
          Value<DateTime?> measuredAt = const Value.absent(),
          Value<int?> editorUserId = const Value.absent(),
          Value<String?> measuredNotes = const Value.absent(),
          Value<String?> whoAcuteStatus = const Value.absent(),
          Value<String?> whoAcuteTriggeredBy = const Value.absent(),
          Value<String?> whoAcuteRationale = const Value.absent()}) =>
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
        effectiveHeightCm: effectiveHeightCm.present
            ? effectiveHeightCm.value
            : this.effectiveHeightCm,
        effectiveWeightKg: effectiveWeightKg.present
            ? effectiveWeightKg.value
            : this.effectiveWeightKg,
        heightMethod:
            heightMethod.present ? heightMethod.value : this.heightMethod,
        weightMethod:
            weightMethod.present ? weightMethod.value : this.weightMethod,
        bmi: bmi.present ? bmi.value : this.bmi,
        bmiStatus: bmiStatus.present ? bmiStatus.value : this.bmiStatus,
        hazZscore: hazZscore.present ? hazZscore.value : this.hazZscore,
        whzZscore: whzZscore.present ? whzZscore.value : this.whzZscore,
        hazStatus: hazStatus.present ? hazStatus.value : this.hazStatus,
        whzStatus: whzStatus.present ? whzStatus.value : this.whzStatus,
        confidenceScore: confidenceScore.present
            ? confidenceScore.value
            : this.confidenceScore,
        heightConfidence: heightConfidence.present
            ? heightConfidence.value
            : this.heightConfidence,
        weightConfidence: weightConfidence.present
            ? weightConfidence.value
            : this.weightConfidence,
        classificationConfidence: classificationConfidence.present
            ? classificationConfidence.value
            : this.classificationConfidence,
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
        wastingMethod:
            wastingMethod.present ? wastingMethod.value : this.wastingMethod,
        muacCm: muacCm.present ? muacCm.value : this.muacCm,
        muacStatus: muacStatus.present ? muacStatus.value : this.muacStatus,
        muacMethod: muacMethod.present ? muacMethod.value : this.muacMethod,
        muacAgeInRange:
            muacAgeInRange.present ? muacAgeInRange.value : this.muacAgeInRange,
        muacConfidence:
            muacConfidence.present ? muacConfidence.value : this.muacConfidence,
        muacUncertaintyLowerCm: muacUncertaintyLowerCm.present
            ? muacUncertaintyLowerCm.value
            : this.muacUncertaintyLowerCm,
        muacUncertaintyUpperCm: muacUncertaintyUpperCm.present
            ? muacUncertaintyUpperCm.value
            : this.muacUncertaintyUpperCm,
        muacModelVersion: muacModelVersion.present
            ? muacModelVersion.value
            : this.muacModelVersion,
        muacCalibrationVersion: muacCalibrationVersion.present
            ? muacCalibrationVersion.value
            : this.muacCalibrationVersion,
        muacIsDirectMeasurement: muacIsDirectMeasurement.present
            ? muacIsDirectMeasurement.value
            : this.muacIsDirectMeasurement,
        muacRequiresConfirmation: muacRequiresConfirmation.present
            ? muacRequiresConfirmation.value
            : this.muacRequiresConfirmation,
        muacReferralGuidance: muacReferralGuidance.present
            ? muacReferralGuidance.value
            : this.muacReferralGuidance,
        combinedStatus:
            combinedStatus.present ? combinedStatus.value : this.combinedStatus,
        combinedTriggeredBy: combinedTriggeredBy.present
            ? combinedTriggeredBy.value
            : this.combinedTriggeredBy,
        combinedRationale: combinedRationale.present
            ? combinedRationale.value
            : this.combinedRationale,
        combinedMethod:
            combinedMethod.present ? combinedMethod.value : this.combinedMethod,
        combinedConfidenceScore: combinedConfidenceScore.present
            ? combinedConfidenceScore.value
            : this.combinedConfidenceScore,
        combinedProtocolVersion: combinedProtocolVersion.present
            ? combinedProtocolVersion.value
            : this.combinedProtocolVersion,
        poshanStatus:
            poshanStatus.present ? poshanStatus.value : this.poshanStatus,
        poshanTriggeredBy: poshanTriggeredBy.present
            ? poshanTriggeredBy.value
            : this.poshanTriggeredBy,
        classificationMethod: classificationMethod.present
            ? classificationMethod.value
            : this.classificationMethod,
        classificationRationale: classificationRationale.present
            ? classificationRationale.value
            : this.classificationRationale,
        poshanComplete:
            poshanComplete.present ? poshanComplete.value : this.poshanComplete,
        measurementMode: measurementMode.present
            ? measurementMode.value
            : this.measurementMode,
        oedema: oedema.present ? oedema.value : this.oedema,
        measuredAt: measuredAt.present ? measuredAt.value : this.measuredAt,
        editorUserId:
            editorUserId.present ? editorUserId.value : this.editorUserId,
        measuredNotes:
            measuredNotes.present ? measuredNotes.value : this.measuredNotes,
        whoAcuteStatus:
            whoAcuteStatus.present ? whoAcuteStatus.value : this.whoAcuteStatus,
        whoAcuteTriggeredBy: whoAcuteTriggeredBy.present
            ? whoAcuteTriggeredBy.value
            : this.whoAcuteTriggeredBy,
        whoAcuteRationale: whoAcuteRationale.present
            ? whoAcuteRationale.value
            : this.whoAcuteRationale,
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
      effectiveHeightCm: data.effectiveHeightCm.present
          ? data.effectiveHeightCm.value
          : this.effectiveHeightCm,
      effectiveWeightKg: data.effectiveWeightKg.present
          ? data.effectiveWeightKg.value
          : this.effectiveWeightKg,
      heightMethod: data.heightMethod.present
          ? data.heightMethod.value
          : this.heightMethod,
      weightMethod: data.weightMethod.present
          ? data.weightMethod.value
          : this.weightMethod,
      bmi: data.bmi.present ? data.bmi.value : this.bmi,
      bmiStatus: data.bmiStatus.present ? data.bmiStatus.value : this.bmiStatus,
      hazZscore: data.hazZscore.present ? data.hazZscore.value : this.hazZscore,
      whzZscore: data.whzZscore.present ? data.whzZscore.value : this.whzZscore,
      hazStatus: data.hazStatus.present ? data.hazStatus.value : this.hazStatus,
      whzStatus: data.whzStatus.present ? data.whzStatus.value : this.whzStatus,
      confidenceScore: data.confidenceScore.present
          ? data.confidenceScore.value
          : this.confidenceScore,
      heightConfidence: data.heightConfidence.present
          ? data.heightConfidence.value
          : this.heightConfidence,
      weightConfidence: data.weightConfidence.present
          ? data.weightConfidence.value
          : this.weightConfidence,
      classificationConfidence: data.classificationConfidence.present
          ? data.classificationConfidence.value
          : this.classificationConfidence,
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
      wastingMethod: data.wastingMethod.present
          ? data.wastingMethod.value
          : this.wastingMethod,
      muacCm: data.muacCm.present ? data.muacCm.value : this.muacCm,
      muacStatus:
          data.muacStatus.present ? data.muacStatus.value : this.muacStatus,
      muacMethod:
          data.muacMethod.present ? data.muacMethod.value : this.muacMethod,
      muacAgeInRange: data.muacAgeInRange.present
          ? data.muacAgeInRange.value
          : this.muacAgeInRange,
      muacConfidence: data.muacConfidence.present
          ? data.muacConfidence.value
          : this.muacConfidence,
      muacUncertaintyLowerCm: data.muacUncertaintyLowerCm.present
          ? data.muacUncertaintyLowerCm.value
          : this.muacUncertaintyLowerCm,
      muacUncertaintyUpperCm: data.muacUncertaintyUpperCm.present
          ? data.muacUncertaintyUpperCm.value
          : this.muacUncertaintyUpperCm,
      muacModelVersion: data.muacModelVersion.present
          ? data.muacModelVersion.value
          : this.muacModelVersion,
      muacCalibrationVersion: data.muacCalibrationVersion.present
          ? data.muacCalibrationVersion.value
          : this.muacCalibrationVersion,
      muacIsDirectMeasurement: data.muacIsDirectMeasurement.present
          ? data.muacIsDirectMeasurement.value
          : this.muacIsDirectMeasurement,
      muacRequiresConfirmation: data.muacRequiresConfirmation.present
          ? data.muacRequiresConfirmation.value
          : this.muacRequiresConfirmation,
      muacReferralGuidance: data.muacReferralGuidance.present
          ? data.muacReferralGuidance.value
          : this.muacReferralGuidance,
      combinedStatus: data.combinedStatus.present
          ? data.combinedStatus.value
          : this.combinedStatus,
      combinedTriggeredBy: data.combinedTriggeredBy.present
          ? data.combinedTriggeredBy.value
          : this.combinedTriggeredBy,
      combinedRationale: data.combinedRationale.present
          ? data.combinedRationale.value
          : this.combinedRationale,
      combinedMethod: data.combinedMethod.present
          ? data.combinedMethod.value
          : this.combinedMethod,
      combinedConfidenceScore: data.combinedConfidenceScore.present
          ? data.combinedConfidenceScore.value
          : this.combinedConfidenceScore,
      combinedProtocolVersion: data.combinedProtocolVersion.present
          ? data.combinedProtocolVersion.value
          : this.combinedProtocolVersion,
      poshanStatus: data.poshanStatus.present
          ? data.poshanStatus.value
          : this.poshanStatus,
      poshanTriggeredBy: data.poshanTriggeredBy.present
          ? data.poshanTriggeredBy.value
          : this.poshanTriggeredBy,
      classificationMethod: data.classificationMethod.present
          ? data.classificationMethod.value
          : this.classificationMethod,
      classificationRationale: data.classificationRationale.present
          ? data.classificationRationale.value
          : this.classificationRationale,
      poshanComplete: data.poshanComplete.present
          ? data.poshanComplete.value
          : this.poshanComplete,
      measurementMode: data.measurementMode.present
          ? data.measurementMode.value
          : this.measurementMode,
      oedema: data.oedema.present ? data.oedema.value : this.oedema,
      measuredAt:
          data.measuredAt.present ? data.measuredAt.value : this.measuredAt,
      editorUserId: data.editorUserId.present
          ? data.editorUserId.value
          : this.editorUserId,
      measuredNotes: data.measuredNotes.present
          ? data.measuredNotes.value
          : this.measuredNotes,
      whoAcuteStatus: data.whoAcuteStatus.present
          ? data.whoAcuteStatus.value
          : this.whoAcuteStatus,
      whoAcuteTriggeredBy: data.whoAcuteTriggeredBy.present
          ? data.whoAcuteTriggeredBy.value
          : this.whoAcuteTriggeredBy,
      whoAcuteRationale: data.whoAcuteRationale.present
          ? data.whoAcuteRationale.value
          : this.whoAcuteRationale,
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
          ..write('effectiveHeightCm: $effectiveHeightCm, ')
          ..write('effectiveWeightKg: $effectiveWeightKg, ')
          ..write('heightMethod: $heightMethod, ')
          ..write('weightMethod: $weightMethod, ')
          ..write('bmi: $bmi, ')
          ..write('bmiStatus: $bmiStatus, ')
          ..write('hazZscore: $hazZscore, ')
          ..write('whzZscore: $whzZscore, ')
          ..write('hazStatus: $hazStatus, ')
          ..write('whzStatus: $whzStatus, ')
          ..write('confidenceScore: $confidenceScore, ')
          ..write('heightConfidence: $heightConfidence, ')
          ..write('weightConfidence: $weightConfidence, ')
          ..write('classificationConfidence: $classificationConfidence, ')
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
          ..write('wastingMethod: $wastingMethod, ')
          ..write('muacCm: $muacCm, ')
          ..write('muacStatus: $muacStatus, ')
          ..write('muacMethod: $muacMethod, ')
          ..write('muacAgeInRange: $muacAgeInRange, ')
          ..write('muacConfidence: $muacConfidence, ')
          ..write('muacUncertaintyLowerCm: $muacUncertaintyLowerCm, ')
          ..write('muacUncertaintyUpperCm: $muacUncertaintyUpperCm, ')
          ..write('muacModelVersion: $muacModelVersion, ')
          ..write('muacCalibrationVersion: $muacCalibrationVersion, ')
          ..write('muacIsDirectMeasurement: $muacIsDirectMeasurement, ')
          ..write('muacRequiresConfirmation: $muacRequiresConfirmation, ')
          ..write('muacReferralGuidance: $muacReferralGuidance, ')
          ..write('combinedStatus: $combinedStatus, ')
          ..write('combinedTriggeredBy: $combinedTriggeredBy, ')
          ..write('combinedRationale: $combinedRationale, ')
          ..write('combinedMethod: $combinedMethod, ')
          ..write('combinedConfidenceScore: $combinedConfidenceScore, ')
          ..write('combinedProtocolVersion: $combinedProtocolVersion, ')
          ..write('poshanStatus: $poshanStatus, ')
          ..write('poshanTriggeredBy: $poshanTriggeredBy, ')
          ..write('classificationMethod: $classificationMethod, ')
          ..write('classificationRationale: $classificationRationale, ')
          ..write('poshanComplete: $poshanComplete, ')
          ..write('measurementMode: $measurementMode, ')
          ..write('oedema: $oedema, ')
          ..write('measuredAt: $measuredAt, ')
          ..write('editorUserId: $editorUserId, ')
          ..write('measuredNotes: $measuredNotes, ')
          ..write('whoAcuteStatus: $whoAcuteStatus, ')
          ..write('whoAcuteTriggeredBy: $whoAcuteTriggeredBy, ')
          ..write('whoAcuteRationale: $whoAcuteRationale')
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
        effectiveHeightCm,
        effectiveWeightKg,
        heightMethod,
        weightMethod,
        bmi,
        bmiStatus,
        hazZscore,
        whzZscore,
        hazStatus,
        whzStatus,
        confidenceScore,
        heightConfidence,
        weightConfidence,
        classificationConfidence,
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
        wastingMethod,
        muacCm,
        muacStatus,
        muacMethod,
        muacAgeInRange,
        muacConfidence,
        muacUncertaintyLowerCm,
        muacUncertaintyUpperCm,
        muacModelVersion,
        muacCalibrationVersion,
        muacIsDirectMeasurement,
        muacRequiresConfirmation,
        muacReferralGuidance,
        combinedStatus,
        combinedTriggeredBy,
        combinedRationale,
        combinedMethod,
        combinedConfidenceScore,
        combinedProtocolVersion,
        poshanStatus,
        poshanTriggeredBy,
        classificationMethod,
        classificationRationale,
        poshanComplete,
        measurementMode,
        oedema,
        measuredAt,
        editorUserId,
        measuredNotes,
        whoAcuteStatus,
        whoAcuteTriggeredBy,
        whoAcuteRationale
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
          other.effectiveHeightCm == this.effectiveHeightCm &&
          other.effectiveWeightKg == this.effectiveWeightKg &&
          other.heightMethod == this.heightMethod &&
          other.weightMethod == this.weightMethod &&
          other.bmi == this.bmi &&
          other.bmiStatus == this.bmiStatus &&
          other.hazZscore == this.hazZscore &&
          other.whzZscore == this.whzZscore &&
          other.hazStatus == this.hazStatus &&
          other.whzStatus == this.whzStatus &&
          other.confidenceScore == this.confidenceScore &&
          other.heightConfidence == this.heightConfidence &&
          other.weightConfidence == this.weightConfidence &&
          other.classificationConfidence == this.classificationConfidence &&
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
          other.wastingMethod == this.wastingMethod &&
          other.muacCm == this.muacCm &&
          other.muacStatus == this.muacStatus &&
          other.muacMethod == this.muacMethod &&
          other.muacAgeInRange == this.muacAgeInRange &&
          other.muacConfidence == this.muacConfidence &&
          other.muacUncertaintyLowerCm == this.muacUncertaintyLowerCm &&
          other.muacUncertaintyUpperCm == this.muacUncertaintyUpperCm &&
          other.muacModelVersion == this.muacModelVersion &&
          other.muacCalibrationVersion == this.muacCalibrationVersion &&
          other.muacIsDirectMeasurement == this.muacIsDirectMeasurement &&
          other.muacRequiresConfirmation == this.muacRequiresConfirmation &&
          other.muacReferralGuidance == this.muacReferralGuidance &&
          other.combinedStatus == this.combinedStatus &&
          other.combinedTriggeredBy == this.combinedTriggeredBy &&
          other.combinedRationale == this.combinedRationale &&
          other.combinedMethod == this.combinedMethod &&
          other.combinedConfidenceScore == this.combinedConfidenceScore &&
          other.combinedProtocolVersion == this.combinedProtocolVersion &&
          other.poshanStatus == this.poshanStatus &&
          other.poshanTriggeredBy == this.poshanTriggeredBy &&
          other.classificationMethod == this.classificationMethod &&
          other.classificationRationale == this.classificationRationale &&
          other.poshanComplete == this.poshanComplete &&
          other.measurementMode == this.measurementMode &&
          other.oedema == this.oedema &&
          other.measuredAt == this.measuredAt &&
          other.editorUserId == this.editorUserId &&
          other.measuredNotes == this.measuredNotes &&
          other.whoAcuteStatus == this.whoAcuteStatus &&
          other.whoAcuteTriggeredBy == this.whoAcuteTriggeredBy &&
          other.whoAcuteRationale == this.whoAcuteRationale);
}

class MeasurementsCompanion extends UpdateCompanion<Measurement> {
  final Value<int> id;
  final Value<int> visitId;
  final Value<double?> predictedHeightCm;
  final Value<double?> predictedWeightKg;
  final Value<double?> manualHeightCm;
  final Value<double?> manualWeightKg;
  final Value<double?> effectiveHeightCm;
  final Value<double?> effectiveWeightKg;
  final Value<String?> heightMethod;
  final Value<String?> weightMethod;
  final Value<double?> bmi;
  final Value<String?> bmiStatus;
  final Value<double?> hazZscore;
  final Value<double?> whzZscore;
  final Value<String?> hazStatus;
  final Value<String?> whzStatus;
  final Value<double?> confidenceScore;
  final Value<double?> heightConfidence;
  final Value<double?> weightConfidence;
  final Value<double?> classificationConfidence;
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
  final Value<String?> wastingMethod;
  final Value<double?> muacCm;
  final Value<String?> muacStatus;
  final Value<String?> muacMethod;
  final Value<bool?> muacAgeInRange;
  final Value<double?> muacConfidence;
  final Value<double?> muacUncertaintyLowerCm;
  final Value<double?> muacUncertaintyUpperCm;
  final Value<String?> muacModelVersion;
  final Value<String?> muacCalibrationVersion;
  final Value<bool?> muacIsDirectMeasurement;
  final Value<bool?> muacRequiresConfirmation;
  final Value<String?> muacReferralGuidance;
  final Value<String?> combinedStatus;
  final Value<String?> combinedTriggeredBy;
  final Value<String?> combinedRationale;
  final Value<String?> combinedMethod;
  final Value<double?> combinedConfidenceScore;
  final Value<String?> combinedProtocolVersion;
  final Value<String?> poshanStatus;
  final Value<String?> poshanTriggeredBy;
  final Value<String?> classificationMethod;
  final Value<String?> classificationRationale;
  final Value<bool?> poshanComplete;
  final Value<String?> measurementMode;
  final Value<String?> oedema;
  final Value<DateTime?> measuredAt;
  final Value<int?> editorUserId;
  final Value<String?> measuredNotes;
  final Value<String?> whoAcuteStatus;
  final Value<String?> whoAcuteTriggeredBy;
  final Value<String?> whoAcuteRationale;
  const MeasurementsCompanion({
    this.id = const Value.absent(),
    this.visitId = const Value.absent(),
    this.predictedHeightCm = const Value.absent(),
    this.predictedWeightKg = const Value.absent(),
    this.manualHeightCm = const Value.absent(),
    this.manualWeightKg = const Value.absent(),
    this.effectiveHeightCm = const Value.absent(),
    this.effectiveWeightKg = const Value.absent(),
    this.heightMethod = const Value.absent(),
    this.weightMethod = const Value.absent(),
    this.bmi = const Value.absent(),
    this.bmiStatus = const Value.absent(),
    this.hazZscore = const Value.absent(),
    this.whzZscore = const Value.absent(),
    this.hazStatus = const Value.absent(),
    this.whzStatus = const Value.absent(),
    this.confidenceScore = const Value.absent(),
    this.heightConfidence = const Value.absent(),
    this.weightConfidence = const Value.absent(),
    this.classificationConfidence = const Value.absent(),
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
    this.wastingMethod = const Value.absent(),
    this.muacCm = const Value.absent(),
    this.muacStatus = const Value.absent(),
    this.muacMethod = const Value.absent(),
    this.muacAgeInRange = const Value.absent(),
    this.muacConfidence = const Value.absent(),
    this.muacUncertaintyLowerCm = const Value.absent(),
    this.muacUncertaintyUpperCm = const Value.absent(),
    this.muacModelVersion = const Value.absent(),
    this.muacCalibrationVersion = const Value.absent(),
    this.muacIsDirectMeasurement = const Value.absent(),
    this.muacRequiresConfirmation = const Value.absent(),
    this.muacReferralGuidance = const Value.absent(),
    this.combinedStatus = const Value.absent(),
    this.combinedTriggeredBy = const Value.absent(),
    this.combinedRationale = const Value.absent(),
    this.combinedMethod = const Value.absent(),
    this.combinedConfidenceScore = const Value.absent(),
    this.combinedProtocolVersion = const Value.absent(),
    this.poshanStatus = const Value.absent(),
    this.poshanTriggeredBy = const Value.absent(),
    this.classificationMethod = const Value.absent(),
    this.classificationRationale = const Value.absent(),
    this.poshanComplete = const Value.absent(),
    this.measurementMode = const Value.absent(),
    this.oedema = const Value.absent(),
    this.measuredAt = const Value.absent(),
    this.editorUserId = const Value.absent(),
    this.measuredNotes = const Value.absent(),
    this.whoAcuteStatus = const Value.absent(),
    this.whoAcuteTriggeredBy = const Value.absent(),
    this.whoAcuteRationale = const Value.absent(),
  });
  MeasurementsCompanion.insert({
    this.id = const Value.absent(),
    required int visitId,
    this.predictedHeightCm = const Value.absent(),
    this.predictedWeightKg = const Value.absent(),
    this.manualHeightCm = const Value.absent(),
    this.manualWeightKg = const Value.absent(),
    this.effectiveHeightCm = const Value.absent(),
    this.effectiveWeightKg = const Value.absent(),
    this.heightMethod = const Value.absent(),
    this.weightMethod = const Value.absent(),
    this.bmi = const Value.absent(),
    this.bmiStatus = const Value.absent(),
    this.hazZscore = const Value.absent(),
    this.whzZscore = const Value.absent(),
    this.hazStatus = const Value.absent(),
    this.whzStatus = const Value.absent(),
    this.confidenceScore = const Value.absent(),
    this.heightConfidence = const Value.absent(),
    this.weightConfidence = const Value.absent(),
    this.classificationConfidence = const Value.absent(),
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
    this.wastingMethod = const Value.absent(),
    this.muacCm = const Value.absent(),
    this.muacStatus = const Value.absent(),
    this.muacMethod = const Value.absent(),
    this.muacAgeInRange = const Value.absent(),
    this.muacConfidence = const Value.absent(),
    this.muacUncertaintyLowerCm = const Value.absent(),
    this.muacUncertaintyUpperCm = const Value.absent(),
    this.muacModelVersion = const Value.absent(),
    this.muacCalibrationVersion = const Value.absent(),
    this.muacIsDirectMeasurement = const Value.absent(),
    this.muacRequiresConfirmation = const Value.absent(),
    this.muacReferralGuidance = const Value.absent(),
    this.combinedStatus = const Value.absent(),
    this.combinedTriggeredBy = const Value.absent(),
    this.combinedRationale = const Value.absent(),
    this.combinedMethod = const Value.absent(),
    this.combinedConfidenceScore = const Value.absent(),
    this.combinedProtocolVersion = const Value.absent(),
    this.poshanStatus = const Value.absent(),
    this.poshanTriggeredBy = const Value.absent(),
    this.classificationMethod = const Value.absent(),
    this.classificationRationale = const Value.absent(),
    this.poshanComplete = const Value.absent(),
    this.measurementMode = const Value.absent(),
    this.oedema = const Value.absent(),
    this.measuredAt = const Value.absent(),
    this.editorUserId = const Value.absent(),
    this.measuredNotes = const Value.absent(),
    this.whoAcuteStatus = const Value.absent(),
    this.whoAcuteTriggeredBy = const Value.absent(),
    this.whoAcuteRationale = const Value.absent(),
  }) : visitId = Value(visitId);
  static Insertable<Measurement> custom({
    Expression<int>? id,
    Expression<int>? visitId,
    Expression<double>? predictedHeightCm,
    Expression<double>? predictedWeightKg,
    Expression<double>? manualHeightCm,
    Expression<double>? manualWeightKg,
    Expression<double>? effectiveHeightCm,
    Expression<double>? effectiveWeightKg,
    Expression<String>? heightMethod,
    Expression<String>? weightMethod,
    Expression<double>? bmi,
    Expression<String>? bmiStatus,
    Expression<double>? hazZscore,
    Expression<double>? whzZscore,
    Expression<String>? hazStatus,
    Expression<String>? whzStatus,
    Expression<double>? confidenceScore,
    Expression<double>? heightConfidence,
    Expression<double>? weightConfidence,
    Expression<double>? classificationConfidence,
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
    Expression<String>? wastingMethod,
    Expression<double>? muacCm,
    Expression<String>? muacStatus,
    Expression<String>? muacMethod,
    Expression<bool>? muacAgeInRange,
    Expression<double>? muacConfidence,
    Expression<double>? muacUncertaintyLowerCm,
    Expression<double>? muacUncertaintyUpperCm,
    Expression<String>? muacModelVersion,
    Expression<String>? muacCalibrationVersion,
    Expression<bool>? muacIsDirectMeasurement,
    Expression<bool>? muacRequiresConfirmation,
    Expression<String>? muacReferralGuidance,
    Expression<String>? combinedStatus,
    Expression<String>? combinedTriggeredBy,
    Expression<String>? combinedRationale,
    Expression<String>? combinedMethod,
    Expression<double>? combinedConfidenceScore,
    Expression<String>? combinedProtocolVersion,
    Expression<String>? poshanStatus,
    Expression<String>? poshanTriggeredBy,
    Expression<String>? classificationMethod,
    Expression<String>? classificationRationale,
    Expression<bool>? poshanComplete,
    Expression<String>? measurementMode,
    Expression<String>? oedema,
    Expression<DateTime>? measuredAt,
    Expression<int>? editorUserId,
    Expression<String>? measuredNotes,
    Expression<String>? whoAcuteStatus,
    Expression<String>? whoAcuteTriggeredBy,
    Expression<String>? whoAcuteRationale,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (visitId != null) 'visit_id': visitId,
      if (predictedHeightCm != null) 'predicted_height_cm': predictedHeightCm,
      if (predictedWeightKg != null) 'predicted_weight_kg': predictedWeightKg,
      if (manualHeightCm != null) 'manual_height_cm': manualHeightCm,
      if (manualWeightKg != null) 'manual_weight_kg': manualWeightKg,
      if (effectiveHeightCm != null) 'effective_height_cm': effectiveHeightCm,
      if (effectiveWeightKg != null) 'effective_weight_kg': effectiveWeightKg,
      if (heightMethod != null) 'height_method': heightMethod,
      if (weightMethod != null) 'weight_method': weightMethod,
      if (bmi != null) 'bmi': bmi,
      if (bmiStatus != null) 'bmi_status': bmiStatus,
      if (hazZscore != null) 'haz_zscore': hazZscore,
      if (whzZscore != null) 'whz_zscore': whzZscore,
      if (hazStatus != null) 'haz_status': hazStatus,
      if (whzStatus != null) 'whz_status': whzStatus,
      if (confidenceScore != null) 'confidence_score': confidenceScore,
      if (heightConfidence != null) 'height_confidence': heightConfidence,
      if (weightConfidence != null) 'weight_confidence': weightConfidence,
      if (classificationConfidence != null)
        'classification_confidence': classificationConfidence,
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
      if (wastingMethod != null) 'wasting_method': wastingMethod,
      if (muacCm != null) 'muac_cm': muacCm,
      if (muacStatus != null) 'muac_status': muacStatus,
      if (muacMethod != null) 'muac_method': muacMethod,
      if (muacAgeInRange != null) 'muac_age_in_range': muacAgeInRange,
      if (muacConfidence != null) 'muac_confidence': muacConfidence,
      if (muacUncertaintyLowerCm != null)
        'muac_uncertainty_lower_cm': muacUncertaintyLowerCm,
      if (muacUncertaintyUpperCm != null)
        'muac_uncertainty_upper_cm': muacUncertaintyUpperCm,
      if (muacModelVersion != null) 'muac_model_version': muacModelVersion,
      if (muacCalibrationVersion != null)
        'muac_calibration_version': muacCalibrationVersion,
      if (muacIsDirectMeasurement != null)
        'muac_is_direct_measurement': muacIsDirectMeasurement,
      if (muacRequiresConfirmation != null)
        'muac_requires_confirmation': muacRequiresConfirmation,
      if (muacReferralGuidance != null)
        'muac_referral_guidance': muacReferralGuidance,
      if (combinedStatus != null) 'combined_status': combinedStatus,
      if (combinedTriggeredBy != null)
        'combined_triggered_by': combinedTriggeredBy,
      if (combinedRationale != null) 'combined_rationale': combinedRationale,
      if (combinedMethod != null) 'combined_method': combinedMethod,
      if (combinedConfidenceScore != null)
        'combined_confidence_score': combinedConfidenceScore,
      if (combinedProtocolVersion != null)
        'combined_protocol_version': combinedProtocolVersion,
      if (poshanStatus != null) 'poshan_status': poshanStatus,
      if (poshanTriggeredBy != null) 'poshan_triggered_by': poshanTriggeredBy,
      if (classificationMethod != null)
        'classification_method': classificationMethod,
      if (classificationRationale != null)
        'classification_rationale': classificationRationale,
      if (poshanComplete != null) 'poshan_complete': poshanComplete,
      if (measurementMode != null) 'measurement_mode': measurementMode,
      if (oedema != null) 'oedema': oedema,
      if (measuredAt != null) 'measured_at': measuredAt,
      if (editorUserId != null) 'editor_user_id': editorUserId,
      if (measuredNotes != null) 'measured_notes': measuredNotes,
      if (whoAcuteStatus != null) 'who_acute_status': whoAcuteStatus,
      if (whoAcuteTriggeredBy != null)
        'who_acute_triggered_by': whoAcuteTriggeredBy,
      if (whoAcuteRationale != null) 'who_acute_rationale': whoAcuteRationale,
    });
  }

  MeasurementsCompanion copyWith(
      {Value<int>? id,
      Value<int>? visitId,
      Value<double?>? predictedHeightCm,
      Value<double?>? predictedWeightKg,
      Value<double?>? manualHeightCm,
      Value<double?>? manualWeightKg,
      Value<double?>? effectiveHeightCm,
      Value<double?>? effectiveWeightKg,
      Value<String?>? heightMethod,
      Value<String?>? weightMethod,
      Value<double?>? bmi,
      Value<String?>? bmiStatus,
      Value<double?>? hazZscore,
      Value<double?>? whzZscore,
      Value<String?>? hazStatus,
      Value<String?>? whzStatus,
      Value<double?>? confidenceScore,
      Value<double?>? heightConfidence,
      Value<double?>? weightConfidence,
      Value<double?>? classificationConfidence,
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
      Value<String?>? wastingMethod,
      Value<double?>? muacCm,
      Value<String?>? muacStatus,
      Value<String?>? muacMethod,
      Value<bool?>? muacAgeInRange,
      Value<double?>? muacConfidence,
      Value<double?>? muacUncertaintyLowerCm,
      Value<double?>? muacUncertaintyUpperCm,
      Value<String?>? muacModelVersion,
      Value<String?>? muacCalibrationVersion,
      Value<bool?>? muacIsDirectMeasurement,
      Value<bool?>? muacRequiresConfirmation,
      Value<String?>? muacReferralGuidance,
      Value<String?>? combinedStatus,
      Value<String?>? combinedTriggeredBy,
      Value<String?>? combinedRationale,
      Value<String?>? combinedMethod,
      Value<double?>? combinedConfidenceScore,
      Value<String?>? combinedProtocolVersion,
      Value<String?>? poshanStatus,
      Value<String?>? poshanTriggeredBy,
      Value<String?>? classificationMethod,
      Value<String?>? classificationRationale,
      Value<bool?>? poshanComplete,
      Value<String?>? measurementMode,
      Value<String?>? oedema,
      Value<DateTime?>? measuredAt,
      Value<int?>? editorUserId,
      Value<String?>? measuredNotes,
      Value<String?>? whoAcuteStatus,
      Value<String?>? whoAcuteTriggeredBy,
      Value<String?>? whoAcuteRationale}) {
    return MeasurementsCompanion(
      id: id ?? this.id,
      visitId: visitId ?? this.visitId,
      predictedHeightCm: predictedHeightCm ?? this.predictedHeightCm,
      predictedWeightKg: predictedWeightKg ?? this.predictedWeightKg,
      manualHeightCm: manualHeightCm ?? this.manualHeightCm,
      manualWeightKg: manualWeightKg ?? this.manualWeightKg,
      effectiveHeightCm: effectiveHeightCm ?? this.effectiveHeightCm,
      effectiveWeightKg: effectiveWeightKg ?? this.effectiveWeightKg,
      heightMethod: heightMethod ?? this.heightMethod,
      weightMethod: weightMethod ?? this.weightMethod,
      bmi: bmi ?? this.bmi,
      bmiStatus: bmiStatus ?? this.bmiStatus,
      hazZscore: hazZscore ?? this.hazZscore,
      whzZscore: whzZscore ?? this.whzZscore,
      hazStatus: hazStatus ?? this.hazStatus,
      whzStatus: whzStatus ?? this.whzStatus,
      confidenceScore: confidenceScore ?? this.confidenceScore,
      heightConfidence: heightConfidence ?? this.heightConfidence,
      weightConfidence: weightConfidence ?? this.weightConfidence,
      classificationConfidence:
          classificationConfidence ?? this.classificationConfidence,
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
      wastingMethod: wastingMethod ?? this.wastingMethod,
      muacCm: muacCm ?? this.muacCm,
      muacStatus: muacStatus ?? this.muacStatus,
      muacMethod: muacMethod ?? this.muacMethod,
      muacAgeInRange: muacAgeInRange ?? this.muacAgeInRange,
      muacConfidence: muacConfidence ?? this.muacConfidence,
      muacUncertaintyLowerCm:
          muacUncertaintyLowerCm ?? this.muacUncertaintyLowerCm,
      muacUncertaintyUpperCm:
          muacUncertaintyUpperCm ?? this.muacUncertaintyUpperCm,
      muacModelVersion: muacModelVersion ?? this.muacModelVersion,
      muacCalibrationVersion:
          muacCalibrationVersion ?? this.muacCalibrationVersion,
      muacIsDirectMeasurement:
          muacIsDirectMeasurement ?? this.muacIsDirectMeasurement,
      muacRequiresConfirmation:
          muacRequiresConfirmation ?? this.muacRequiresConfirmation,
      muacReferralGuidance: muacReferralGuidance ?? this.muacReferralGuidance,
      combinedStatus: combinedStatus ?? this.combinedStatus,
      combinedTriggeredBy: combinedTriggeredBy ?? this.combinedTriggeredBy,
      combinedRationale: combinedRationale ?? this.combinedRationale,
      combinedMethod: combinedMethod ?? this.combinedMethod,
      combinedConfidenceScore:
          combinedConfidenceScore ?? this.combinedConfidenceScore,
      combinedProtocolVersion:
          combinedProtocolVersion ?? this.combinedProtocolVersion,
      poshanStatus: poshanStatus ?? this.poshanStatus,
      poshanTriggeredBy: poshanTriggeredBy ?? this.poshanTriggeredBy,
      classificationMethod: classificationMethod ?? this.classificationMethod,
      classificationRationale:
          classificationRationale ?? this.classificationRationale,
      poshanComplete: poshanComplete ?? this.poshanComplete,
      measurementMode: measurementMode ?? this.measurementMode,
      oedema: oedema ?? this.oedema,
      measuredAt: measuredAt ?? this.measuredAt,
      editorUserId: editorUserId ?? this.editorUserId,
      measuredNotes: measuredNotes ?? this.measuredNotes,
      whoAcuteStatus: whoAcuteStatus ?? this.whoAcuteStatus,
      whoAcuteTriggeredBy: whoAcuteTriggeredBy ?? this.whoAcuteTriggeredBy,
      whoAcuteRationale: whoAcuteRationale ?? this.whoAcuteRationale,
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
    if (effectiveHeightCm.present) {
      map['effective_height_cm'] = Variable<double>(effectiveHeightCm.value);
    }
    if (effectiveWeightKg.present) {
      map['effective_weight_kg'] = Variable<double>(effectiveWeightKg.value);
    }
    if (heightMethod.present) {
      map['height_method'] = Variable<String>(heightMethod.value);
    }
    if (weightMethod.present) {
      map['weight_method'] = Variable<String>(weightMethod.value);
    }
    if (bmi.present) {
      map['bmi'] = Variable<double>(bmi.value);
    }
    if (bmiStatus.present) {
      map['bmi_status'] = Variable<String>(bmiStatus.value);
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
    if (heightConfidence.present) {
      map['height_confidence'] = Variable<double>(heightConfidence.value);
    }
    if (weightConfidence.present) {
      map['weight_confidence'] = Variable<double>(weightConfidence.value);
    }
    if (classificationConfidence.present) {
      map['classification_confidence'] =
          Variable<double>(classificationConfidence.value);
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
    if (wastingMethod.present) {
      map['wasting_method'] = Variable<String>(wastingMethod.value);
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
    if (muacAgeInRange.present) {
      map['muac_age_in_range'] = Variable<bool>(muacAgeInRange.value);
    }
    if (muacConfidence.present) {
      map['muac_confidence'] = Variable<double>(muacConfidence.value);
    }
    if (muacUncertaintyLowerCm.present) {
      map['muac_uncertainty_lower_cm'] =
          Variable<double>(muacUncertaintyLowerCm.value);
    }
    if (muacUncertaintyUpperCm.present) {
      map['muac_uncertainty_upper_cm'] =
          Variable<double>(muacUncertaintyUpperCm.value);
    }
    if (muacModelVersion.present) {
      map['muac_model_version'] = Variable<String>(muacModelVersion.value);
    }
    if (muacCalibrationVersion.present) {
      map['muac_calibration_version'] =
          Variable<String>(muacCalibrationVersion.value);
    }
    if (muacIsDirectMeasurement.present) {
      map['muac_is_direct_measurement'] =
          Variable<bool>(muacIsDirectMeasurement.value);
    }
    if (muacRequiresConfirmation.present) {
      map['muac_requires_confirmation'] =
          Variable<bool>(muacRequiresConfirmation.value);
    }
    if (muacReferralGuidance.present) {
      map['muac_referral_guidance'] =
          Variable<String>(muacReferralGuidance.value);
    }
    if (combinedStatus.present) {
      map['combined_status'] = Variable<String>(combinedStatus.value);
    }
    if (combinedTriggeredBy.present) {
      map['combined_triggered_by'] =
          Variable<String>(combinedTriggeredBy.value);
    }
    if (combinedRationale.present) {
      map['combined_rationale'] = Variable<String>(combinedRationale.value);
    }
    if (combinedMethod.present) {
      map['combined_method'] = Variable<String>(combinedMethod.value);
    }
    if (combinedConfidenceScore.present) {
      map['combined_confidence_score'] =
          Variable<double>(combinedConfidenceScore.value);
    }
    if (combinedProtocolVersion.present) {
      map['combined_protocol_version'] =
          Variable<String>(combinedProtocolVersion.value);
    }
    if (poshanStatus.present) {
      map['poshan_status'] = Variable<String>(poshanStatus.value);
    }
    if (poshanTriggeredBy.present) {
      map['poshan_triggered_by'] = Variable<String>(poshanTriggeredBy.value);
    }
    if (classificationMethod.present) {
      map['classification_method'] =
          Variable<String>(classificationMethod.value);
    }
    if (classificationRationale.present) {
      map['classification_rationale'] =
          Variable<String>(classificationRationale.value);
    }
    if (poshanComplete.present) {
      map['poshan_complete'] = Variable<bool>(poshanComplete.value);
    }
    if (measurementMode.present) {
      map['measurement_mode'] = Variable<String>(measurementMode.value);
    }
    if (oedema.present) {
      map['oedema'] = Variable<String>(oedema.value);
    }
    if (measuredAt.present) {
      map['measured_at'] = Variable<DateTime>(measuredAt.value);
    }
    if (editorUserId.present) {
      map['editor_user_id'] = Variable<int>(editorUserId.value);
    }
    if (measuredNotes.present) {
      map['measured_notes'] = Variable<String>(measuredNotes.value);
    }
    if (whoAcuteStatus.present) {
      map['who_acute_status'] = Variable<String>(whoAcuteStatus.value);
    }
    if (whoAcuteTriggeredBy.present) {
      map['who_acute_triggered_by'] =
          Variable<String>(whoAcuteTriggeredBy.value);
    }
    if (whoAcuteRationale.present) {
      map['who_acute_rationale'] = Variable<String>(whoAcuteRationale.value);
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
          ..write('effectiveHeightCm: $effectiveHeightCm, ')
          ..write('effectiveWeightKg: $effectiveWeightKg, ')
          ..write('heightMethod: $heightMethod, ')
          ..write('weightMethod: $weightMethod, ')
          ..write('bmi: $bmi, ')
          ..write('bmiStatus: $bmiStatus, ')
          ..write('hazZscore: $hazZscore, ')
          ..write('whzZscore: $whzZscore, ')
          ..write('hazStatus: $hazStatus, ')
          ..write('whzStatus: $whzStatus, ')
          ..write('confidenceScore: $confidenceScore, ')
          ..write('heightConfidence: $heightConfidence, ')
          ..write('weightConfidence: $weightConfidence, ')
          ..write('classificationConfidence: $classificationConfidence, ')
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
          ..write('wastingMethod: $wastingMethod, ')
          ..write('muacCm: $muacCm, ')
          ..write('muacStatus: $muacStatus, ')
          ..write('muacMethod: $muacMethod, ')
          ..write('muacAgeInRange: $muacAgeInRange, ')
          ..write('muacConfidence: $muacConfidence, ')
          ..write('muacUncertaintyLowerCm: $muacUncertaintyLowerCm, ')
          ..write('muacUncertaintyUpperCm: $muacUncertaintyUpperCm, ')
          ..write('muacModelVersion: $muacModelVersion, ')
          ..write('muacCalibrationVersion: $muacCalibrationVersion, ')
          ..write('muacIsDirectMeasurement: $muacIsDirectMeasurement, ')
          ..write('muacRequiresConfirmation: $muacRequiresConfirmation, ')
          ..write('muacReferralGuidance: $muacReferralGuidance, ')
          ..write('combinedStatus: $combinedStatus, ')
          ..write('combinedTriggeredBy: $combinedTriggeredBy, ')
          ..write('combinedRationale: $combinedRationale, ')
          ..write('combinedMethod: $combinedMethod, ')
          ..write('combinedConfidenceScore: $combinedConfidenceScore, ')
          ..write('combinedProtocolVersion: $combinedProtocolVersion, ')
          ..write('poshanStatus: $poshanStatus, ')
          ..write('poshanTriggeredBy: $poshanTriggeredBy, ')
          ..write('classificationMethod: $classificationMethod, ')
          ..write('classificationRationale: $classificationRationale, ')
          ..write('poshanComplete: $poshanComplete, ')
          ..write('measurementMode: $measurementMode, ')
          ..write('oedema: $oedema, ')
          ..write('measuredAt: $measuredAt, ')
          ..write('editorUserId: $editorUserId, ')
          ..write('measuredNotes: $measuredNotes, ')
          ..write('whoAcuteStatus: $whoAcuteStatus, ')
          ..write('whoAcuteTriggeredBy: $whoAcuteTriggeredBy, ')
          ..write('whoAcuteRationale: $whoAcuteRationale')
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

class $CaptureAssetsTable extends CaptureAssets
    with TableInfo<$CaptureAssetsTable, CaptureAsset> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $CaptureAssetsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _assetUuidMeta =
      const VerificationMeta('assetUuid');
  @override
  late final GeneratedColumn<String> assetUuid = GeneratedColumn<String>(
      'asset_uuid', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways('UNIQUE'));
  static const VerificationMeta _visitIdMeta =
      const VerificationMeta('visitId');
  @override
  late final GeneratedColumn<int> visitId = GeneratedColumn<int>(
      'visit_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'REFERENCES visits (id) ON DELETE CASCADE'));
  static const VerificationMeta _roleMeta = const VerificationMeta('role');
  @override
  late final GeneratedColumn<String> role = GeneratedColumn<String>(
      'role', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _localPathMeta =
      const VerificationMeta('localPath');
  @override
  late final GeneratedColumn<String> localPath = GeneratedColumn<String>(
      'local_path', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _serverIdMeta =
      const VerificationMeta('serverId');
  @override
  late final GeneratedColumn<int> serverId = GeneratedColumn<int>(
      'server_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _serverObjectIdMeta =
      const VerificationMeta('serverObjectId');
  @override
  late final GeneratedColumn<String> serverObjectId = GeneratedColumn<String>(
      'server_object_id', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _capturedAtMeta =
      const VerificationMeta('capturedAt');
  @override
  late final GeneratedColumn<DateTime> capturedAt = GeneratedColumn<DateTime>(
      'captured_at', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  static const VerificationMeta _selectedRankMeta =
      const VerificationMeta('selectedRank');
  @override
  late final GeneratedColumn<int> selectedRank = GeneratedColumn<int>(
      'selected_rank', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _poseScoreMeta =
      const VerificationMeta('poseScore');
  @override
  late final GeneratedColumn<double> poseScore = GeneratedColumn<double>(
      'pose_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _coverageScoreMeta =
      const VerificationMeta('coverageScore');
  @override
  late final GeneratedColumn<double> coverageScore = GeneratedColumn<double>(
      'coverage_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _orientationScoreMeta =
      const VerificationMeta('orientationScore');
  @override
  late final GeneratedColumn<double> orientationScore = GeneratedColumn<double>(
      'orientation_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _sharpnessScoreMeta =
      const VerificationMeta('sharpnessScore');
  @override
  late final GeneratedColumn<double> sharpnessScore = GeneratedColumn<double>(
      'sharpness_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _lightingScoreMeta =
      const VerificationMeta('lightingScore');
  @override
  late final GeneratedColumn<double> lightingScore = GeneratedColumn<double>(
      'lighting_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _overallScoreMeta =
      const VerificationMeta('overallScore');
  @override
  late final GeneratedColumn<double> overallScore = GeneratedColumn<double>(
      'overall_score', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _qualityVerdictMeta =
      const VerificationMeta('qualityVerdict');
  @override
  late final GeneratedColumn<String> qualityVerdict = GeneratedColumn<String>(
      'quality_verdict', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _rejectionReasonMeta =
      const VerificationMeta('rejectionReason');
  @override
  late final GeneratedColumn<String> rejectionReason = GeneratedColumn<String>(
      'rejection_reason', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _qualityThresholdVersionMeta =
      const VerificationMeta('qualityThresholdVersion');
  @override
  late final GeneratedColumn<String> qualityThresholdVersion =
      GeneratedColumn<String>('quality_threshold_version', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _imageWidthMeta =
      const VerificationMeta('imageWidth');
  @override
  late final GeneratedColumn<int> imageWidth = GeneratedColumn<int>(
      'image_width', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _imageHeightMeta =
      const VerificationMeta('imageHeight');
  @override
  late final GeneratedColumn<int> imageHeight = GeneratedColumn<int>(
      'image_height', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _exifOrientationMeta =
      const VerificationMeta('exifOrientation');
  @override
  late final GeneratedColumn<int> exifOrientation = GeneratedColumn<int>(
      'exif_orientation', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _displayOrientationMeta =
      const VerificationMeta('displayOrientation');
  @override
  late final GeneratedColumn<int> displayOrientation = GeneratedColumn<int>(
      'display_orientation', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _deviceCameraMetadataJsonMeta =
      const VerificationMeta('deviceCameraMetadataJson');
  @override
  late final GeneratedColumn<String> deviceCameraMetadataJson =
      GeneratedColumn<String>('device_camera_metadata_json', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _syncStateMeta =
      const VerificationMeta('syncState');
  @override
  late final GeneratedColumn<String> syncState = GeneratedColumn<String>(
      'sync_state', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant('pending'));
  static const VerificationMeta _serverAcknowledgedAtMeta =
      const VerificationMeta('serverAcknowledgedAt');
  @override
  late final GeneratedColumn<DateTime> serverAcknowledgedAt =
      GeneratedColumn<DateTime>('server_acknowledged_at', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        assetUuid,
        visitId,
        role,
        localPath,
        serverId,
        serverObjectId,
        capturedAt,
        selectedRank,
        poseScore,
        coverageScore,
        orientationScore,
        sharpnessScore,
        lightingScore,
        overallScore,
        qualityVerdict,
        rejectionReason,
        qualityThresholdVersion,
        imageWidth,
        imageHeight,
        exifOrientation,
        displayOrientation,
        deviceCameraMetadataJson,
        syncState,
        serverAcknowledgedAt
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'capture_assets';
  @override
  VerificationContext validateIntegrity(Insertable<CaptureAsset> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('asset_uuid')) {
      context.handle(_assetUuidMeta,
          assetUuid.isAcceptableOrUnknown(data['asset_uuid']!, _assetUuidMeta));
    } else if (isInserting) {
      context.missing(_assetUuidMeta);
    }
    if (data.containsKey('visit_id')) {
      context.handle(_visitIdMeta,
          visitId.isAcceptableOrUnknown(data['visit_id']!, _visitIdMeta));
    } else if (isInserting) {
      context.missing(_visitIdMeta);
    }
    if (data.containsKey('role')) {
      context.handle(
          _roleMeta, role.isAcceptableOrUnknown(data['role']!, _roleMeta));
    } else if (isInserting) {
      context.missing(_roleMeta);
    }
    if (data.containsKey('local_path')) {
      context.handle(_localPathMeta,
          localPath.isAcceptableOrUnknown(data['local_path']!, _localPathMeta));
    }
    if (data.containsKey('server_id')) {
      context.handle(_serverIdMeta,
          serverId.isAcceptableOrUnknown(data['server_id']!, _serverIdMeta));
    }
    if (data.containsKey('server_object_id')) {
      context.handle(
          _serverObjectIdMeta,
          serverObjectId.isAcceptableOrUnknown(
              data['server_object_id']!, _serverObjectIdMeta));
    }
    if (data.containsKey('captured_at')) {
      context.handle(
          _capturedAtMeta,
          capturedAt.isAcceptableOrUnknown(
              data['captured_at']!, _capturedAtMeta));
    } else if (isInserting) {
      context.missing(_capturedAtMeta);
    }
    if (data.containsKey('selected_rank')) {
      context.handle(
          _selectedRankMeta,
          selectedRank.isAcceptableOrUnknown(
              data['selected_rank']!, _selectedRankMeta));
    }
    if (data.containsKey('pose_score')) {
      context.handle(_poseScoreMeta,
          poseScore.isAcceptableOrUnknown(data['pose_score']!, _poseScoreMeta));
    }
    if (data.containsKey('coverage_score')) {
      context.handle(
          _coverageScoreMeta,
          coverageScore.isAcceptableOrUnknown(
              data['coverage_score']!, _coverageScoreMeta));
    }
    if (data.containsKey('orientation_score')) {
      context.handle(
          _orientationScoreMeta,
          orientationScore.isAcceptableOrUnknown(
              data['orientation_score']!, _orientationScoreMeta));
    }
    if (data.containsKey('sharpness_score')) {
      context.handle(
          _sharpnessScoreMeta,
          sharpnessScore.isAcceptableOrUnknown(
              data['sharpness_score']!, _sharpnessScoreMeta));
    }
    if (data.containsKey('lighting_score')) {
      context.handle(
          _lightingScoreMeta,
          lightingScore.isAcceptableOrUnknown(
              data['lighting_score']!, _lightingScoreMeta));
    }
    if (data.containsKey('overall_score')) {
      context.handle(
          _overallScoreMeta,
          overallScore.isAcceptableOrUnknown(
              data['overall_score']!, _overallScoreMeta));
    }
    if (data.containsKey('quality_verdict')) {
      context.handle(
          _qualityVerdictMeta,
          qualityVerdict.isAcceptableOrUnknown(
              data['quality_verdict']!, _qualityVerdictMeta));
    }
    if (data.containsKey('rejection_reason')) {
      context.handle(
          _rejectionReasonMeta,
          rejectionReason.isAcceptableOrUnknown(
              data['rejection_reason']!, _rejectionReasonMeta));
    }
    if (data.containsKey('quality_threshold_version')) {
      context.handle(
          _qualityThresholdVersionMeta,
          qualityThresholdVersion.isAcceptableOrUnknown(
              data['quality_threshold_version']!,
              _qualityThresholdVersionMeta));
    }
    if (data.containsKey('image_width')) {
      context.handle(
          _imageWidthMeta,
          imageWidth.isAcceptableOrUnknown(
              data['image_width']!, _imageWidthMeta));
    }
    if (data.containsKey('image_height')) {
      context.handle(
          _imageHeightMeta,
          imageHeight.isAcceptableOrUnknown(
              data['image_height']!, _imageHeightMeta));
    }
    if (data.containsKey('exif_orientation')) {
      context.handle(
          _exifOrientationMeta,
          exifOrientation.isAcceptableOrUnknown(
              data['exif_orientation']!, _exifOrientationMeta));
    }
    if (data.containsKey('display_orientation')) {
      context.handle(
          _displayOrientationMeta,
          displayOrientation.isAcceptableOrUnknown(
              data['display_orientation']!, _displayOrientationMeta));
    }
    if (data.containsKey('device_camera_metadata_json')) {
      context.handle(
          _deviceCameraMetadataJsonMeta,
          deviceCameraMetadataJson.isAcceptableOrUnknown(
              data['device_camera_metadata_json']!,
              _deviceCameraMetadataJsonMeta));
    }
    if (data.containsKey('sync_state')) {
      context.handle(_syncStateMeta,
          syncState.isAcceptableOrUnknown(data['sync_state']!, _syncStateMeta));
    }
    if (data.containsKey('server_acknowledged_at')) {
      context.handle(
          _serverAcknowledgedAtMeta,
          serverAcknowledgedAt.isAcceptableOrUnknown(
              data['server_acknowledged_at']!, _serverAcknowledgedAtMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  CaptureAsset map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return CaptureAsset(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      assetUuid: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}asset_uuid'])!,
      visitId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}visit_id'])!,
      role: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}role'])!,
      localPath: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}local_path']),
      serverId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}server_id']),
      serverObjectId: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}server_object_id']),
      capturedAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}captured_at'])!,
      selectedRank: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}selected_rank']),
      poseScore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}pose_score']),
      coverageScore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}coverage_score']),
      orientationScore: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}orientation_score']),
      sharpnessScore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}sharpness_score']),
      lightingScore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}lighting_score']),
      overallScore: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}overall_score']),
      qualityVerdict: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}quality_verdict']),
      rejectionReason: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}rejection_reason']),
      qualityThresholdVersion: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}quality_threshold_version']),
      imageWidth: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}image_width']),
      imageHeight: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}image_height']),
      exifOrientation: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}exif_orientation']),
      displayOrientation: attachedDatabase.typeMapping.read(
          DriftSqlType.int, data['${effectivePrefix}display_orientation']),
      deviceCameraMetadataJson: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}device_camera_metadata_json']),
      syncState: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}sync_state'])!,
      serverAcknowledgedAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime,
          data['${effectivePrefix}server_acknowledged_at']),
    );
  }

  @override
  $CaptureAssetsTable createAlias(String alias) {
    return $CaptureAssetsTable(attachedDatabase, alias);
  }
}

class CaptureAsset extends DataClass implements Insertable<CaptureAsset> {
  final int id;
  final String assetUuid;
  final int visitId;
  final String role;
  final String? localPath;
  final int? serverId;
  final String? serverObjectId;
  final DateTime capturedAt;
  final int? selectedRank;
  final double? poseScore;
  final double? coverageScore;
  final double? orientationScore;
  final double? sharpnessScore;
  final double? lightingScore;
  final double? overallScore;
  final String? qualityVerdict;
  final String? rejectionReason;
  final String? qualityThresholdVersion;
  final int? imageWidth;
  final int? imageHeight;
  final int? exifOrientation;
  final int? displayOrientation;
  final String? deviceCameraMetadataJson;
  final String syncState;
  final DateTime? serverAcknowledgedAt;
  const CaptureAsset(
      {required this.id,
      required this.assetUuid,
      required this.visitId,
      required this.role,
      this.localPath,
      this.serverId,
      this.serverObjectId,
      required this.capturedAt,
      this.selectedRank,
      this.poseScore,
      this.coverageScore,
      this.orientationScore,
      this.sharpnessScore,
      this.lightingScore,
      this.overallScore,
      this.qualityVerdict,
      this.rejectionReason,
      this.qualityThresholdVersion,
      this.imageWidth,
      this.imageHeight,
      this.exifOrientation,
      this.displayOrientation,
      this.deviceCameraMetadataJson,
      required this.syncState,
      this.serverAcknowledgedAt});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['asset_uuid'] = Variable<String>(assetUuid);
    map['visit_id'] = Variable<int>(visitId);
    map['role'] = Variable<String>(role);
    if (!nullToAbsent || localPath != null) {
      map['local_path'] = Variable<String>(localPath);
    }
    if (!nullToAbsent || serverId != null) {
      map['server_id'] = Variable<int>(serverId);
    }
    if (!nullToAbsent || serverObjectId != null) {
      map['server_object_id'] = Variable<String>(serverObjectId);
    }
    map['captured_at'] = Variable<DateTime>(capturedAt);
    if (!nullToAbsent || selectedRank != null) {
      map['selected_rank'] = Variable<int>(selectedRank);
    }
    if (!nullToAbsent || poseScore != null) {
      map['pose_score'] = Variable<double>(poseScore);
    }
    if (!nullToAbsent || coverageScore != null) {
      map['coverage_score'] = Variable<double>(coverageScore);
    }
    if (!nullToAbsent || orientationScore != null) {
      map['orientation_score'] = Variable<double>(orientationScore);
    }
    if (!nullToAbsent || sharpnessScore != null) {
      map['sharpness_score'] = Variable<double>(sharpnessScore);
    }
    if (!nullToAbsent || lightingScore != null) {
      map['lighting_score'] = Variable<double>(lightingScore);
    }
    if (!nullToAbsent || overallScore != null) {
      map['overall_score'] = Variable<double>(overallScore);
    }
    if (!nullToAbsent || qualityVerdict != null) {
      map['quality_verdict'] = Variable<String>(qualityVerdict);
    }
    if (!nullToAbsent || rejectionReason != null) {
      map['rejection_reason'] = Variable<String>(rejectionReason);
    }
    if (!nullToAbsent || qualityThresholdVersion != null) {
      map['quality_threshold_version'] =
          Variable<String>(qualityThresholdVersion);
    }
    if (!nullToAbsent || imageWidth != null) {
      map['image_width'] = Variable<int>(imageWidth);
    }
    if (!nullToAbsent || imageHeight != null) {
      map['image_height'] = Variable<int>(imageHeight);
    }
    if (!nullToAbsent || exifOrientation != null) {
      map['exif_orientation'] = Variable<int>(exifOrientation);
    }
    if (!nullToAbsent || displayOrientation != null) {
      map['display_orientation'] = Variable<int>(displayOrientation);
    }
    if (!nullToAbsent || deviceCameraMetadataJson != null) {
      map['device_camera_metadata_json'] =
          Variable<String>(deviceCameraMetadataJson);
    }
    map['sync_state'] = Variable<String>(syncState);
    if (!nullToAbsent || serverAcknowledgedAt != null) {
      map['server_acknowledged_at'] = Variable<DateTime>(serverAcknowledgedAt);
    }
    return map;
  }

  CaptureAssetsCompanion toCompanion(bool nullToAbsent) {
    return CaptureAssetsCompanion(
      id: Value(id),
      assetUuid: Value(assetUuid),
      visitId: Value(visitId),
      role: Value(role),
      localPath: localPath == null && nullToAbsent
          ? const Value.absent()
          : Value(localPath),
      serverId: serverId == null && nullToAbsent
          ? const Value.absent()
          : Value(serverId),
      serverObjectId: serverObjectId == null && nullToAbsent
          ? const Value.absent()
          : Value(serverObjectId),
      capturedAt: Value(capturedAt),
      selectedRank: selectedRank == null && nullToAbsent
          ? const Value.absent()
          : Value(selectedRank),
      poseScore: poseScore == null && nullToAbsent
          ? const Value.absent()
          : Value(poseScore),
      coverageScore: coverageScore == null && nullToAbsent
          ? const Value.absent()
          : Value(coverageScore),
      orientationScore: orientationScore == null && nullToAbsent
          ? const Value.absent()
          : Value(orientationScore),
      sharpnessScore: sharpnessScore == null && nullToAbsent
          ? const Value.absent()
          : Value(sharpnessScore),
      lightingScore: lightingScore == null && nullToAbsent
          ? const Value.absent()
          : Value(lightingScore),
      overallScore: overallScore == null && nullToAbsent
          ? const Value.absent()
          : Value(overallScore),
      qualityVerdict: qualityVerdict == null && nullToAbsent
          ? const Value.absent()
          : Value(qualityVerdict),
      rejectionReason: rejectionReason == null && nullToAbsent
          ? const Value.absent()
          : Value(rejectionReason),
      qualityThresholdVersion: qualityThresholdVersion == null && nullToAbsent
          ? const Value.absent()
          : Value(qualityThresholdVersion),
      imageWidth: imageWidth == null && nullToAbsent
          ? const Value.absent()
          : Value(imageWidth),
      imageHeight: imageHeight == null && nullToAbsent
          ? const Value.absent()
          : Value(imageHeight),
      exifOrientation: exifOrientation == null && nullToAbsent
          ? const Value.absent()
          : Value(exifOrientation),
      displayOrientation: displayOrientation == null && nullToAbsent
          ? const Value.absent()
          : Value(displayOrientation),
      deviceCameraMetadataJson: deviceCameraMetadataJson == null && nullToAbsent
          ? const Value.absent()
          : Value(deviceCameraMetadataJson),
      syncState: Value(syncState),
      serverAcknowledgedAt: serverAcknowledgedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(serverAcknowledgedAt),
    );
  }

  factory CaptureAsset.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return CaptureAsset(
      id: serializer.fromJson<int>(json['id']),
      assetUuid: serializer.fromJson<String>(json['assetUuid']),
      visitId: serializer.fromJson<int>(json['visitId']),
      role: serializer.fromJson<String>(json['role']),
      localPath: serializer.fromJson<String?>(json['localPath']),
      serverId: serializer.fromJson<int?>(json['serverId']),
      serverObjectId: serializer.fromJson<String?>(json['serverObjectId']),
      capturedAt: serializer.fromJson<DateTime>(json['capturedAt']),
      selectedRank: serializer.fromJson<int?>(json['selectedRank']),
      poseScore: serializer.fromJson<double?>(json['poseScore']),
      coverageScore: serializer.fromJson<double?>(json['coverageScore']),
      orientationScore: serializer.fromJson<double?>(json['orientationScore']),
      sharpnessScore: serializer.fromJson<double?>(json['sharpnessScore']),
      lightingScore: serializer.fromJson<double?>(json['lightingScore']),
      overallScore: serializer.fromJson<double?>(json['overallScore']),
      qualityVerdict: serializer.fromJson<String?>(json['qualityVerdict']),
      rejectionReason: serializer.fromJson<String?>(json['rejectionReason']),
      qualityThresholdVersion:
          serializer.fromJson<String?>(json['qualityThresholdVersion']),
      imageWidth: serializer.fromJson<int?>(json['imageWidth']),
      imageHeight: serializer.fromJson<int?>(json['imageHeight']),
      exifOrientation: serializer.fromJson<int?>(json['exifOrientation']),
      displayOrientation: serializer.fromJson<int?>(json['displayOrientation']),
      deviceCameraMetadataJson:
          serializer.fromJson<String?>(json['deviceCameraMetadataJson']),
      syncState: serializer.fromJson<String>(json['syncState']),
      serverAcknowledgedAt:
          serializer.fromJson<DateTime?>(json['serverAcknowledgedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'assetUuid': serializer.toJson<String>(assetUuid),
      'visitId': serializer.toJson<int>(visitId),
      'role': serializer.toJson<String>(role),
      'localPath': serializer.toJson<String?>(localPath),
      'serverId': serializer.toJson<int?>(serverId),
      'serverObjectId': serializer.toJson<String?>(serverObjectId),
      'capturedAt': serializer.toJson<DateTime>(capturedAt),
      'selectedRank': serializer.toJson<int?>(selectedRank),
      'poseScore': serializer.toJson<double?>(poseScore),
      'coverageScore': serializer.toJson<double?>(coverageScore),
      'orientationScore': serializer.toJson<double?>(orientationScore),
      'sharpnessScore': serializer.toJson<double?>(sharpnessScore),
      'lightingScore': serializer.toJson<double?>(lightingScore),
      'overallScore': serializer.toJson<double?>(overallScore),
      'qualityVerdict': serializer.toJson<String?>(qualityVerdict),
      'rejectionReason': serializer.toJson<String?>(rejectionReason),
      'qualityThresholdVersion':
          serializer.toJson<String?>(qualityThresholdVersion),
      'imageWidth': serializer.toJson<int?>(imageWidth),
      'imageHeight': serializer.toJson<int?>(imageHeight),
      'exifOrientation': serializer.toJson<int?>(exifOrientation),
      'displayOrientation': serializer.toJson<int?>(displayOrientation),
      'deviceCameraMetadataJson':
          serializer.toJson<String?>(deviceCameraMetadataJson),
      'syncState': serializer.toJson<String>(syncState),
      'serverAcknowledgedAt':
          serializer.toJson<DateTime?>(serverAcknowledgedAt),
    };
  }

  CaptureAsset copyWith(
          {int? id,
          String? assetUuid,
          int? visitId,
          String? role,
          Value<String?> localPath = const Value.absent(),
          Value<int?> serverId = const Value.absent(),
          Value<String?> serverObjectId = const Value.absent(),
          DateTime? capturedAt,
          Value<int?> selectedRank = const Value.absent(),
          Value<double?> poseScore = const Value.absent(),
          Value<double?> coverageScore = const Value.absent(),
          Value<double?> orientationScore = const Value.absent(),
          Value<double?> sharpnessScore = const Value.absent(),
          Value<double?> lightingScore = const Value.absent(),
          Value<double?> overallScore = const Value.absent(),
          Value<String?> qualityVerdict = const Value.absent(),
          Value<String?> rejectionReason = const Value.absent(),
          Value<String?> qualityThresholdVersion = const Value.absent(),
          Value<int?> imageWidth = const Value.absent(),
          Value<int?> imageHeight = const Value.absent(),
          Value<int?> exifOrientation = const Value.absent(),
          Value<int?> displayOrientation = const Value.absent(),
          Value<String?> deviceCameraMetadataJson = const Value.absent(),
          String? syncState,
          Value<DateTime?> serverAcknowledgedAt = const Value.absent()}) =>
      CaptureAsset(
        id: id ?? this.id,
        assetUuid: assetUuid ?? this.assetUuid,
        visitId: visitId ?? this.visitId,
        role: role ?? this.role,
        localPath: localPath.present ? localPath.value : this.localPath,
        serverId: serverId.present ? serverId.value : this.serverId,
        serverObjectId:
            serverObjectId.present ? serverObjectId.value : this.serverObjectId,
        capturedAt: capturedAt ?? this.capturedAt,
        selectedRank:
            selectedRank.present ? selectedRank.value : this.selectedRank,
        poseScore: poseScore.present ? poseScore.value : this.poseScore,
        coverageScore:
            coverageScore.present ? coverageScore.value : this.coverageScore,
        orientationScore: orientationScore.present
            ? orientationScore.value
            : this.orientationScore,
        sharpnessScore:
            sharpnessScore.present ? sharpnessScore.value : this.sharpnessScore,
        lightingScore:
            lightingScore.present ? lightingScore.value : this.lightingScore,
        overallScore:
            overallScore.present ? overallScore.value : this.overallScore,
        qualityVerdict:
            qualityVerdict.present ? qualityVerdict.value : this.qualityVerdict,
        rejectionReason: rejectionReason.present
            ? rejectionReason.value
            : this.rejectionReason,
        qualityThresholdVersion: qualityThresholdVersion.present
            ? qualityThresholdVersion.value
            : this.qualityThresholdVersion,
        imageWidth: imageWidth.present ? imageWidth.value : this.imageWidth,
        imageHeight: imageHeight.present ? imageHeight.value : this.imageHeight,
        exifOrientation: exifOrientation.present
            ? exifOrientation.value
            : this.exifOrientation,
        displayOrientation: displayOrientation.present
            ? displayOrientation.value
            : this.displayOrientation,
        deviceCameraMetadataJson: deviceCameraMetadataJson.present
            ? deviceCameraMetadataJson.value
            : this.deviceCameraMetadataJson,
        syncState: syncState ?? this.syncState,
        serverAcknowledgedAt: serverAcknowledgedAt.present
            ? serverAcknowledgedAt.value
            : this.serverAcknowledgedAt,
      );
  CaptureAsset copyWithCompanion(CaptureAssetsCompanion data) {
    return CaptureAsset(
      id: data.id.present ? data.id.value : this.id,
      assetUuid: data.assetUuid.present ? data.assetUuid.value : this.assetUuid,
      visitId: data.visitId.present ? data.visitId.value : this.visitId,
      role: data.role.present ? data.role.value : this.role,
      localPath: data.localPath.present ? data.localPath.value : this.localPath,
      serverId: data.serverId.present ? data.serverId.value : this.serverId,
      serverObjectId: data.serverObjectId.present
          ? data.serverObjectId.value
          : this.serverObjectId,
      capturedAt:
          data.capturedAt.present ? data.capturedAt.value : this.capturedAt,
      selectedRank: data.selectedRank.present
          ? data.selectedRank.value
          : this.selectedRank,
      poseScore: data.poseScore.present ? data.poseScore.value : this.poseScore,
      coverageScore: data.coverageScore.present
          ? data.coverageScore.value
          : this.coverageScore,
      orientationScore: data.orientationScore.present
          ? data.orientationScore.value
          : this.orientationScore,
      sharpnessScore: data.sharpnessScore.present
          ? data.sharpnessScore.value
          : this.sharpnessScore,
      lightingScore: data.lightingScore.present
          ? data.lightingScore.value
          : this.lightingScore,
      overallScore: data.overallScore.present
          ? data.overallScore.value
          : this.overallScore,
      qualityVerdict: data.qualityVerdict.present
          ? data.qualityVerdict.value
          : this.qualityVerdict,
      rejectionReason: data.rejectionReason.present
          ? data.rejectionReason.value
          : this.rejectionReason,
      qualityThresholdVersion: data.qualityThresholdVersion.present
          ? data.qualityThresholdVersion.value
          : this.qualityThresholdVersion,
      imageWidth:
          data.imageWidth.present ? data.imageWidth.value : this.imageWidth,
      imageHeight:
          data.imageHeight.present ? data.imageHeight.value : this.imageHeight,
      exifOrientation: data.exifOrientation.present
          ? data.exifOrientation.value
          : this.exifOrientation,
      displayOrientation: data.displayOrientation.present
          ? data.displayOrientation.value
          : this.displayOrientation,
      deviceCameraMetadataJson: data.deviceCameraMetadataJson.present
          ? data.deviceCameraMetadataJson.value
          : this.deviceCameraMetadataJson,
      syncState: data.syncState.present ? data.syncState.value : this.syncState,
      serverAcknowledgedAt: data.serverAcknowledgedAt.present
          ? data.serverAcknowledgedAt.value
          : this.serverAcknowledgedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('CaptureAsset(')
          ..write('id: $id, ')
          ..write('assetUuid: $assetUuid, ')
          ..write('visitId: $visitId, ')
          ..write('role: $role, ')
          ..write('localPath: $localPath, ')
          ..write('serverId: $serverId, ')
          ..write('serverObjectId: $serverObjectId, ')
          ..write('capturedAt: $capturedAt, ')
          ..write('selectedRank: $selectedRank, ')
          ..write('poseScore: $poseScore, ')
          ..write('coverageScore: $coverageScore, ')
          ..write('orientationScore: $orientationScore, ')
          ..write('sharpnessScore: $sharpnessScore, ')
          ..write('lightingScore: $lightingScore, ')
          ..write('overallScore: $overallScore, ')
          ..write('qualityVerdict: $qualityVerdict, ')
          ..write('rejectionReason: $rejectionReason, ')
          ..write('qualityThresholdVersion: $qualityThresholdVersion, ')
          ..write('imageWidth: $imageWidth, ')
          ..write('imageHeight: $imageHeight, ')
          ..write('exifOrientation: $exifOrientation, ')
          ..write('displayOrientation: $displayOrientation, ')
          ..write('deviceCameraMetadataJson: $deviceCameraMetadataJson, ')
          ..write('syncState: $syncState, ')
          ..write('serverAcknowledgedAt: $serverAcknowledgedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hashAll([
        id,
        assetUuid,
        visitId,
        role,
        localPath,
        serverId,
        serverObjectId,
        capturedAt,
        selectedRank,
        poseScore,
        coverageScore,
        orientationScore,
        sharpnessScore,
        lightingScore,
        overallScore,
        qualityVerdict,
        rejectionReason,
        qualityThresholdVersion,
        imageWidth,
        imageHeight,
        exifOrientation,
        displayOrientation,
        deviceCameraMetadataJson,
        syncState,
        serverAcknowledgedAt
      ]);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is CaptureAsset &&
          other.id == this.id &&
          other.assetUuid == this.assetUuid &&
          other.visitId == this.visitId &&
          other.role == this.role &&
          other.localPath == this.localPath &&
          other.serverId == this.serverId &&
          other.serverObjectId == this.serverObjectId &&
          other.capturedAt == this.capturedAt &&
          other.selectedRank == this.selectedRank &&
          other.poseScore == this.poseScore &&
          other.coverageScore == this.coverageScore &&
          other.orientationScore == this.orientationScore &&
          other.sharpnessScore == this.sharpnessScore &&
          other.lightingScore == this.lightingScore &&
          other.overallScore == this.overallScore &&
          other.qualityVerdict == this.qualityVerdict &&
          other.rejectionReason == this.rejectionReason &&
          other.qualityThresholdVersion == this.qualityThresholdVersion &&
          other.imageWidth == this.imageWidth &&
          other.imageHeight == this.imageHeight &&
          other.exifOrientation == this.exifOrientation &&
          other.displayOrientation == this.displayOrientation &&
          other.deviceCameraMetadataJson == this.deviceCameraMetadataJson &&
          other.syncState == this.syncState &&
          other.serverAcknowledgedAt == this.serverAcknowledgedAt);
}

class CaptureAssetsCompanion extends UpdateCompanion<CaptureAsset> {
  final Value<int> id;
  final Value<String> assetUuid;
  final Value<int> visitId;
  final Value<String> role;
  final Value<String?> localPath;
  final Value<int?> serverId;
  final Value<String?> serverObjectId;
  final Value<DateTime> capturedAt;
  final Value<int?> selectedRank;
  final Value<double?> poseScore;
  final Value<double?> coverageScore;
  final Value<double?> orientationScore;
  final Value<double?> sharpnessScore;
  final Value<double?> lightingScore;
  final Value<double?> overallScore;
  final Value<String?> qualityVerdict;
  final Value<String?> rejectionReason;
  final Value<String?> qualityThresholdVersion;
  final Value<int?> imageWidth;
  final Value<int?> imageHeight;
  final Value<int?> exifOrientation;
  final Value<int?> displayOrientation;
  final Value<String?> deviceCameraMetadataJson;
  final Value<String> syncState;
  final Value<DateTime?> serverAcknowledgedAt;
  const CaptureAssetsCompanion({
    this.id = const Value.absent(),
    this.assetUuid = const Value.absent(),
    this.visitId = const Value.absent(),
    this.role = const Value.absent(),
    this.localPath = const Value.absent(),
    this.serverId = const Value.absent(),
    this.serverObjectId = const Value.absent(),
    this.capturedAt = const Value.absent(),
    this.selectedRank = const Value.absent(),
    this.poseScore = const Value.absent(),
    this.coverageScore = const Value.absent(),
    this.orientationScore = const Value.absent(),
    this.sharpnessScore = const Value.absent(),
    this.lightingScore = const Value.absent(),
    this.overallScore = const Value.absent(),
    this.qualityVerdict = const Value.absent(),
    this.rejectionReason = const Value.absent(),
    this.qualityThresholdVersion = const Value.absent(),
    this.imageWidth = const Value.absent(),
    this.imageHeight = const Value.absent(),
    this.exifOrientation = const Value.absent(),
    this.displayOrientation = const Value.absent(),
    this.deviceCameraMetadataJson = const Value.absent(),
    this.syncState = const Value.absent(),
    this.serverAcknowledgedAt = const Value.absent(),
  });
  CaptureAssetsCompanion.insert({
    this.id = const Value.absent(),
    required String assetUuid,
    required int visitId,
    required String role,
    this.localPath = const Value.absent(),
    this.serverId = const Value.absent(),
    this.serverObjectId = const Value.absent(),
    required DateTime capturedAt,
    this.selectedRank = const Value.absent(),
    this.poseScore = const Value.absent(),
    this.coverageScore = const Value.absent(),
    this.orientationScore = const Value.absent(),
    this.sharpnessScore = const Value.absent(),
    this.lightingScore = const Value.absent(),
    this.overallScore = const Value.absent(),
    this.qualityVerdict = const Value.absent(),
    this.rejectionReason = const Value.absent(),
    this.qualityThresholdVersion = const Value.absent(),
    this.imageWidth = const Value.absent(),
    this.imageHeight = const Value.absent(),
    this.exifOrientation = const Value.absent(),
    this.displayOrientation = const Value.absent(),
    this.deviceCameraMetadataJson = const Value.absent(),
    this.syncState = const Value.absent(),
    this.serverAcknowledgedAt = const Value.absent(),
  })  : assetUuid = Value(assetUuid),
        visitId = Value(visitId),
        role = Value(role),
        capturedAt = Value(capturedAt);
  static Insertable<CaptureAsset> custom({
    Expression<int>? id,
    Expression<String>? assetUuid,
    Expression<int>? visitId,
    Expression<String>? role,
    Expression<String>? localPath,
    Expression<int>? serverId,
    Expression<String>? serverObjectId,
    Expression<DateTime>? capturedAt,
    Expression<int>? selectedRank,
    Expression<double>? poseScore,
    Expression<double>? coverageScore,
    Expression<double>? orientationScore,
    Expression<double>? sharpnessScore,
    Expression<double>? lightingScore,
    Expression<double>? overallScore,
    Expression<String>? qualityVerdict,
    Expression<String>? rejectionReason,
    Expression<String>? qualityThresholdVersion,
    Expression<int>? imageWidth,
    Expression<int>? imageHeight,
    Expression<int>? exifOrientation,
    Expression<int>? displayOrientation,
    Expression<String>? deviceCameraMetadataJson,
    Expression<String>? syncState,
    Expression<DateTime>? serverAcknowledgedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (assetUuid != null) 'asset_uuid': assetUuid,
      if (visitId != null) 'visit_id': visitId,
      if (role != null) 'role': role,
      if (localPath != null) 'local_path': localPath,
      if (serverId != null) 'server_id': serverId,
      if (serverObjectId != null) 'server_object_id': serverObjectId,
      if (capturedAt != null) 'captured_at': capturedAt,
      if (selectedRank != null) 'selected_rank': selectedRank,
      if (poseScore != null) 'pose_score': poseScore,
      if (coverageScore != null) 'coverage_score': coverageScore,
      if (orientationScore != null) 'orientation_score': orientationScore,
      if (sharpnessScore != null) 'sharpness_score': sharpnessScore,
      if (lightingScore != null) 'lighting_score': lightingScore,
      if (overallScore != null) 'overall_score': overallScore,
      if (qualityVerdict != null) 'quality_verdict': qualityVerdict,
      if (rejectionReason != null) 'rejection_reason': rejectionReason,
      if (qualityThresholdVersion != null)
        'quality_threshold_version': qualityThresholdVersion,
      if (imageWidth != null) 'image_width': imageWidth,
      if (imageHeight != null) 'image_height': imageHeight,
      if (exifOrientation != null) 'exif_orientation': exifOrientation,
      if (displayOrientation != null) 'display_orientation': displayOrientation,
      if (deviceCameraMetadataJson != null)
        'device_camera_metadata_json': deviceCameraMetadataJson,
      if (syncState != null) 'sync_state': syncState,
      if (serverAcknowledgedAt != null)
        'server_acknowledged_at': serverAcknowledgedAt,
    });
  }

  CaptureAssetsCompanion copyWith(
      {Value<int>? id,
      Value<String>? assetUuid,
      Value<int>? visitId,
      Value<String>? role,
      Value<String?>? localPath,
      Value<int?>? serverId,
      Value<String?>? serverObjectId,
      Value<DateTime>? capturedAt,
      Value<int?>? selectedRank,
      Value<double?>? poseScore,
      Value<double?>? coverageScore,
      Value<double?>? orientationScore,
      Value<double?>? sharpnessScore,
      Value<double?>? lightingScore,
      Value<double?>? overallScore,
      Value<String?>? qualityVerdict,
      Value<String?>? rejectionReason,
      Value<String?>? qualityThresholdVersion,
      Value<int?>? imageWidth,
      Value<int?>? imageHeight,
      Value<int?>? exifOrientation,
      Value<int?>? displayOrientation,
      Value<String?>? deviceCameraMetadataJson,
      Value<String>? syncState,
      Value<DateTime?>? serverAcknowledgedAt}) {
    return CaptureAssetsCompanion(
      id: id ?? this.id,
      assetUuid: assetUuid ?? this.assetUuid,
      visitId: visitId ?? this.visitId,
      role: role ?? this.role,
      localPath: localPath ?? this.localPath,
      serverId: serverId ?? this.serverId,
      serverObjectId: serverObjectId ?? this.serverObjectId,
      capturedAt: capturedAt ?? this.capturedAt,
      selectedRank: selectedRank ?? this.selectedRank,
      poseScore: poseScore ?? this.poseScore,
      coverageScore: coverageScore ?? this.coverageScore,
      orientationScore: orientationScore ?? this.orientationScore,
      sharpnessScore: sharpnessScore ?? this.sharpnessScore,
      lightingScore: lightingScore ?? this.lightingScore,
      overallScore: overallScore ?? this.overallScore,
      qualityVerdict: qualityVerdict ?? this.qualityVerdict,
      rejectionReason: rejectionReason ?? this.rejectionReason,
      qualityThresholdVersion:
          qualityThresholdVersion ?? this.qualityThresholdVersion,
      imageWidth: imageWidth ?? this.imageWidth,
      imageHeight: imageHeight ?? this.imageHeight,
      exifOrientation: exifOrientation ?? this.exifOrientation,
      displayOrientation: displayOrientation ?? this.displayOrientation,
      deviceCameraMetadataJson:
          deviceCameraMetadataJson ?? this.deviceCameraMetadataJson,
      syncState: syncState ?? this.syncState,
      serverAcknowledgedAt: serverAcknowledgedAt ?? this.serverAcknowledgedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (assetUuid.present) {
      map['asset_uuid'] = Variable<String>(assetUuid.value);
    }
    if (visitId.present) {
      map['visit_id'] = Variable<int>(visitId.value);
    }
    if (role.present) {
      map['role'] = Variable<String>(role.value);
    }
    if (localPath.present) {
      map['local_path'] = Variable<String>(localPath.value);
    }
    if (serverId.present) {
      map['server_id'] = Variable<int>(serverId.value);
    }
    if (serverObjectId.present) {
      map['server_object_id'] = Variable<String>(serverObjectId.value);
    }
    if (capturedAt.present) {
      map['captured_at'] = Variable<DateTime>(capturedAt.value);
    }
    if (selectedRank.present) {
      map['selected_rank'] = Variable<int>(selectedRank.value);
    }
    if (poseScore.present) {
      map['pose_score'] = Variable<double>(poseScore.value);
    }
    if (coverageScore.present) {
      map['coverage_score'] = Variable<double>(coverageScore.value);
    }
    if (orientationScore.present) {
      map['orientation_score'] = Variable<double>(orientationScore.value);
    }
    if (sharpnessScore.present) {
      map['sharpness_score'] = Variable<double>(sharpnessScore.value);
    }
    if (lightingScore.present) {
      map['lighting_score'] = Variable<double>(lightingScore.value);
    }
    if (overallScore.present) {
      map['overall_score'] = Variable<double>(overallScore.value);
    }
    if (qualityVerdict.present) {
      map['quality_verdict'] = Variable<String>(qualityVerdict.value);
    }
    if (rejectionReason.present) {
      map['rejection_reason'] = Variable<String>(rejectionReason.value);
    }
    if (qualityThresholdVersion.present) {
      map['quality_threshold_version'] =
          Variable<String>(qualityThresholdVersion.value);
    }
    if (imageWidth.present) {
      map['image_width'] = Variable<int>(imageWidth.value);
    }
    if (imageHeight.present) {
      map['image_height'] = Variable<int>(imageHeight.value);
    }
    if (exifOrientation.present) {
      map['exif_orientation'] = Variable<int>(exifOrientation.value);
    }
    if (displayOrientation.present) {
      map['display_orientation'] = Variable<int>(displayOrientation.value);
    }
    if (deviceCameraMetadataJson.present) {
      map['device_camera_metadata_json'] =
          Variable<String>(deviceCameraMetadataJson.value);
    }
    if (syncState.present) {
      map['sync_state'] = Variable<String>(syncState.value);
    }
    if (serverAcknowledgedAt.present) {
      map['server_acknowledged_at'] =
          Variable<DateTime>(serverAcknowledgedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('CaptureAssetsCompanion(')
          ..write('id: $id, ')
          ..write('assetUuid: $assetUuid, ')
          ..write('visitId: $visitId, ')
          ..write('role: $role, ')
          ..write('localPath: $localPath, ')
          ..write('serverId: $serverId, ')
          ..write('serverObjectId: $serverObjectId, ')
          ..write('capturedAt: $capturedAt, ')
          ..write('selectedRank: $selectedRank, ')
          ..write('poseScore: $poseScore, ')
          ..write('coverageScore: $coverageScore, ')
          ..write('orientationScore: $orientationScore, ')
          ..write('sharpnessScore: $sharpnessScore, ')
          ..write('lightingScore: $lightingScore, ')
          ..write('overallScore: $overallScore, ')
          ..write('qualityVerdict: $qualityVerdict, ')
          ..write('rejectionReason: $rejectionReason, ')
          ..write('qualityThresholdVersion: $qualityThresholdVersion, ')
          ..write('imageWidth: $imageWidth, ')
          ..write('imageHeight: $imageHeight, ')
          ..write('exifOrientation: $exifOrientation, ')
          ..write('displayOrientation: $displayOrientation, ')
          ..write('deviceCameraMetadataJson: $deviceCameraMetadataJson, ')
          ..write('syncState: $syncState, ')
          ..write('serverAcknowledgedAt: $serverAcknowledgedAt')
          ..write(')'))
        .toString();
  }
}

class $CameraResultsTable extends CameraResults
    with TableInfo<$CameraResultsTable, CameraResult> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $CameraResultsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _resultUuidMeta =
      const VerificationMeta('resultUuid');
  @override
  late final GeneratedColumn<String> resultUuid = GeneratedColumn<String>(
      'result_uuid', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways('UNIQUE'));
  static const VerificationMeta _serverIdMeta =
      const VerificationMeta('serverId');
  @override
  late final GeneratedColumn<int> serverId = GeneratedColumn<int>(
      'server_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _visitIdMeta =
      const VerificationMeta('visitId');
  @override
  late final GeneratedColumn<int> visitId = GeneratedColumn<int>(
      'visit_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'REFERENCES visits (id) ON DELETE CASCADE'));
  static const VerificationMeta _versionMeta =
      const VerificationMeta('version');
  @override
  late final GeneratedColumn<int> version = GeneratedColumn<int>(
      'version', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _supersedesResultUuidMeta =
      const VerificationMeta('supersedesResultUuid');
  @override
  late final GeneratedColumn<String> supersedesResultUuid =
      GeneratedColumn<String>('supersedes_result_uuid', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _estimatedHeightCmMeta =
      const VerificationMeta('estimatedHeightCm');
  @override
  late final GeneratedColumn<double> estimatedHeightCm =
      GeneratedColumn<double>('estimated_height_cm', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _estimatedWeightKgMeta =
      const VerificationMeta('estimatedWeightKg');
  @override
  late final GeneratedColumn<double> estimatedWeightKg =
      GeneratedColumn<double>('estimated_weight_kg', aliasedName, true,
          type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _heightSourceMeta =
      const VerificationMeta('heightSource');
  @override
  late final GeneratedColumn<String> heightSource = GeneratedColumn<String>(
      'height_source', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _weightSourceMeta =
      const VerificationMeta('weightSource');
  @override
  late final GeneratedColumn<String> weightSource = GeneratedColumn<String>(
      'weight_source', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _estimatedHazMeta =
      const VerificationMeta('estimatedHaz');
  @override
  late final GeneratedColumn<double> estimatedHaz = GeneratedColumn<double>(
      'estimated_haz', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _estimatedWhzMeta =
      const VerificationMeta('estimatedWhz');
  @override
  late final GeneratedColumn<double> estimatedWhz = GeneratedColumn<double>(
      'estimated_whz', aliasedName, true,
      type: DriftSqlType.double, requiredDuringInsert: false);
  static const VerificationMeta _estimatedStuntingStatusMeta =
      const VerificationMeta('estimatedStuntingStatus');
  @override
  late final GeneratedColumn<String> estimatedStuntingStatus =
      GeneratedColumn<String>('estimated_stunting_status', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _estimatedWastingStatusMeta =
      const VerificationMeta('estimatedWastingStatus');
  @override
  late final GeneratedColumn<String> estimatedWastingStatus =
      GeneratedColumn<String>('estimated_wasting_status', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _experimentalOverallCategoryMeta =
      const VerificationMeta('experimentalOverallCategory');
  @override
  late final GeneratedColumn<String> experimentalOverallCategory =
      GeneratedColumn<String>(
          'experimental_overall_category', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _componentProbabilitiesJsonMeta =
      const VerificationMeta('componentProbabilitiesJson');
  @override
  late final GeneratedColumn<String> componentProbabilitiesJson =
      GeneratedColumn<String>('component_probabilities_json', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _bodyProportionFeaturesJsonMeta =
      const VerificationMeta('bodyProportionFeaturesJson');
  @override
  late final GeneratedColumn<String> bodyProportionFeaturesJson =
      GeneratedColumn<String>(
          'body_proportion_features_json', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _captureQualitySummaryJsonMeta =
      const VerificationMeta('captureQualitySummaryJson');
  @override
  late final GeneratedColumn<String> captureQualitySummaryJson =
      GeneratedColumn<String>('capture_quality_summary_json', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _methodMeta = const VerificationMeta('method');
  @override
  late final GeneratedColumn<String> method = GeneratedColumn<String>(
      'method', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _modelVersionMeta =
      const VerificationMeta('modelVersion');
  @override
  late final GeneratedColumn<String> modelVersion = GeneratedColumn<String>(
      'model_version', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _manifestChecksumMeta =
      const VerificationMeta('manifestChecksum');
  @override
  late final GeneratedColumn<String> manifestChecksum = GeneratedColumn<String>(
      'manifest_checksum', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _trainingDataLabelMeta =
      const VerificationMeta('trainingDataLabel');
  @override
  late final GeneratedColumn<String> trainingDataLabel =
      GeneratedColumn<String>('training_data_label', aliasedName, false,
          type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _nonClinicalMeta =
      const VerificationMeta('nonClinical');
  @override
  late final GeneratedColumn<bool> nonClinical = GeneratedColumn<bool>(
      'non_clinical', aliasedName, false,
      type: DriftSqlType.bool,
      requiredDuringInsert: false,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'CHECK ("non_clinical" IN (0, 1))'),
      defaultValue: const Constant(true));
  static const VerificationMeta _createdAtMeta =
      const VerificationMeta('createdAt');
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
      'created_at', aliasedName, false,
      type: DriftSqlType.dateTime,
      requiredDuringInsert: false,
      defaultValue: currentDateAndTime);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        resultUuid,
        serverId,
        visitId,
        version,
        supersedesResultUuid,
        estimatedHeightCm,
        estimatedWeightKg,
        heightSource,
        weightSource,
        estimatedHaz,
        estimatedWhz,
        estimatedStuntingStatus,
        estimatedWastingStatus,
        experimentalOverallCategory,
        componentProbabilitiesJson,
        bodyProportionFeaturesJson,
        captureQualitySummaryJson,
        method,
        modelVersion,
        manifestChecksum,
        trainingDataLabel,
        nonClinical,
        createdAt
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'camera_results';
  @override
  VerificationContext validateIntegrity(Insertable<CameraResult> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('result_uuid')) {
      context.handle(
          _resultUuidMeta,
          resultUuid.isAcceptableOrUnknown(
              data['result_uuid']!, _resultUuidMeta));
    } else if (isInserting) {
      context.missing(_resultUuidMeta);
    }
    if (data.containsKey('server_id')) {
      context.handle(_serverIdMeta,
          serverId.isAcceptableOrUnknown(data['server_id']!, _serverIdMeta));
    }
    if (data.containsKey('visit_id')) {
      context.handle(_visitIdMeta,
          visitId.isAcceptableOrUnknown(data['visit_id']!, _visitIdMeta));
    } else if (isInserting) {
      context.missing(_visitIdMeta);
    }
    if (data.containsKey('version')) {
      context.handle(_versionMeta,
          version.isAcceptableOrUnknown(data['version']!, _versionMeta));
    } else if (isInserting) {
      context.missing(_versionMeta);
    }
    if (data.containsKey('supersedes_result_uuid')) {
      context.handle(
          _supersedesResultUuidMeta,
          supersedesResultUuid.isAcceptableOrUnknown(
              data['supersedes_result_uuid']!, _supersedesResultUuidMeta));
    }
    if (data.containsKey('estimated_height_cm')) {
      context.handle(
          _estimatedHeightCmMeta,
          estimatedHeightCm.isAcceptableOrUnknown(
              data['estimated_height_cm']!, _estimatedHeightCmMeta));
    }
    if (data.containsKey('estimated_weight_kg')) {
      context.handle(
          _estimatedWeightKgMeta,
          estimatedWeightKg.isAcceptableOrUnknown(
              data['estimated_weight_kg']!, _estimatedWeightKgMeta));
    }
    if (data.containsKey('height_source')) {
      context.handle(
          _heightSourceMeta,
          heightSource.isAcceptableOrUnknown(
              data['height_source']!, _heightSourceMeta));
    }
    if (data.containsKey('weight_source')) {
      context.handle(
          _weightSourceMeta,
          weightSource.isAcceptableOrUnknown(
              data['weight_source']!, _weightSourceMeta));
    }
    if (data.containsKey('estimated_haz')) {
      context.handle(
          _estimatedHazMeta,
          estimatedHaz.isAcceptableOrUnknown(
              data['estimated_haz']!, _estimatedHazMeta));
    }
    if (data.containsKey('estimated_whz')) {
      context.handle(
          _estimatedWhzMeta,
          estimatedWhz.isAcceptableOrUnknown(
              data['estimated_whz']!, _estimatedWhzMeta));
    }
    if (data.containsKey('estimated_stunting_status')) {
      context.handle(
          _estimatedStuntingStatusMeta,
          estimatedStuntingStatus.isAcceptableOrUnknown(
              data['estimated_stunting_status']!,
              _estimatedStuntingStatusMeta));
    }
    if (data.containsKey('estimated_wasting_status')) {
      context.handle(
          _estimatedWastingStatusMeta,
          estimatedWastingStatus.isAcceptableOrUnknown(
              data['estimated_wasting_status']!, _estimatedWastingStatusMeta));
    }
    if (data.containsKey('experimental_overall_category')) {
      context.handle(
          _experimentalOverallCategoryMeta,
          experimentalOverallCategory.isAcceptableOrUnknown(
              data['experimental_overall_category']!,
              _experimentalOverallCategoryMeta));
    }
    if (data.containsKey('component_probabilities_json')) {
      context.handle(
          _componentProbabilitiesJsonMeta,
          componentProbabilitiesJson.isAcceptableOrUnknown(
              data['component_probabilities_json']!,
              _componentProbabilitiesJsonMeta));
    }
    if (data.containsKey('body_proportion_features_json')) {
      context.handle(
          _bodyProportionFeaturesJsonMeta,
          bodyProportionFeaturesJson.isAcceptableOrUnknown(
              data['body_proportion_features_json']!,
              _bodyProportionFeaturesJsonMeta));
    }
    if (data.containsKey('capture_quality_summary_json')) {
      context.handle(
          _captureQualitySummaryJsonMeta,
          captureQualitySummaryJson.isAcceptableOrUnknown(
              data['capture_quality_summary_json']!,
              _captureQualitySummaryJsonMeta));
    }
    if (data.containsKey('method')) {
      context.handle(_methodMeta,
          method.isAcceptableOrUnknown(data['method']!, _methodMeta));
    } else if (isInserting) {
      context.missing(_methodMeta);
    }
    if (data.containsKey('model_version')) {
      context.handle(
          _modelVersionMeta,
          modelVersion.isAcceptableOrUnknown(
              data['model_version']!, _modelVersionMeta));
    } else if (isInserting) {
      context.missing(_modelVersionMeta);
    }
    if (data.containsKey('manifest_checksum')) {
      context.handle(
          _manifestChecksumMeta,
          manifestChecksum.isAcceptableOrUnknown(
              data['manifest_checksum']!, _manifestChecksumMeta));
    } else if (isInserting) {
      context.missing(_manifestChecksumMeta);
    }
    if (data.containsKey('training_data_label')) {
      context.handle(
          _trainingDataLabelMeta,
          trainingDataLabel.isAcceptableOrUnknown(
              data['training_data_label']!, _trainingDataLabelMeta));
    } else if (isInserting) {
      context.missing(_trainingDataLabelMeta);
    }
    if (data.containsKey('non_clinical')) {
      context.handle(
          _nonClinicalMeta,
          nonClinical.isAcceptableOrUnknown(
              data['non_clinical']!, _nonClinicalMeta));
    }
    if (data.containsKey('created_at')) {
      context.handle(_createdAtMeta,
          createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  List<Set<GeneratedColumn>> get uniqueKeys => [
        {visitId, version},
      ];
  @override
  CameraResult map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return CameraResult(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      resultUuid: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}result_uuid'])!,
      serverId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}server_id']),
      visitId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}visit_id'])!,
      version: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}version'])!,
      supersedesResultUuid: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}supersedes_result_uuid']),
      estimatedHeightCm: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}estimated_height_cm']),
      estimatedWeightKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}estimated_weight_kg']),
      heightSource: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}height_source']),
      weightSource: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}weight_source']),
      estimatedHaz: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}estimated_haz']),
      estimatedWhz: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}estimated_whz']),
      estimatedStuntingStatus: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}estimated_stunting_status']),
      estimatedWastingStatus: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}estimated_wasting_status']),
      experimentalOverallCategory: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}experimental_overall_category']),
      componentProbabilitiesJson: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}component_probabilities_json']),
      bodyProportionFeaturesJson: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}body_proportion_features_json']),
      captureQualitySummaryJson: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}capture_quality_summary_json']),
      method: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}method'])!,
      modelVersion: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}model_version'])!,
      manifestChecksum: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}manifest_checksum'])!,
      trainingDataLabel: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}training_data_label'])!,
      nonClinical: attachedDatabase.typeMapping
          .read(DriftSqlType.bool, data['${effectivePrefix}non_clinical'])!,
      createdAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}created_at'])!,
    );
  }

  @override
  $CameraResultsTable createAlias(String alias) {
    return $CameraResultsTable(attachedDatabase, alias);
  }
}

class CameraResult extends DataClass implements Insertable<CameraResult> {
  final int id;
  final String resultUuid;
  final int? serverId;
  final int visitId;
  final int version;
  final String? supersedesResultUuid;
  final double? estimatedHeightCm;
  final double? estimatedWeightKg;
  final String? heightSource;
  final String? weightSource;
  final double? estimatedHaz;
  final double? estimatedWhz;
  final String? estimatedStuntingStatus;
  final String? estimatedWastingStatus;
  final String? experimentalOverallCategory;
  final String? componentProbabilitiesJson;
  final String? bodyProportionFeaturesJson;
  final String? captureQualitySummaryJson;
  final String method;
  final String modelVersion;
  final String manifestChecksum;
  final String trainingDataLabel;
  final bool nonClinical;
  final DateTime createdAt;
  const CameraResult(
      {required this.id,
      required this.resultUuid,
      this.serverId,
      required this.visitId,
      required this.version,
      this.supersedesResultUuid,
      this.estimatedHeightCm,
      this.estimatedWeightKg,
      this.heightSource,
      this.weightSource,
      this.estimatedHaz,
      this.estimatedWhz,
      this.estimatedStuntingStatus,
      this.estimatedWastingStatus,
      this.experimentalOverallCategory,
      this.componentProbabilitiesJson,
      this.bodyProportionFeaturesJson,
      this.captureQualitySummaryJson,
      required this.method,
      required this.modelVersion,
      required this.manifestChecksum,
      required this.trainingDataLabel,
      required this.nonClinical,
      required this.createdAt});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['result_uuid'] = Variable<String>(resultUuid);
    if (!nullToAbsent || serverId != null) {
      map['server_id'] = Variable<int>(serverId);
    }
    map['visit_id'] = Variable<int>(visitId);
    map['version'] = Variable<int>(version);
    if (!nullToAbsent || supersedesResultUuid != null) {
      map['supersedes_result_uuid'] = Variable<String>(supersedesResultUuid);
    }
    if (!nullToAbsent || estimatedHeightCm != null) {
      map['estimated_height_cm'] = Variable<double>(estimatedHeightCm);
    }
    if (!nullToAbsent || estimatedWeightKg != null) {
      map['estimated_weight_kg'] = Variable<double>(estimatedWeightKg);
    }
    if (!nullToAbsent || heightSource != null) {
      map['height_source'] = Variable<String>(heightSource);
    }
    if (!nullToAbsent || weightSource != null) {
      map['weight_source'] = Variable<String>(weightSource);
    }
    if (!nullToAbsent || estimatedHaz != null) {
      map['estimated_haz'] = Variable<double>(estimatedHaz);
    }
    if (!nullToAbsent || estimatedWhz != null) {
      map['estimated_whz'] = Variable<double>(estimatedWhz);
    }
    if (!nullToAbsent || estimatedStuntingStatus != null) {
      map['estimated_stunting_status'] =
          Variable<String>(estimatedStuntingStatus);
    }
    if (!nullToAbsent || estimatedWastingStatus != null) {
      map['estimated_wasting_status'] =
          Variable<String>(estimatedWastingStatus);
    }
    if (!nullToAbsent || experimentalOverallCategory != null) {
      map['experimental_overall_category'] =
          Variable<String>(experimentalOverallCategory);
    }
    if (!nullToAbsent || componentProbabilitiesJson != null) {
      map['component_probabilities_json'] =
          Variable<String>(componentProbabilitiesJson);
    }
    if (!nullToAbsent || bodyProportionFeaturesJson != null) {
      map['body_proportion_features_json'] =
          Variable<String>(bodyProportionFeaturesJson);
    }
    if (!nullToAbsent || captureQualitySummaryJson != null) {
      map['capture_quality_summary_json'] =
          Variable<String>(captureQualitySummaryJson);
    }
    map['method'] = Variable<String>(method);
    map['model_version'] = Variable<String>(modelVersion);
    map['manifest_checksum'] = Variable<String>(manifestChecksum);
    map['training_data_label'] = Variable<String>(trainingDataLabel);
    map['non_clinical'] = Variable<bool>(nonClinical);
    map['created_at'] = Variable<DateTime>(createdAt);
    return map;
  }

  CameraResultsCompanion toCompanion(bool nullToAbsent) {
    return CameraResultsCompanion(
      id: Value(id),
      resultUuid: Value(resultUuid),
      serverId: serverId == null && nullToAbsent
          ? const Value.absent()
          : Value(serverId),
      visitId: Value(visitId),
      version: Value(version),
      supersedesResultUuid: supersedesResultUuid == null && nullToAbsent
          ? const Value.absent()
          : Value(supersedesResultUuid),
      estimatedHeightCm: estimatedHeightCm == null && nullToAbsent
          ? const Value.absent()
          : Value(estimatedHeightCm),
      estimatedWeightKg: estimatedWeightKg == null && nullToAbsent
          ? const Value.absent()
          : Value(estimatedWeightKg),
      heightSource: heightSource == null && nullToAbsent
          ? const Value.absent()
          : Value(heightSource),
      weightSource: weightSource == null && nullToAbsent
          ? const Value.absent()
          : Value(weightSource),
      estimatedHaz: estimatedHaz == null && nullToAbsent
          ? const Value.absent()
          : Value(estimatedHaz),
      estimatedWhz: estimatedWhz == null && nullToAbsent
          ? const Value.absent()
          : Value(estimatedWhz),
      estimatedStuntingStatus: estimatedStuntingStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(estimatedStuntingStatus),
      estimatedWastingStatus: estimatedWastingStatus == null && nullToAbsent
          ? const Value.absent()
          : Value(estimatedWastingStatus),
      experimentalOverallCategory:
          experimentalOverallCategory == null && nullToAbsent
              ? const Value.absent()
              : Value(experimentalOverallCategory),
      componentProbabilitiesJson:
          componentProbabilitiesJson == null && nullToAbsent
              ? const Value.absent()
              : Value(componentProbabilitiesJson),
      bodyProportionFeaturesJson:
          bodyProportionFeaturesJson == null && nullToAbsent
              ? const Value.absent()
              : Value(bodyProportionFeaturesJson),
      captureQualitySummaryJson:
          captureQualitySummaryJson == null && nullToAbsent
              ? const Value.absent()
              : Value(captureQualitySummaryJson),
      method: Value(method),
      modelVersion: Value(modelVersion),
      manifestChecksum: Value(manifestChecksum),
      trainingDataLabel: Value(trainingDataLabel),
      nonClinical: Value(nonClinical),
      createdAt: Value(createdAt),
    );
  }

  factory CameraResult.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return CameraResult(
      id: serializer.fromJson<int>(json['id']),
      resultUuid: serializer.fromJson<String>(json['resultUuid']),
      serverId: serializer.fromJson<int?>(json['serverId']),
      visitId: serializer.fromJson<int>(json['visitId']),
      version: serializer.fromJson<int>(json['version']),
      supersedesResultUuid:
          serializer.fromJson<String?>(json['supersedesResultUuid']),
      estimatedHeightCm:
          serializer.fromJson<double?>(json['estimatedHeightCm']),
      estimatedWeightKg:
          serializer.fromJson<double?>(json['estimatedWeightKg']),
      heightSource: serializer.fromJson<String?>(json['heightSource']),
      weightSource: serializer.fromJson<String?>(json['weightSource']),
      estimatedHaz: serializer.fromJson<double?>(json['estimatedHaz']),
      estimatedWhz: serializer.fromJson<double?>(json['estimatedWhz']),
      estimatedStuntingStatus:
          serializer.fromJson<String?>(json['estimatedStuntingStatus']),
      estimatedWastingStatus:
          serializer.fromJson<String?>(json['estimatedWastingStatus']),
      experimentalOverallCategory:
          serializer.fromJson<String?>(json['experimentalOverallCategory']),
      componentProbabilitiesJson:
          serializer.fromJson<String?>(json['componentProbabilitiesJson']),
      bodyProportionFeaturesJson:
          serializer.fromJson<String?>(json['bodyProportionFeaturesJson']),
      captureQualitySummaryJson:
          serializer.fromJson<String?>(json['captureQualitySummaryJson']),
      method: serializer.fromJson<String>(json['method']),
      modelVersion: serializer.fromJson<String>(json['modelVersion']),
      manifestChecksum: serializer.fromJson<String>(json['manifestChecksum']),
      trainingDataLabel: serializer.fromJson<String>(json['trainingDataLabel']),
      nonClinical: serializer.fromJson<bool>(json['nonClinical']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'resultUuid': serializer.toJson<String>(resultUuid),
      'serverId': serializer.toJson<int?>(serverId),
      'visitId': serializer.toJson<int>(visitId),
      'version': serializer.toJson<int>(version),
      'supersedesResultUuid': serializer.toJson<String?>(supersedesResultUuid),
      'estimatedHeightCm': serializer.toJson<double?>(estimatedHeightCm),
      'estimatedWeightKg': serializer.toJson<double?>(estimatedWeightKg),
      'heightSource': serializer.toJson<String?>(heightSource),
      'weightSource': serializer.toJson<String?>(weightSource),
      'estimatedHaz': serializer.toJson<double?>(estimatedHaz),
      'estimatedWhz': serializer.toJson<double?>(estimatedWhz),
      'estimatedStuntingStatus':
          serializer.toJson<String?>(estimatedStuntingStatus),
      'estimatedWastingStatus':
          serializer.toJson<String?>(estimatedWastingStatus),
      'experimentalOverallCategory':
          serializer.toJson<String?>(experimentalOverallCategory),
      'componentProbabilitiesJson':
          serializer.toJson<String?>(componentProbabilitiesJson),
      'bodyProportionFeaturesJson':
          serializer.toJson<String?>(bodyProportionFeaturesJson),
      'captureQualitySummaryJson':
          serializer.toJson<String?>(captureQualitySummaryJson),
      'method': serializer.toJson<String>(method),
      'modelVersion': serializer.toJson<String>(modelVersion),
      'manifestChecksum': serializer.toJson<String>(manifestChecksum),
      'trainingDataLabel': serializer.toJson<String>(trainingDataLabel),
      'nonClinical': serializer.toJson<bool>(nonClinical),
      'createdAt': serializer.toJson<DateTime>(createdAt),
    };
  }

  CameraResult copyWith(
          {int? id,
          String? resultUuid,
          Value<int?> serverId = const Value.absent(),
          int? visitId,
          int? version,
          Value<String?> supersedesResultUuid = const Value.absent(),
          Value<double?> estimatedHeightCm = const Value.absent(),
          Value<double?> estimatedWeightKg = const Value.absent(),
          Value<String?> heightSource = const Value.absent(),
          Value<String?> weightSource = const Value.absent(),
          Value<double?> estimatedHaz = const Value.absent(),
          Value<double?> estimatedWhz = const Value.absent(),
          Value<String?> estimatedStuntingStatus = const Value.absent(),
          Value<String?> estimatedWastingStatus = const Value.absent(),
          Value<String?> experimentalOverallCategory = const Value.absent(),
          Value<String?> componentProbabilitiesJson = const Value.absent(),
          Value<String?> bodyProportionFeaturesJson = const Value.absent(),
          Value<String?> captureQualitySummaryJson = const Value.absent(),
          String? method,
          String? modelVersion,
          String? manifestChecksum,
          String? trainingDataLabel,
          bool? nonClinical,
          DateTime? createdAt}) =>
      CameraResult(
        id: id ?? this.id,
        resultUuid: resultUuid ?? this.resultUuid,
        serverId: serverId.present ? serverId.value : this.serverId,
        visitId: visitId ?? this.visitId,
        version: version ?? this.version,
        supersedesResultUuid: supersedesResultUuid.present
            ? supersedesResultUuid.value
            : this.supersedesResultUuid,
        estimatedHeightCm: estimatedHeightCm.present
            ? estimatedHeightCm.value
            : this.estimatedHeightCm,
        estimatedWeightKg: estimatedWeightKg.present
            ? estimatedWeightKg.value
            : this.estimatedWeightKg,
        heightSource:
            heightSource.present ? heightSource.value : this.heightSource,
        weightSource:
            weightSource.present ? weightSource.value : this.weightSource,
        estimatedHaz:
            estimatedHaz.present ? estimatedHaz.value : this.estimatedHaz,
        estimatedWhz:
            estimatedWhz.present ? estimatedWhz.value : this.estimatedWhz,
        estimatedStuntingStatus: estimatedStuntingStatus.present
            ? estimatedStuntingStatus.value
            : this.estimatedStuntingStatus,
        estimatedWastingStatus: estimatedWastingStatus.present
            ? estimatedWastingStatus.value
            : this.estimatedWastingStatus,
        experimentalOverallCategory: experimentalOverallCategory.present
            ? experimentalOverallCategory.value
            : this.experimentalOverallCategory,
        componentProbabilitiesJson: componentProbabilitiesJson.present
            ? componentProbabilitiesJson.value
            : this.componentProbabilitiesJson,
        bodyProportionFeaturesJson: bodyProportionFeaturesJson.present
            ? bodyProportionFeaturesJson.value
            : this.bodyProportionFeaturesJson,
        captureQualitySummaryJson: captureQualitySummaryJson.present
            ? captureQualitySummaryJson.value
            : this.captureQualitySummaryJson,
        method: method ?? this.method,
        modelVersion: modelVersion ?? this.modelVersion,
        manifestChecksum: manifestChecksum ?? this.manifestChecksum,
        trainingDataLabel: trainingDataLabel ?? this.trainingDataLabel,
        nonClinical: nonClinical ?? this.nonClinical,
        createdAt: createdAt ?? this.createdAt,
      );
  CameraResult copyWithCompanion(CameraResultsCompanion data) {
    return CameraResult(
      id: data.id.present ? data.id.value : this.id,
      resultUuid:
          data.resultUuid.present ? data.resultUuid.value : this.resultUuid,
      serverId: data.serverId.present ? data.serverId.value : this.serverId,
      visitId: data.visitId.present ? data.visitId.value : this.visitId,
      version: data.version.present ? data.version.value : this.version,
      supersedesResultUuid: data.supersedesResultUuid.present
          ? data.supersedesResultUuid.value
          : this.supersedesResultUuid,
      estimatedHeightCm: data.estimatedHeightCm.present
          ? data.estimatedHeightCm.value
          : this.estimatedHeightCm,
      estimatedWeightKg: data.estimatedWeightKg.present
          ? data.estimatedWeightKg.value
          : this.estimatedWeightKg,
      heightSource: data.heightSource.present
          ? data.heightSource.value
          : this.heightSource,
      weightSource: data.weightSource.present
          ? data.weightSource.value
          : this.weightSource,
      estimatedHaz: data.estimatedHaz.present
          ? data.estimatedHaz.value
          : this.estimatedHaz,
      estimatedWhz: data.estimatedWhz.present
          ? data.estimatedWhz.value
          : this.estimatedWhz,
      estimatedStuntingStatus: data.estimatedStuntingStatus.present
          ? data.estimatedStuntingStatus.value
          : this.estimatedStuntingStatus,
      estimatedWastingStatus: data.estimatedWastingStatus.present
          ? data.estimatedWastingStatus.value
          : this.estimatedWastingStatus,
      experimentalOverallCategory: data.experimentalOverallCategory.present
          ? data.experimentalOverallCategory.value
          : this.experimentalOverallCategory,
      componentProbabilitiesJson: data.componentProbabilitiesJson.present
          ? data.componentProbabilitiesJson.value
          : this.componentProbabilitiesJson,
      bodyProportionFeaturesJson: data.bodyProportionFeaturesJson.present
          ? data.bodyProportionFeaturesJson.value
          : this.bodyProportionFeaturesJson,
      captureQualitySummaryJson: data.captureQualitySummaryJson.present
          ? data.captureQualitySummaryJson.value
          : this.captureQualitySummaryJson,
      method: data.method.present ? data.method.value : this.method,
      modelVersion: data.modelVersion.present
          ? data.modelVersion.value
          : this.modelVersion,
      manifestChecksum: data.manifestChecksum.present
          ? data.manifestChecksum.value
          : this.manifestChecksum,
      trainingDataLabel: data.trainingDataLabel.present
          ? data.trainingDataLabel.value
          : this.trainingDataLabel,
      nonClinical:
          data.nonClinical.present ? data.nonClinical.value : this.nonClinical,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('CameraResult(')
          ..write('id: $id, ')
          ..write('resultUuid: $resultUuid, ')
          ..write('serverId: $serverId, ')
          ..write('visitId: $visitId, ')
          ..write('version: $version, ')
          ..write('supersedesResultUuid: $supersedesResultUuid, ')
          ..write('estimatedHeightCm: $estimatedHeightCm, ')
          ..write('estimatedWeightKg: $estimatedWeightKg, ')
          ..write('heightSource: $heightSource, ')
          ..write('weightSource: $weightSource, ')
          ..write('estimatedHaz: $estimatedHaz, ')
          ..write('estimatedWhz: $estimatedWhz, ')
          ..write('estimatedStuntingStatus: $estimatedStuntingStatus, ')
          ..write('estimatedWastingStatus: $estimatedWastingStatus, ')
          ..write('experimentalOverallCategory: $experimentalOverallCategory, ')
          ..write('componentProbabilitiesJson: $componentProbabilitiesJson, ')
          ..write('bodyProportionFeaturesJson: $bodyProportionFeaturesJson, ')
          ..write('captureQualitySummaryJson: $captureQualitySummaryJson, ')
          ..write('method: $method, ')
          ..write('modelVersion: $modelVersion, ')
          ..write('manifestChecksum: $manifestChecksum, ')
          ..write('trainingDataLabel: $trainingDataLabel, ')
          ..write('nonClinical: $nonClinical, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hashAll([
        id,
        resultUuid,
        serverId,
        visitId,
        version,
        supersedesResultUuid,
        estimatedHeightCm,
        estimatedWeightKg,
        heightSource,
        weightSource,
        estimatedHaz,
        estimatedWhz,
        estimatedStuntingStatus,
        estimatedWastingStatus,
        experimentalOverallCategory,
        componentProbabilitiesJson,
        bodyProportionFeaturesJson,
        captureQualitySummaryJson,
        method,
        modelVersion,
        manifestChecksum,
        trainingDataLabel,
        nonClinical,
        createdAt
      ]);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is CameraResult &&
          other.id == this.id &&
          other.resultUuid == this.resultUuid &&
          other.serverId == this.serverId &&
          other.visitId == this.visitId &&
          other.version == this.version &&
          other.supersedesResultUuid == this.supersedesResultUuid &&
          other.estimatedHeightCm == this.estimatedHeightCm &&
          other.estimatedWeightKg == this.estimatedWeightKg &&
          other.heightSource == this.heightSource &&
          other.weightSource == this.weightSource &&
          other.estimatedHaz == this.estimatedHaz &&
          other.estimatedWhz == this.estimatedWhz &&
          other.estimatedStuntingStatus == this.estimatedStuntingStatus &&
          other.estimatedWastingStatus == this.estimatedWastingStatus &&
          other.experimentalOverallCategory ==
              this.experimentalOverallCategory &&
          other.componentProbabilitiesJson == this.componentProbabilitiesJson &&
          other.bodyProportionFeaturesJson == this.bodyProportionFeaturesJson &&
          other.captureQualitySummaryJson == this.captureQualitySummaryJson &&
          other.method == this.method &&
          other.modelVersion == this.modelVersion &&
          other.manifestChecksum == this.manifestChecksum &&
          other.trainingDataLabel == this.trainingDataLabel &&
          other.nonClinical == this.nonClinical &&
          other.createdAt == this.createdAt);
}

class CameraResultsCompanion extends UpdateCompanion<CameraResult> {
  final Value<int> id;
  final Value<String> resultUuid;
  final Value<int?> serverId;
  final Value<int> visitId;
  final Value<int> version;
  final Value<String?> supersedesResultUuid;
  final Value<double?> estimatedHeightCm;
  final Value<double?> estimatedWeightKg;
  final Value<String?> heightSource;
  final Value<String?> weightSource;
  final Value<double?> estimatedHaz;
  final Value<double?> estimatedWhz;
  final Value<String?> estimatedStuntingStatus;
  final Value<String?> estimatedWastingStatus;
  final Value<String?> experimentalOverallCategory;
  final Value<String?> componentProbabilitiesJson;
  final Value<String?> bodyProportionFeaturesJson;
  final Value<String?> captureQualitySummaryJson;
  final Value<String> method;
  final Value<String> modelVersion;
  final Value<String> manifestChecksum;
  final Value<String> trainingDataLabel;
  final Value<bool> nonClinical;
  final Value<DateTime> createdAt;
  const CameraResultsCompanion({
    this.id = const Value.absent(),
    this.resultUuid = const Value.absent(),
    this.serverId = const Value.absent(),
    this.visitId = const Value.absent(),
    this.version = const Value.absent(),
    this.supersedesResultUuid = const Value.absent(),
    this.estimatedHeightCm = const Value.absent(),
    this.estimatedWeightKg = const Value.absent(),
    this.heightSource = const Value.absent(),
    this.weightSource = const Value.absent(),
    this.estimatedHaz = const Value.absent(),
    this.estimatedWhz = const Value.absent(),
    this.estimatedStuntingStatus = const Value.absent(),
    this.estimatedWastingStatus = const Value.absent(),
    this.experimentalOverallCategory = const Value.absent(),
    this.componentProbabilitiesJson = const Value.absent(),
    this.bodyProportionFeaturesJson = const Value.absent(),
    this.captureQualitySummaryJson = const Value.absent(),
    this.method = const Value.absent(),
    this.modelVersion = const Value.absent(),
    this.manifestChecksum = const Value.absent(),
    this.trainingDataLabel = const Value.absent(),
    this.nonClinical = const Value.absent(),
    this.createdAt = const Value.absent(),
  });
  CameraResultsCompanion.insert({
    this.id = const Value.absent(),
    required String resultUuid,
    this.serverId = const Value.absent(),
    required int visitId,
    required int version,
    this.supersedesResultUuid = const Value.absent(),
    this.estimatedHeightCm = const Value.absent(),
    this.estimatedWeightKg = const Value.absent(),
    this.heightSource = const Value.absent(),
    this.weightSource = const Value.absent(),
    this.estimatedHaz = const Value.absent(),
    this.estimatedWhz = const Value.absent(),
    this.estimatedStuntingStatus = const Value.absent(),
    this.estimatedWastingStatus = const Value.absent(),
    this.experimentalOverallCategory = const Value.absent(),
    this.componentProbabilitiesJson = const Value.absent(),
    this.bodyProportionFeaturesJson = const Value.absent(),
    this.captureQualitySummaryJson = const Value.absent(),
    required String method,
    required String modelVersion,
    required String manifestChecksum,
    required String trainingDataLabel,
    this.nonClinical = const Value.absent(),
    this.createdAt = const Value.absent(),
  })  : resultUuid = Value(resultUuid),
        visitId = Value(visitId),
        version = Value(version),
        method = Value(method),
        modelVersion = Value(modelVersion),
        manifestChecksum = Value(manifestChecksum),
        trainingDataLabel = Value(trainingDataLabel);
  static Insertable<CameraResult> custom({
    Expression<int>? id,
    Expression<String>? resultUuid,
    Expression<int>? serverId,
    Expression<int>? visitId,
    Expression<int>? version,
    Expression<String>? supersedesResultUuid,
    Expression<double>? estimatedHeightCm,
    Expression<double>? estimatedWeightKg,
    Expression<String>? heightSource,
    Expression<String>? weightSource,
    Expression<double>? estimatedHaz,
    Expression<double>? estimatedWhz,
    Expression<String>? estimatedStuntingStatus,
    Expression<String>? estimatedWastingStatus,
    Expression<String>? experimentalOverallCategory,
    Expression<String>? componentProbabilitiesJson,
    Expression<String>? bodyProportionFeaturesJson,
    Expression<String>? captureQualitySummaryJson,
    Expression<String>? method,
    Expression<String>? modelVersion,
    Expression<String>? manifestChecksum,
    Expression<String>? trainingDataLabel,
    Expression<bool>? nonClinical,
    Expression<DateTime>? createdAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (resultUuid != null) 'result_uuid': resultUuid,
      if (serverId != null) 'server_id': serverId,
      if (visitId != null) 'visit_id': visitId,
      if (version != null) 'version': version,
      if (supersedesResultUuid != null)
        'supersedes_result_uuid': supersedesResultUuid,
      if (estimatedHeightCm != null) 'estimated_height_cm': estimatedHeightCm,
      if (estimatedWeightKg != null) 'estimated_weight_kg': estimatedWeightKg,
      if (heightSource != null) 'height_source': heightSource,
      if (weightSource != null) 'weight_source': weightSource,
      if (estimatedHaz != null) 'estimated_haz': estimatedHaz,
      if (estimatedWhz != null) 'estimated_whz': estimatedWhz,
      if (estimatedStuntingStatus != null)
        'estimated_stunting_status': estimatedStuntingStatus,
      if (estimatedWastingStatus != null)
        'estimated_wasting_status': estimatedWastingStatus,
      if (experimentalOverallCategory != null)
        'experimental_overall_category': experimentalOverallCategory,
      if (componentProbabilitiesJson != null)
        'component_probabilities_json': componentProbabilitiesJson,
      if (bodyProportionFeaturesJson != null)
        'body_proportion_features_json': bodyProportionFeaturesJson,
      if (captureQualitySummaryJson != null)
        'capture_quality_summary_json': captureQualitySummaryJson,
      if (method != null) 'method': method,
      if (modelVersion != null) 'model_version': modelVersion,
      if (manifestChecksum != null) 'manifest_checksum': manifestChecksum,
      if (trainingDataLabel != null) 'training_data_label': trainingDataLabel,
      if (nonClinical != null) 'non_clinical': nonClinical,
      if (createdAt != null) 'created_at': createdAt,
    });
  }

  CameraResultsCompanion copyWith(
      {Value<int>? id,
      Value<String>? resultUuid,
      Value<int?>? serverId,
      Value<int>? visitId,
      Value<int>? version,
      Value<String?>? supersedesResultUuid,
      Value<double?>? estimatedHeightCm,
      Value<double?>? estimatedWeightKg,
      Value<String?>? heightSource,
      Value<String?>? weightSource,
      Value<double?>? estimatedHaz,
      Value<double?>? estimatedWhz,
      Value<String?>? estimatedStuntingStatus,
      Value<String?>? estimatedWastingStatus,
      Value<String?>? experimentalOverallCategory,
      Value<String?>? componentProbabilitiesJson,
      Value<String?>? bodyProportionFeaturesJson,
      Value<String?>? captureQualitySummaryJson,
      Value<String>? method,
      Value<String>? modelVersion,
      Value<String>? manifestChecksum,
      Value<String>? trainingDataLabel,
      Value<bool>? nonClinical,
      Value<DateTime>? createdAt}) {
    return CameraResultsCompanion(
      id: id ?? this.id,
      resultUuid: resultUuid ?? this.resultUuid,
      serverId: serverId ?? this.serverId,
      visitId: visitId ?? this.visitId,
      version: version ?? this.version,
      supersedesResultUuid: supersedesResultUuid ?? this.supersedesResultUuid,
      estimatedHeightCm: estimatedHeightCm ?? this.estimatedHeightCm,
      estimatedWeightKg: estimatedWeightKg ?? this.estimatedWeightKg,
      heightSource: heightSource ?? this.heightSource,
      weightSource: weightSource ?? this.weightSource,
      estimatedHaz: estimatedHaz ?? this.estimatedHaz,
      estimatedWhz: estimatedWhz ?? this.estimatedWhz,
      estimatedStuntingStatus:
          estimatedStuntingStatus ?? this.estimatedStuntingStatus,
      estimatedWastingStatus:
          estimatedWastingStatus ?? this.estimatedWastingStatus,
      experimentalOverallCategory:
          experimentalOverallCategory ?? this.experimentalOverallCategory,
      componentProbabilitiesJson:
          componentProbabilitiesJson ?? this.componentProbabilitiesJson,
      bodyProportionFeaturesJson:
          bodyProportionFeaturesJson ?? this.bodyProportionFeaturesJson,
      captureQualitySummaryJson:
          captureQualitySummaryJson ?? this.captureQualitySummaryJson,
      method: method ?? this.method,
      modelVersion: modelVersion ?? this.modelVersion,
      manifestChecksum: manifestChecksum ?? this.manifestChecksum,
      trainingDataLabel: trainingDataLabel ?? this.trainingDataLabel,
      nonClinical: nonClinical ?? this.nonClinical,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (resultUuid.present) {
      map['result_uuid'] = Variable<String>(resultUuid.value);
    }
    if (serverId.present) {
      map['server_id'] = Variable<int>(serverId.value);
    }
    if (visitId.present) {
      map['visit_id'] = Variable<int>(visitId.value);
    }
    if (version.present) {
      map['version'] = Variable<int>(version.value);
    }
    if (supersedesResultUuid.present) {
      map['supersedes_result_uuid'] =
          Variable<String>(supersedesResultUuid.value);
    }
    if (estimatedHeightCm.present) {
      map['estimated_height_cm'] = Variable<double>(estimatedHeightCm.value);
    }
    if (estimatedWeightKg.present) {
      map['estimated_weight_kg'] = Variable<double>(estimatedWeightKg.value);
    }
    if (heightSource.present) {
      map['height_source'] = Variable<String>(heightSource.value);
    }
    if (weightSource.present) {
      map['weight_source'] = Variable<String>(weightSource.value);
    }
    if (estimatedHaz.present) {
      map['estimated_haz'] = Variable<double>(estimatedHaz.value);
    }
    if (estimatedWhz.present) {
      map['estimated_whz'] = Variable<double>(estimatedWhz.value);
    }
    if (estimatedStuntingStatus.present) {
      map['estimated_stunting_status'] =
          Variable<String>(estimatedStuntingStatus.value);
    }
    if (estimatedWastingStatus.present) {
      map['estimated_wasting_status'] =
          Variable<String>(estimatedWastingStatus.value);
    }
    if (experimentalOverallCategory.present) {
      map['experimental_overall_category'] =
          Variable<String>(experimentalOverallCategory.value);
    }
    if (componentProbabilitiesJson.present) {
      map['component_probabilities_json'] =
          Variable<String>(componentProbabilitiesJson.value);
    }
    if (bodyProportionFeaturesJson.present) {
      map['body_proportion_features_json'] =
          Variable<String>(bodyProportionFeaturesJson.value);
    }
    if (captureQualitySummaryJson.present) {
      map['capture_quality_summary_json'] =
          Variable<String>(captureQualitySummaryJson.value);
    }
    if (method.present) {
      map['method'] = Variable<String>(method.value);
    }
    if (modelVersion.present) {
      map['model_version'] = Variable<String>(modelVersion.value);
    }
    if (manifestChecksum.present) {
      map['manifest_checksum'] = Variable<String>(manifestChecksum.value);
    }
    if (trainingDataLabel.present) {
      map['training_data_label'] = Variable<String>(trainingDataLabel.value);
    }
    if (nonClinical.present) {
      map['non_clinical'] = Variable<bool>(nonClinical.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('CameraResultsCompanion(')
          ..write('id: $id, ')
          ..write('resultUuid: $resultUuid, ')
          ..write('serverId: $serverId, ')
          ..write('visitId: $visitId, ')
          ..write('version: $version, ')
          ..write('supersedesResultUuid: $supersedesResultUuid, ')
          ..write('estimatedHeightCm: $estimatedHeightCm, ')
          ..write('estimatedWeightKg: $estimatedWeightKg, ')
          ..write('heightSource: $heightSource, ')
          ..write('weightSource: $weightSource, ')
          ..write('estimatedHaz: $estimatedHaz, ')
          ..write('estimatedWhz: $estimatedWhz, ')
          ..write('estimatedStuntingStatus: $estimatedStuntingStatus, ')
          ..write('estimatedWastingStatus: $estimatedWastingStatus, ')
          ..write('experimentalOverallCategory: $experimentalOverallCategory, ')
          ..write('componentProbabilitiesJson: $componentProbabilitiesJson, ')
          ..write('bodyProportionFeaturesJson: $bodyProportionFeaturesJson, ')
          ..write('captureQualitySummaryJson: $captureQualitySummaryJson, ')
          ..write('method: $method, ')
          ..write('modelVersion: $modelVersion, ')
          ..write('manifestChecksum: $manifestChecksum, ')
          ..write('trainingDataLabel: $trainingDataLabel, ')
          ..write('nonClinical: $nonClinical, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }
}

class $MeasuredDetailRevisionsTable extends MeasuredDetailRevisions
    with TableInfo<$MeasuredDetailRevisionsTable, MeasuredDetailRevision> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $MeasuredDetailRevisionsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _revisionUuidMeta =
      const VerificationMeta('revisionUuid');
  @override
  late final GeneratedColumn<String> revisionUuid = GeneratedColumn<String>(
      'revision_uuid', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways('UNIQUE'));
  static const VerificationMeta _serverIdMeta =
      const VerificationMeta('serverId');
  @override
  late final GeneratedColumn<int> serverId = GeneratedColumn<int>(
      'server_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _visitIdMeta =
      const VerificationMeta('visitId');
  @override
  late final GeneratedColumn<int> visitId = GeneratedColumn<int>(
      'visit_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'REFERENCES visits (id) ON DELETE CASCADE'));
  static const VerificationMeta _revisionNumberMeta =
      const VerificationMeta('revisionNumber');
  @override
  late final GeneratedColumn<int> revisionNumber = GeneratedColumn<int>(
      'revision_number', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _beforeJsonMeta =
      const VerificationMeta('beforeJson');
  @override
  late final GeneratedColumn<String> beforeJson = GeneratedColumn<String>(
      'before_json', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _afterJsonMeta =
      const VerificationMeta('afterJson');
  @override
  late final GeneratedColumn<String> afterJson = GeneratedColumn<String>(
      'after_json', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _editorUserIdMeta =
      const VerificationMeta('editorUserId');
  @override
  late final GeneratedColumn<int> editorUserId = GeneratedColumn<int>(
      'editor_user_id', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _createdAtMeta =
      const VerificationMeta('createdAt');
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
      'created_at', aliasedName, false,
      type: DriftSqlType.dateTime,
      requiredDuringInsert: false,
      defaultValue: currentDateAndTime);
  static const VerificationMeta _reasonMeta = const VerificationMeta('reason');
  @override
  late final GeneratedColumn<String> reason = GeneratedColumn<String>(
      'reason', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        revisionUuid,
        serverId,
        visitId,
        revisionNumber,
        beforeJson,
        afterJson,
        editorUserId,
        createdAt,
        reason
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'measured_detail_revisions';
  @override
  VerificationContext validateIntegrity(
      Insertable<MeasuredDetailRevision> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('revision_uuid')) {
      context.handle(
          _revisionUuidMeta,
          revisionUuid.isAcceptableOrUnknown(
              data['revision_uuid']!, _revisionUuidMeta));
    } else if (isInserting) {
      context.missing(_revisionUuidMeta);
    }
    if (data.containsKey('server_id')) {
      context.handle(_serverIdMeta,
          serverId.isAcceptableOrUnknown(data['server_id']!, _serverIdMeta));
    }
    if (data.containsKey('visit_id')) {
      context.handle(_visitIdMeta,
          visitId.isAcceptableOrUnknown(data['visit_id']!, _visitIdMeta));
    } else if (isInserting) {
      context.missing(_visitIdMeta);
    }
    if (data.containsKey('revision_number')) {
      context.handle(
          _revisionNumberMeta,
          revisionNumber.isAcceptableOrUnknown(
              data['revision_number']!, _revisionNumberMeta));
    } else if (isInserting) {
      context.missing(_revisionNumberMeta);
    }
    if (data.containsKey('before_json')) {
      context.handle(
          _beforeJsonMeta,
          beforeJson.isAcceptableOrUnknown(
              data['before_json']!, _beforeJsonMeta));
    } else if (isInserting) {
      context.missing(_beforeJsonMeta);
    }
    if (data.containsKey('after_json')) {
      context.handle(_afterJsonMeta,
          afterJson.isAcceptableOrUnknown(data['after_json']!, _afterJsonMeta));
    } else if (isInserting) {
      context.missing(_afterJsonMeta);
    }
    if (data.containsKey('editor_user_id')) {
      context.handle(
          _editorUserIdMeta,
          editorUserId.isAcceptableOrUnknown(
              data['editor_user_id']!, _editorUserIdMeta));
    }
    if (data.containsKey('created_at')) {
      context.handle(_createdAtMeta,
          createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta));
    }
    if (data.containsKey('reason')) {
      context.handle(_reasonMeta,
          reason.isAcceptableOrUnknown(data['reason']!, _reasonMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  List<Set<GeneratedColumn>> get uniqueKeys => [
        {visitId, revisionNumber},
      ];
  @override
  MeasuredDetailRevision map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return MeasuredDetailRevision(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      revisionUuid: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}revision_uuid'])!,
      serverId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}server_id']),
      visitId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}visit_id'])!,
      revisionNumber: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}revision_number'])!,
      beforeJson: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}before_json'])!,
      afterJson: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}after_json'])!,
      editorUserId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}editor_user_id']),
      createdAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}created_at'])!,
      reason: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}reason']),
    );
  }

  @override
  $MeasuredDetailRevisionsTable createAlias(String alias) {
    return $MeasuredDetailRevisionsTable(attachedDatabase, alias);
  }
}

class MeasuredDetailRevision extends DataClass
    implements Insertable<MeasuredDetailRevision> {
  final int id;
  final String revisionUuid;
  final int? serverId;
  final int visitId;
  final int revisionNumber;
  final String beforeJson;
  final String afterJson;
  final int? editorUserId;
  final DateTime createdAt;
  final String? reason;
  const MeasuredDetailRevision(
      {required this.id,
      required this.revisionUuid,
      this.serverId,
      required this.visitId,
      required this.revisionNumber,
      required this.beforeJson,
      required this.afterJson,
      this.editorUserId,
      required this.createdAt,
      this.reason});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['revision_uuid'] = Variable<String>(revisionUuid);
    if (!nullToAbsent || serverId != null) {
      map['server_id'] = Variable<int>(serverId);
    }
    map['visit_id'] = Variable<int>(visitId);
    map['revision_number'] = Variable<int>(revisionNumber);
    map['before_json'] = Variable<String>(beforeJson);
    map['after_json'] = Variable<String>(afterJson);
    if (!nullToAbsent || editorUserId != null) {
      map['editor_user_id'] = Variable<int>(editorUserId);
    }
    map['created_at'] = Variable<DateTime>(createdAt);
    if (!nullToAbsent || reason != null) {
      map['reason'] = Variable<String>(reason);
    }
    return map;
  }

  MeasuredDetailRevisionsCompanion toCompanion(bool nullToAbsent) {
    return MeasuredDetailRevisionsCompanion(
      id: Value(id),
      revisionUuid: Value(revisionUuid),
      serverId: serverId == null && nullToAbsent
          ? const Value.absent()
          : Value(serverId),
      visitId: Value(visitId),
      revisionNumber: Value(revisionNumber),
      beforeJson: Value(beforeJson),
      afterJson: Value(afterJson),
      editorUserId: editorUserId == null && nullToAbsent
          ? const Value.absent()
          : Value(editorUserId),
      createdAt: Value(createdAt),
      reason:
          reason == null && nullToAbsent ? const Value.absent() : Value(reason),
    );
  }

  factory MeasuredDetailRevision.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return MeasuredDetailRevision(
      id: serializer.fromJson<int>(json['id']),
      revisionUuid: serializer.fromJson<String>(json['revisionUuid']),
      serverId: serializer.fromJson<int?>(json['serverId']),
      visitId: serializer.fromJson<int>(json['visitId']),
      revisionNumber: serializer.fromJson<int>(json['revisionNumber']),
      beforeJson: serializer.fromJson<String>(json['beforeJson']),
      afterJson: serializer.fromJson<String>(json['afterJson']),
      editorUserId: serializer.fromJson<int?>(json['editorUserId']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
      reason: serializer.fromJson<String?>(json['reason']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'revisionUuid': serializer.toJson<String>(revisionUuid),
      'serverId': serializer.toJson<int?>(serverId),
      'visitId': serializer.toJson<int>(visitId),
      'revisionNumber': serializer.toJson<int>(revisionNumber),
      'beforeJson': serializer.toJson<String>(beforeJson),
      'afterJson': serializer.toJson<String>(afterJson),
      'editorUserId': serializer.toJson<int?>(editorUserId),
      'createdAt': serializer.toJson<DateTime>(createdAt),
      'reason': serializer.toJson<String?>(reason),
    };
  }

  MeasuredDetailRevision copyWith(
          {int? id,
          String? revisionUuid,
          Value<int?> serverId = const Value.absent(),
          int? visitId,
          int? revisionNumber,
          String? beforeJson,
          String? afterJson,
          Value<int?> editorUserId = const Value.absent(),
          DateTime? createdAt,
          Value<String?> reason = const Value.absent()}) =>
      MeasuredDetailRevision(
        id: id ?? this.id,
        revisionUuid: revisionUuid ?? this.revisionUuid,
        serverId: serverId.present ? serverId.value : this.serverId,
        visitId: visitId ?? this.visitId,
        revisionNumber: revisionNumber ?? this.revisionNumber,
        beforeJson: beforeJson ?? this.beforeJson,
        afterJson: afterJson ?? this.afterJson,
        editorUserId:
            editorUserId.present ? editorUserId.value : this.editorUserId,
        createdAt: createdAt ?? this.createdAt,
        reason: reason.present ? reason.value : this.reason,
      );
  MeasuredDetailRevision copyWithCompanion(
      MeasuredDetailRevisionsCompanion data) {
    return MeasuredDetailRevision(
      id: data.id.present ? data.id.value : this.id,
      revisionUuid: data.revisionUuid.present
          ? data.revisionUuid.value
          : this.revisionUuid,
      serverId: data.serverId.present ? data.serverId.value : this.serverId,
      visitId: data.visitId.present ? data.visitId.value : this.visitId,
      revisionNumber: data.revisionNumber.present
          ? data.revisionNumber.value
          : this.revisionNumber,
      beforeJson:
          data.beforeJson.present ? data.beforeJson.value : this.beforeJson,
      afterJson: data.afterJson.present ? data.afterJson.value : this.afterJson,
      editorUserId: data.editorUserId.present
          ? data.editorUserId.value
          : this.editorUserId,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
      reason: data.reason.present ? data.reason.value : this.reason,
    );
  }

  @override
  String toString() {
    return (StringBuffer('MeasuredDetailRevision(')
          ..write('id: $id, ')
          ..write('revisionUuid: $revisionUuid, ')
          ..write('serverId: $serverId, ')
          ..write('visitId: $visitId, ')
          ..write('revisionNumber: $revisionNumber, ')
          ..write('beforeJson: $beforeJson, ')
          ..write('afterJson: $afterJson, ')
          ..write('editorUserId: $editorUserId, ')
          ..write('createdAt: $createdAt, ')
          ..write('reason: $reason')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, revisionUuid, serverId, visitId,
      revisionNumber, beforeJson, afterJson, editorUserId, createdAt, reason);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is MeasuredDetailRevision &&
          other.id == this.id &&
          other.revisionUuid == this.revisionUuid &&
          other.serverId == this.serverId &&
          other.visitId == this.visitId &&
          other.revisionNumber == this.revisionNumber &&
          other.beforeJson == this.beforeJson &&
          other.afterJson == this.afterJson &&
          other.editorUserId == this.editorUserId &&
          other.createdAt == this.createdAt &&
          other.reason == this.reason);
}

class MeasuredDetailRevisionsCompanion
    extends UpdateCompanion<MeasuredDetailRevision> {
  final Value<int> id;
  final Value<String> revisionUuid;
  final Value<int?> serverId;
  final Value<int> visitId;
  final Value<int> revisionNumber;
  final Value<String> beforeJson;
  final Value<String> afterJson;
  final Value<int?> editorUserId;
  final Value<DateTime> createdAt;
  final Value<String?> reason;
  const MeasuredDetailRevisionsCompanion({
    this.id = const Value.absent(),
    this.revisionUuid = const Value.absent(),
    this.serverId = const Value.absent(),
    this.visitId = const Value.absent(),
    this.revisionNumber = const Value.absent(),
    this.beforeJson = const Value.absent(),
    this.afterJson = const Value.absent(),
    this.editorUserId = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.reason = const Value.absent(),
  });
  MeasuredDetailRevisionsCompanion.insert({
    this.id = const Value.absent(),
    required String revisionUuid,
    this.serverId = const Value.absent(),
    required int visitId,
    required int revisionNumber,
    required String beforeJson,
    required String afterJson,
    this.editorUserId = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.reason = const Value.absent(),
  })  : revisionUuid = Value(revisionUuid),
        visitId = Value(visitId),
        revisionNumber = Value(revisionNumber),
        beforeJson = Value(beforeJson),
        afterJson = Value(afterJson);
  static Insertable<MeasuredDetailRevision> custom({
    Expression<int>? id,
    Expression<String>? revisionUuid,
    Expression<int>? serverId,
    Expression<int>? visitId,
    Expression<int>? revisionNumber,
    Expression<String>? beforeJson,
    Expression<String>? afterJson,
    Expression<int>? editorUserId,
    Expression<DateTime>? createdAt,
    Expression<String>? reason,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (revisionUuid != null) 'revision_uuid': revisionUuid,
      if (serverId != null) 'server_id': serverId,
      if (visitId != null) 'visit_id': visitId,
      if (revisionNumber != null) 'revision_number': revisionNumber,
      if (beforeJson != null) 'before_json': beforeJson,
      if (afterJson != null) 'after_json': afterJson,
      if (editorUserId != null) 'editor_user_id': editorUserId,
      if (createdAt != null) 'created_at': createdAt,
      if (reason != null) 'reason': reason,
    });
  }

  MeasuredDetailRevisionsCompanion copyWith(
      {Value<int>? id,
      Value<String>? revisionUuid,
      Value<int?>? serverId,
      Value<int>? visitId,
      Value<int>? revisionNumber,
      Value<String>? beforeJson,
      Value<String>? afterJson,
      Value<int?>? editorUserId,
      Value<DateTime>? createdAt,
      Value<String?>? reason}) {
    return MeasuredDetailRevisionsCompanion(
      id: id ?? this.id,
      revisionUuid: revisionUuid ?? this.revisionUuid,
      serverId: serverId ?? this.serverId,
      visitId: visitId ?? this.visitId,
      revisionNumber: revisionNumber ?? this.revisionNumber,
      beforeJson: beforeJson ?? this.beforeJson,
      afterJson: afterJson ?? this.afterJson,
      editorUserId: editorUserId ?? this.editorUserId,
      createdAt: createdAt ?? this.createdAt,
      reason: reason ?? this.reason,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (revisionUuid.present) {
      map['revision_uuid'] = Variable<String>(revisionUuid.value);
    }
    if (serverId.present) {
      map['server_id'] = Variable<int>(serverId.value);
    }
    if (visitId.present) {
      map['visit_id'] = Variable<int>(visitId.value);
    }
    if (revisionNumber.present) {
      map['revision_number'] = Variable<int>(revisionNumber.value);
    }
    if (beforeJson.present) {
      map['before_json'] = Variable<String>(beforeJson.value);
    }
    if (afterJson.present) {
      map['after_json'] = Variable<String>(afterJson.value);
    }
    if (editorUserId.present) {
      map['editor_user_id'] = Variable<int>(editorUserId.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    if (reason.present) {
      map['reason'] = Variable<String>(reason.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('MeasuredDetailRevisionsCompanion(')
          ..write('id: $id, ')
          ..write('revisionUuid: $revisionUuid, ')
          ..write('serverId: $serverId, ')
          ..write('visitId: $visitId, ')
          ..write('revisionNumber: $revisionNumber, ')
          ..write('beforeJson: $beforeJson, ')
          ..write('afterJson: $afterJson, ')
          ..write('editorUserId: $editorUserId, ')
          ..write('createdAt: $createdAt, ')
          ..write('reason: $reason')
          ..write(')'))
        .toString();
  }
}

class $SyncOutboxTable extends SyncOutbox
    with TableInfo<$SyncOutboxTable, SyncOutboxData> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $SyncOutboxTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _ownerUserIdMeta =
      const VerificationMeta('ownerUserId');
  @override
  late final GeneratedColumn<int> ownerUserId = GeneratedColumn<int>(
      'owner_user_id', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _visitUuidMeta =
      const VerificationMeta('visitUuid');
  @override
  late final GeneratedColumn<String> visitUuid = GeneratedColumn<String>(
      'visit_uuid', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _entityTypeMeta =
      const VerificationMeta('entityType');
  @override
  late final GeneratedColumn<String> entityType = GeneratedColumn<String>(
      'entity_type', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _entityUuidMeta =
      const VerificationMeta('entityUuid');
  @override
  late final GeneratedColumn<String> entityUuid = GeneratedColumn<String>(
      'entity_uuid', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _operationMeta =
      const VerificationMeta('operation');
  @override
  late final GeneratedColumn<String> operation = GeneratedColumn<String>(
      'operation', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant('upsert'));
  static const VerificationMeta _dependencyEntityUuidMeta =
      const VerificationMeta('dependencyEntityUuid');
  @override
  late final GeneratedColumn<String> dependencyEntityUuid =
      GeneratedColumn<String>('dependency_entity_uuid', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _payloadJsonMeta =
      const VerificationMeta('payloadJson');
  @override
  late final GeneratedColumn<String> payloadJson = GeneratedColumn<String>(
      'payload_json', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _payloadChecksumMeta =
      const VerificationMeta('payloadChecksum');
  @override
  late final GeneratedColumn<String> payloadChecksum = GeneratedColumn<String>(
      'payload_checksum', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
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
  static const VerificationMeta _acknowledgedAtMeta =
      const VerificationMeta('acknowledgedAt');
  @override
  late final GeneratedColumn<DateTime> acknowledgedAt =
      GeneratedColumn<DateTime>('acknowledged_at', aliasedName, true,
          type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _acknowledgementPayloadJsonMeta =
      const VerificationMeta('acknowledgementPayloadJson');
  @override
  late final GeneratedColumn<String> acknowledgementPayloadJson =
      GeneratedColumn<String>('acknowledgement_payload_json', aliasedName, true,
          type: DriftSqlType.string, requiredDuringInsert: false);
  static const VerificationMeta _errorMessageMeta =
      const VerificationMeta('errorMessage');
  @override
  late final GeneratedColumn<String> errorMessage = GeneratedColumn<String>(
      'error_message', aliasedName, true,
      type: DriftSqlType.string, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns => [
        id,
        ownerUserId,
        visitUuid,
        entityType,
        entityUuid,
        operation,
        dependencyEntityUuid,
        payloadJson,
        payloadChecksum,
        status,
        retryCount,
        createdAt,
        lastAttemptAt,
        acknowledgedAt,
        acknowledgementPayloadJson,
        errorMessage
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'sync_outbox';
  @override
  VerificationContext validateIntegrity(Insertable<SyncOutboxData> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('owner_user_id')) {
      context.handle(
          _ownerUserIdMeta,
          ownerUserId.isAcceptableOrUnknown(
              data['owner_user_id']!, _ownerUserIdMeta));
    } else if (isInserting) {
      context.missing(_ownerUserIdMeta);
    }
    if (data.containsKey('visit_uuid')) {
      context.handle(_visitUuidMeta,
          visitUuid.isAcceptableOrUnknown(data['visit_uuid']!, _visitUuidMeta));
    } else if (isInserting) {
      context.missing(_visitUuidMeta);
    }
    if (data.containsKey('entity_type')) {
      context.handle(
          _entityTypeMeta,
          entityType.isAcceptableOrUnknown(
              data['entity_type']!, _entityTypeMeta));
    } else if (isInserting) {
      context.missing(_entityTypeMeta);
    }
    if (data.containsKey('entity_uuid')) {
      context.handle(
          _entityUuidMeta,
          entityUuid.isAcceptableOrUnknown(
              data['entity_uuid']!, _entityUuidMeta));
    } else if (isInserting) {
      context.missing(_entityUuidMeta);
    }
    if (data.containsKey('operation')) {
      context.handle(_operationMeta,
          operation.isAcceptableOrUnknown(data['operation']!, _operationMeta));
    }
    if (data.containsKey('dependency_entity_uuid')) {
      context.handle(
          _dependencyEntityUuidMeta,
          dependencyEntityUuid.isAcceptableOrUnknown(
              data['dependency_entity_uuid']!, _dependencyEntityUuidMeta));
    }
    if (data.containsKey('payload_json')) {
      context.handle(
          _payloadJsonMeta,
          payloadJson.isAcceptableOrUnknown(
              data['payload_json']!, _payloadJsonMeta));
    } else if (isInserting) {
      context.missing(_payloadJsonMeta);
    }
    if (data.containsKey('payload_checksum')) {
      context.handle(
          _payloadChecksumMeta,
          payloadChecksum.isAcceptableOrUnknown(
              data['payload_checksum']!, _payloadChecksumMeta));
    } else if (isInserting) {
      context.missing(_payloadChecksumMeta);
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
    if (data.containsKey('acknowledged_at')) {
      context.handle(
          _acknowledgedAtMeta,
          acknowledgedAt.isAcceptableOrUnknown(
              data['acknowledged_at']!, _acknowledgedAtMeta));
    }
    if (data.containsKey('acknowledgement_payload_json')) {
      context.handle(
          _acknowledgementPayloadJsonMeta,
          acknowledgementPayloadJson.isAcceptableOrUnknown(
              data['acknowledgement_payload_json']!,
              _acknowledgementPayloadJsonMeta));
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
  List<Set<GeneratedColumn>> get uniqueKeys => [
        {entityType, entityUuid},
      ];
  @override
  SyncOutboxData map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return SyncOutboxData(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      ownerUserId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}owner_user_id'])!,
      visitUuid: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}visit_uuid'])!,
      entityType: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}entity_type'])!,
      entityUuid: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}entity_uuid'])!,
      operation: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}operation'])!,
      dependencyEntityUuid: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}dependency_entity_uuid']),
      payloadJson: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}payload_json'])!,
      payloadChecksum: attachedDatabase.typeMapping.read(
          DriftSqlType.string, data['${effectivePrefix}payload_checksum'])!,
      status: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}status'])!,
      retryCount: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}retry_count'])!,
      createdAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}created_at'])!,
      lastAttemptAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime, data['${effectivePrefix}last_attempt_at']),
      acknowledgedAt: attachedDatabase.typeMapping.read(
          DriftSqlType.dateTime, data['${effectivePrefix}acknowledged_at']),
      acknowledgementPayloadJson: attachedDatabase.typeMapping.read(
          DriftSqlType.string,
          data['${effectivePrefix}acknowledgement_payload_json']),
      errorMessage: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}error_message']),
    );
  }

  @override
  $SyncOutboxTable createAlias(String alias) {
    return $SyncOutboxTable(attachedDatabase, alias);
  }
}

class SyncOutboxData extends DataClass implements Insertable<SyncOutboxData> {
  final int id;
  final int ownerUserId;
  final String visitUuid;
  final String entityType;
  final String entityUuid;
  final String operation;
  final String? dependencyEntityUuid;
  final String payloadJson;
  final String payloadChecksum;
  final String status;
  final int retryCount;
  final DateTime createdAt;
  final DateTime? lastAttemptAt;
  final DateTime? acknowledgedAt;
  final String? acknowledgementPayloadJson;
  final String? errorMessage;
  const SyncOutboxData(
      {required this.id,
      required this.ownerUserId,
      required this.visitUuid,
      required this.entityType,
      required this.entityUuid,
      required this.operation,
      this.dependencyEntityUuid,
      required this.payloadJson,
      required this.payloadChecksum,
      required this.status,
      required this.retryCount,
      required this.createdAt,
      this.lastAttemptAt,
      this.acknowledgedAt,
      this.acknowledgementPayloadJson,
      this.errorMessage});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['owner_user_id'] = Variable<int>(ownerUserId);
    map['visit_uuid'] = Variable<String>(visitUuid);
    map['entity_type'] = Variable<String>(entityType);
    map['entity_uuid'] = Variable<String>(entityUuid);
    map['operation'] = Variable<String>(operation);
    if (!nullToAbsent || dependencyEntityUuid != null) {
      map['dependency_entity_uuid'] = Variable<String>(dependencyEntityUuid);
    }
    map['payload_json'] = Variable<String>(payloadJson);
    map['payload_checksum'] = Variable<String>(payloadChecksum);
    map['status'] = Variable<String>(status);
    map['retry_count'] = Variable<int>(retryCount);
    map['created_at'] = Variable<DateTime>(createdAt);
    if (!nullToAbsent || lastAttemptAt != null) {
      map['last_attempt_at'] = Variable<DateTime>(lastAttemptAt);
    }
    if (!nullToAbsent || acknowledgedAt != null) {
      map['acknowledged_at'] = Variable<DateTime>(acknowledgedAt);
    }
    if (!nullToAbsent || acknowledgementPayloadJson != null) {
      map['acknowledgement_payload_json'] =
          Variable<String>(acknowledgementPayloadJson);
    }
    if (!nullToAbsent || errorMessage != null) {
      map['error_message'] = Variable<String>(errorMessage);
    }
    return map;
  }

  SyncOutboxCompanion toCompanion(bool nullToAbsent) {
    return SyncOutboxCompanion(
      id: Value(id),
      ownerUserId: Value(ownerUserId),
      visitUuid: Value(visitUuid),
      entityType: Value(entityType),
      entityUuid: Value(entityUuid),
      operation: Value(operation),
      dependencyEntityUuid: dependencyEntityUuid == null && nullToAbsent
          ? const Value.absent()
          : Value(dependencyEntityUuid),
      payloadJson: Value(payloadJson),
      payloadChecksum: Value(payloadChecksum),
      status: Value(status),
      retryCount: Value(retryCount),
      createdAt: Value(createdAt),
      lastAttemptAt: lastAttemptAt == null && nullToAbsent
          ? const Value.absent()
          : Value(lastAttemptAt),
      acknowledgedAt: acknowledgedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(acknowledgedAt),
      acknowledgementPayloadJson:
          acknowledgementPayloadJson == null && nullToAbsent
              ? const Value.absent()
              : Value(acknowledgementPayloadJson),
      errorMessage: errorMessage == null && nullToAbsent
          ? const Value.absent()
          : Value(errorMessage),
    );
  }

  factory SyncOutboxData.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return SyncOutboxData(
      id: serializer.fromJson<int>(json['id']),
      ownerUserId: serializer.fromJson<int>(json['ownerUserId']),
      visitUuid: serializer.fromJson<String>(json['visitUuid']),
      entityType: serializer.fromJson<String>(json['entityType']),
      entityUuid: serializer.fromJson<String>(json['entityUuid']),
      operation: serializer.fromJson<String>(json['operation']),
      dependencyEntityUuid:
          serializer.fromJson<String?>(json['dependencyEntityUuid']),
      payloadJson: serializer.fromJson<String>(json['payloadJson']),
      payloadChecksum: serializer.fromJson<String>(json['payloadChecksum']),
      status: serializer.fromJson<String>(json['status']),
      retryCount: serializer.fromJson<int>(json['retryCount']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
      lastAttemptAt: serializer.fromJson<DateTime?>(json['lastAttemptAt']),
      acknowledgedAt: serializer.fromJson<DateTime?>(json['acknowledgedAt']),
      acknowledgementPayloadJson:
          serializer.fromJson<String?>(json['acknowledgementPayloadJson']),
      errorMessage: serializer.fromJson<String?>(json['errorMessage']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'ownerUserId': serializer.toJson<int>(ownerUserId),
      'visitUuid': serializer.toJson<String>(visitUuid),
      'entityType': serializer.toJson<String>(entityType),
      'entityUuid': serializer.toJson<String>(entityUuid),
      'operation': serializer.toJson<String>(operation),
      'dependencyEntityUuid': serializer.toJson<String?>(dependencyEntityUuid),
      'payloadJson': serializer.toJson<String>(payloadJson),
      'payloadChecksum': serializer.toJson<String>(payloadChecksum),
      'status': serializer.toJson<String>(status),
      'retryCount': serializer.toJson<int>(retryCount),
      'createdAt': serializer.toJson<DateTime>(createdAt),
      'lastAttemptAt': serializer.toJson<DateTime?>(lastAttemptAt),
      'acknowledgedAt': serializer.toJson<DateTime?>(acknowledgedAt),
      'acknowledgementPayloadJson':
          serializer.toJson<String?>(acknowledgementPayloadJson),
      'errorMessage': serializer.toJson<String?>(errorMessage),
    };
  }

  SyncOutboxData copyWith(
          {int? id,
          int? ownerUserId,
          String? visitUuid,
          String? entityType,
          String? entityUuid,
          String? operation,
          Value<String?> dependencyEntityUuid = const Value.absent(),
          String? payloadJson,
          String? payloadChecksum,
          String? status,
          int? retryCount,
          DateTime? createdAt,
          Value<DateTime?> lastAttemptAt = const Value.absent(),
          Value<DateTime?> acknowledgedAt = const Value.absent(),
          Value<String?> acknowledgementPayloadJson = const Value.absent(),
          Value<String?> errorMessage = const Value.absent()}) =>
      SyncOutboxData(
        id: id ?? this.id,
        ownerUserId: ownerUserId ?? this.ownerUserId,
        visitUuid: visitUuid ?? this.visitUuid,
        entityType: entityType ?? this.entityType,
        entityUuid: entityUuid ?? this.entityUuid,
        operation: operation ?? this.operation,
        dependencyEntityUuid: dependencyEntityUuid.present
            ? dependencyEntityUuid.value
            : this.dependencyEntityUuid,
        payloadJson: payloadJson ?? this.payloadJson,
        payloadChecksum: payloadChecksum ?? this.payloadChecksum,
        status: status ?? this.status,
        retryCount: retryCount ?? this.retryCount,
        createdAt: createdAt ?? this.createdAt,
        lastAttemptAt:
            lastAttemptAt.present ? lastAttemptAt.value : this.lastAttemptAt,
        acknowledgedAt:
            acknowledgedAt.present ? acknowledgedAt.value : this.acknowledgedAt,
        acknowledgementPayloadJson: acknowledgementPayloadJson.present
            ? acknowledgementPayloadJson.value
            : this.acknowledgementPayloadJson,
        errorMessage:
            errorMessage.present ? errorMessage.value : this.errorMessage,
      );
  SyncOutboxData copyWithCompanion(SyncOutboxCompanion data) {
    return SyncOutboxData(
      id: data.id.present ? data.id.value : this.id,
      ownerUserId:
          data.ownerUserId.present ? data.ownerUserId.value : this.ownerUserId,
      visitUuid: data.visitUuid.present ? data.visitUuid.value : this.visitUuid,
      entityType:
          data.entityType.present ? data.entityType.value : this.entityType,
      entityUuid:
          data.entityUuid.present ? data.entityUuid.value : this.entityUuid,
      operation: data.operation.present ? data.operation.value : this.operation,
      dependencyEntityUuid: data.dependencyEntityUuid.present
          ? data.dependencyEntityUuid.value
          : this.dependencyEntityUuid,
      payloadJson:
          data.payloadJson.present ? data.payloadJson.value : this.payloadJson,
      payloadChecksum: data.payloadChecksum.present
          ? data.payloadChecksum.value
          : this.payloadChecksum,
      status: data.status.present ? data.status.value : this.status,
      retryCount:
          data.retryCount.present ? data.retryCount.value : this.retryCount,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
      lastAttemptAt: data.lastAttemptAt.present
          ? data.lastAttemptAt.value
          : this.lastAttemptAt,
      acknowledgedAt: data.acknowledgedAt.present
          ? data.acknowledgedAt.value
          : this.acknowledgedAt,
      acknowledgementPayloadJson: data.acknowledgementPayloadJson.present
          ? data.acknowledgementPayloadJson.value
          : this.acknowledgementPayloadJson,
      errorMessage: data.errorMessage.present
          ? data.errorMessage.value
          : this.errorMessage,
    );
  }

  @override
  String toString() {
    return (StringBuffer('SyncOutboxData(')
          ..write('id: $id, ')
          ..write('ownerUserId: $ownerUserId, ')
          ..write('visitUuid: $visitUuid, ')
          ..write('entityType: $entityType, ')
          ..write('entityUuid: $entityUuid, ')
          ..write('operation: $operation, ')
          ..write('dependencyEntityUuid: $dependencyEntityUuid, ')
          ..write('payloadJson: $payloadJson, ')
          ..write('payloadChecksum: $payloadChecksum, ')
          ..write('status: $status, ')
          ..write('retryCount: $retryCount, ')
          ..write('createdAt: $createdAt, ')
          ..write('lastAttemptAt: $lastAttemptAt, ')
          ..write('acknowledgedAt: $acknowledgedAt, ')
          ..write('acknowledgementPayloadJson: $acknowledgementPayloadJson, ')
          ..write('errorMessage: $errorMessage')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
      id,
      ownerUserId,
      visitUuid,
      entityType,
      entityUuid,
      operation,
      dependencyEntityUuid,
      payloadJson,
      payloadChecksum,
      status,
      retryCount,
      createdAt,
      lastAttemptAt,
      acknowledgedAt,
      acknowledgementPayloadJson,
      errorMessage);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is SyncOutboxData &&
          other.id == this.id &&
          other.ownerUserId == this.ownerUserId &&
          other.visitUuid == this.visitUuid &&
          other.entityType == this.entityType &&
          other.entityUuid == this.entityUuid &&
          other.operation == this.operation &&
          other.dependencyEntityUuid == this.dependencyEntityUuid &&
          other.payloadJson == this.payloadJson &&
          other.payloadChecksum == this.payloadChecksum &&
          other.status == this.status &&
          other.retryCount == this.retryCount &&
          other.createdAt == this.createdAt &&
          other.lastAttemptAt == this.lastAttemptAt &&
          other.acknowledgedAt == this.acknowledgedAt &&
          other.acknowledgementPayloadJson == this.acknowledgementPayloadJson &&
          other.errorMessage == this.errorMessage);
}

class SyncOutboxCompanion extends UpdateCompanion<SyncOutboxData> {
  final Value<int> id;
  final Value<int> ownerUserId;
  final Value<String> visitUuid;
  final Value<String> entityType;
  final Value<String> entityUuid;
  final Value<String> operation;
  final Value<String?> dependencyEntityUuid;
  final Value<String> payloadJson;
  final Value<String> payloadChecksum;
  final Value<String> status;
  final Value<int> retryCount;
  final Value<DateTime> createdAt;
  final Value<DateTime?> lastAttemptAt;
  final Value<DateTime?> acknowledgedAt;
  final Value<String?> acknowledgementPayloadJson;
  final Value<String?> errorMessage;
  const SyncOutboxCompanion({
    this.id = const Value.absent(),
    this.ownerUserId = const Value.absent(),
    this.visitUuid = const Value.absent(),
    this.entityType = const Value.absent(),
    this.entityUuid = const Value.absent(),
    this.operation = const Value.absent(),
    this.dependencyEntityUuid = const Value.absent(),
    this.payloadJson = const Value.absent(),
    this.payloadChecksum = const Value.absent(),
    this.status = const Value.absent(),
    this.retryCount = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.lastAttemptAt = const Value.absent(),
    this.acknowledgedAt = const Value.absent(),
    this.acknowledgementPayloadJson = const Value.absent(),
    this.errorMessage = const Value.absent(),
  });
  SyncOutboxCompanion.insert({
    this.id = const Value.absent(),
    required int ownerUserId,
    required String visitUuid,
    required String entityType,
    required String entityUuid,
    this.operation = const Value.absent(),
    this.dependencyEntityUuid = const Value.absent(),
    required String payloadJson,
    required String payloadChecksum,
    this.status = const Value.absent(),
    this.retryCount = const Value.absent(),
    this.createdAt = const Value.absent(),
    this.lastAttemptAt = const Value.absent(),
    this.acknowledgedAt = const Value.absent(),
    this.acknowledgementPayloadJson = const Value.absent(),
    this.errorMessage = const Value.absent(),
  })  : ownerUserId = Value(ownerUserId),
        visitUuid = Value(visitUuid),
        entityType = Value(entityType),
        entityUuid = Value(entityUuid),
        payloadJson = Value(payloadJson),
        payloadChecksum = Value(payloadChecksum);
  static Insertable<SyncOutboxData> custom({
    Expression<int>? id,
    Expression<int>? ownerUserId,
    Expression<String>? visitUuid,
    Expression<String>? entityType,
    Expression<String>? entityUuid,
    Expression<String>? operation,
    Expression<String>? dependencyEntityUuid,
    Expression<String>? payloadJson,
    Expression<String>? payloadChecksum,
    Expression<String>? status,
    Expression<int>? retryCount,
    Expression<DateTime>? createdAt,
    Expression<DateTime>? lastAttemptAt,
    Expression<DateTime>? acknowledgedAt,
    Expression<String>? acknowledgementPayloadJson,
    Expression<String>? errorMessage,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (ownerUserId != null) 'owner_user_id': ownerUserId,
      if (visitUuid != null) 'visit_uuid': visitUuid,
      if (entityType != null) 'entity_type': entityType,
      if (entityUuid != null) 'entity_uuid': entityUuid,
      if (operation != null) 'operation': operation,
      if (dependencyEntityUuid != null)
        'dependency_entity_uuid': dependencyEntityUuid,
      if (payloadJson != null) 'payload_json': payloadJson,
      if (payloadChecksum != null) 'payload_checksum': payloadChecksum,
      if (status != null) 'status': status,
      if (retryCount != null) 'retry_count': retryCount,
      if (createdAt != null) 'created_at': createdAt,
      if (lastAttemptAt != null) 'last_attempt_at': lastAttemptAt,
      if (acknowledgedAt != null) 'acknowledged_at': acknowledgedAt,
      if (acknowledgementPayloadJson != null)
        'acknowledgement_payload_json': acknowledgementPayloadJson,
      if (errorMessage != null) 'error_message': errorMessage,
    });
  }

  SyncOutboxCompanion copyWith(
      {Value<int>? id,
      Value<int>? ownerUserId,
      Value<String>? visitUuid,
      Value<String>? entityType,
      Value<String>? entityUuid,
      Value<String>? operation,
      Value<String?>? dependencyEntityUuid,
      Value<String>? payloadJson,
      Value<String>? payloadChecksum,
      Value<String>? status,
      Value<int>? retryCount,
      Value<DateTime>? createdAt,
      Value<DateTime?>? lastAttemptAt,
      Value<DateTime?>? acknowledgedAt,
      Value<String?>? acknowledgementPayloadJson,
      Value<String?>? errorMessage}) {
    return SyncOutboxCompanion(
      id: id ?? this.id,
      ownerUserId: ownerUserId ?? this.ownerUserId,
      visitUuid: visitUuid ?? this.visitUuid,
      entityType: entityType ?? this.entityType,
      entityUuid: entityUuid ?? this.entityUuid,
      operation: operation ?? this.operation,
      dependencyEntityUuid: dependencyEntityUuid ?? this.dependencyEntityUuid,
      payloadJson: payloadJson ?? this.payloadJson,
      payloadChecksum: payloadChecksum ?? this.payloadChecksum,
      status: status ?? this.status,
      retryCount: retryCount ?? this.retryCount,
      createdAt: createdAt ?? this.createdAt,
      lastAttemptAt: lastAttemptAt ?? this.lastAttemptAt,
      acknowledgedAt: acknowledgedAt ?? this.acknowledgedAt,
      acknowledgementPayloadJson:
          acknowledgementPayloadJson ?? this.acknowledgementPayloadJson,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (ownerUserId.present) {
      map['owner_user_id'] = Variable<int>(ownerUserId.value);
    }
    if (visitUuid.present) {
      map['visit_uuid'] = Variable<String>(visitUuid.value);
    }
    if (entityType.present) {
      map['entity_type'] = Variable<String>(entityType.value);
    }
    if (entityUuid.present) {
      map['entity_uuid'] = Variable<String>(entityUuid.value);
    }
    if (operation.present) {
      map['operation'] = Variable<String>(operation.value);
    }
    if (dependencyEntityUuid.present) {
      map['dependency_entity_uuid'] =
          Variable<String>(dependencyEntityUuid.value);
    }
    if (payloadJson.present) {
      map['payload_json'] = Variable<String>(payloadJson.value);
    }
    if (payloadChecksum.present) {
      map['payload_checksum'] = Variable<String>(payloadChecksum.value);
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
    if (acknowledgedAt.present) {
      map['acknowledged_at'] = Variable<DateTime>(acknowledgedAt.value);
    }
    if (acknowledgementPayloadJson.present) {
      map['acknowledgement_payload_json'] =
          Variable<String>(acknowledgementPayloadJson.value);
    }
    if (errorMessage.present) {
      map['error_message'] = Variable<String>(errorMessage.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('SyncOutboxCompanion(')
          ..write('id: $id, ')
          ..write('ownerUserId: $ownerUserId, ')
          ..write('visitUuid: $visitUuid, ')
          ..write('entityType: $entityType, ')
          ..write('entityUuid: $entityUuid, ')
          ..write('operation: $operation, ')
          ..write('dependencyEntityUuid: $dependencyEntityUuid, ')
          ..write('payloadJson: $payloadJson, ')
          ..write('payloadChecksum: $payloadChecksum, ')
          ..write('status: $status, ')
          ..write('retryCount: $retryCount, ')
          ..write('createdAt: $createdAt, ')
          ..write('lastAttemptAt: $lastAttemptAt, ')
          ..write('acknowledgedAt: $acknowledgedAt, ')
          ..write('acknowledgementPayloadJson: $acknowledgementPayloadJson, ')
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
  late final $CaptureAssetsTable captureAssets = $CaptureAssetsTable(this);
  late final $CameraResultsTable cameraResults = $CameraResultsTable(this);
  late final $MeasuredDetailRevisionsTable measuredDetailRevisions =
      $MeasuredDetailRevisionsTable(this);
  late final $SyncOutboxTable syncOutbox = $SyncOutboxTable(this);
  late final Index ixVisitsOwnerLocalUuid = Index('ix_visits_owner_local_uuid',
      'CREATE INDEX ix_visits_owner_local_uuid ON visits (owner_user_id, local_uuid)');
  late final Index ixCaptureAssetsVisitRole = Index(
      'ix_capture_assets_visit_role',
      'CREATE INDEX ix_capture_assets_visit_role ON capture_assets (visit_id, role)');
  late final Index ixCameraResultsVisitVersion = Index(
      'ix_camera_results_visit_version',
      'CREATE INDEX ix_camera_results_visit_version ON camera_results (visit_id, version)');
  late final Index ixMeasuredRevisionsVisitRevision = Index(
      'ix_measured_revisions_visit_revision',
      'CREATE INDEX ix_measured_revisions_visit_revision ON measured_detail_revisions (visit_id, revision_number)');
  late final Index ixSyncOutboxOwnerStatusCreated = Index(
      'ix_sync_outbox_owner_status_created',
      'CREATE INDEX ix_sync_outbox_owner_status_created ON sync_outbox (owner_user_id, status, created_at)');
  late final Index ixSyncOutboxVisitType = Index('ix_sync_outbox_visit_type',
      'CREATE INDEX ix_sync_outbox_visit_type ON sync_outbox (visit_uuid, entity_type)');
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities => [
        children,
        visits,
        measurements,
        syncQueue,
        captureAssets,
        cameraResults,
        measuredDetailRevisions,
        syncOutbox,
        ixVisitsOwnerLocalUuid,
        ixCaptureAssetsVisitRole,
        ixCameraResultsVisitVersion,
        ixMeasuredRevisionsVisitRevision,
        ixSyncOutboxOwnerStatusCreated,
        ixSyncOutboxVisitType
      ];
  @override
  StreamQueryUpdateRules get streamUpdateRules => const StreamQueryUpdateRules(
        [
          WritePropagation(
            on: TableUpdateQuery.onTableName('visits',
                limitUpdateKind: UpdateKind.delete),
            result: [
              TableUpdate('capture_assets', kind: UpdateKind.delete),
            ],
          ),
          WritePropagation(
            on: TableUpdateQuery.onTableName('visits',
                limitUpdateKind: UpdateKind.delete),
            result: [
              TableUpdate('camera_results', kind: UpdateKind.delete),
            ],
          ),
          WritePropagation(
            on: TableUpdateQuery.onTableName('visits',
                limitUpdateKind: UpdateKind.delete),
            result: [
              TableUpdate('measured_detail_revisions', kind: UpdateKind.delete),
            ],
          ),
        ],
      );
}

typedef $$ChildrenTableCreateCompanionBuilder = ChildrenCompanion Function({
  Value<int> id,
  required String name,
  required String dateOfBirth,
  required String sex,
  Value<String?> guardianName,
  Value<String?> location,
  Value<int?> ownerUserId,
  Value<String?> photoPath,
  Value<bool> isArchived,
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
  Value<int?> ownerUserId,
  Value<String?> photoPath,
  Value<bool> isArchived,
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

  ColumnFilters<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get photoPath => $composableBuilder(
      column: $table.photoPath, builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get isArchived => $composableBuilder(
      column: $table.isArchived, builder: (column) => ColumnFilters(column));

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

  ColumnOrderings<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get photoPath => $composableBuilder(
      column: $table.photoPath, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get isArchived => $composableBuilder(
      column: $table.isArchived, builder: (column) => ColumnOrderings(column));

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

  GeneratedColumn<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => column);

  GeneratedColumn<String> get photoPath =>
      $composableBuilder(column: $table.photoPath, builder: (column) => column);

  GeneratedColumn<bool> get isArchived => $composableBuilder(
      column: $table.isArchived, builder: (column) => column);

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
            Value<int?> ownerUserId = const Value.absent(),
            Value<String?> photoPath = const Value.absent(),
            Value<bool> isArchived = const Value.absent(),
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
            ownerUserId: ownerUserId,
            photoPath: photoPath,
            isArchived: isArchived,
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
            Value<int?> ownerUserId = const Value.absent(),
            Value<String?> photoPath = const Value.absent(),
            Value<bool> isArchived = const Value.absent(),
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
            ownerUserId: ownerUserId,
            photoPath: photoPath,
            isArchived: isArchived,
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
  required String localUuid,
  Value<DateTime> visitDate,
  required double ageMonths,
  Value<String?> imagePath,
  Value<String?> sideImagePath,
  Value<String?> backImagePath,
  Value<String?> notes,
  Value<int?> ownerUserId,
  Value<int?> serverId,
  Value<String> entryMethod,
  Value<String?> captureState,
  Value<DateTime?> captureStartedAt,
  Value<DateTime?> captureCompletedAt,
  Value<String?> deviceMetadataJson,
  Value<String?> consentVersion,
  Value<DateTime?> consentTimestamp,
  Value<String?> consentOperatorIdentifier,
  Value<DateTime?> mediaDeletedAt,
});
typedef $$VisitsTableUpdateCompanionBuilder = VisitsCompanion Function({
  Value<int> id,
  Value<int> childId,
  Value<String> localUuid,
  Value<DateTime> visitDate,
  Value<double> ageMonths,
  Value<String?> imagePath,
  Value<String?> sideImagePath,
  Value<String?> backImagePath,
  Value<String?> notes,
  Value<int?> ownerUserId,
  Value<int?> serverId,
  Value<String> entryMethod,
  Value<String?> captureState,
  Value<DateTime?> captureStartedAt,
  Value<DateTime?> captureCompletedAt,
  Value<String?> deviceMetadataJson,
  Value<String?> consentVersion,
  Value<DateTime?> consentTimestamp,
  Value<String?> consentOperatorIdentifier,
  Value<DateTime?> mediaDeletedAt,
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

  static MultiTypedResultKey<$CaptureAssetsTable, List<CaptureAsset>>
      _captureAssetsRefsTable(_$AppDatabase db) =>
          MultiTypedResultKey.fromTable(db.captureAssets,
              aliasName:
                  $_aliasNameGenerator(db.visits.id, db.captureAssets.visitId));

  $$CaptureAssetsTableProcessedTableManager get captureAssetsRefs {
    final manager = $$CaptureAssetsTableTableManager($_db, $_db.captureAssets)
        .filter((f) => f.visitId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_captureAssetsRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }

  static MultiTypedResultKey<$CameraResultsTable, List<CameraResult>>
      _cameraResultsRefsTable(_$AppDatabase db) =>
          MultiTypedResultKey.fromTable(db.cameraResults,
              aliasName:
                  $_aliasNameGenerator(db.visits.id, db.cameraResults.visitId));

  $$CameraResultsTableProcessedTableManager get cameraResultsRefs {
    final manager = $$CameraResultsTableTableManager($_db, $_db.cameraResults)
        .filter((f) => f.visitId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_cameraResultsRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }

  static MultiTypedResultKey<$MeasuredDetailRevisionsTable,
      List<MeasuredDetailRevision>> _measuredDetailRevisionsRefsTable(
          _$AppDatabase db) =>
      MultiTypedResultKey.fromTable(db.measuredDetailRevisions,
          aliasName: $_aliasNameGenerator(
              db.visits.id, db.measuredDetailRevisions.visitId));

  $$MeasuredDetailRevisionsTableProcessedTableManager
      get measuredDetailRevisionsRefs {
    final manager = $$MeasuredDetailRevisionsTableTableManager(
            $_db, $_db.measuredDetailRevisions)
        .filter((f) => f.visitId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache =
        $_typedResult.readTableOrNull(_measuredDetailRevisionsRefsTable($_db));
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

  ColumnFilters<String> get localUuid => $composableBuilder(
      column: $table.localUuid, builder: (column) => ColumnFilters(column));

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

  ColumnFilters<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get entryMethod => $composableBuilder(
      column: $table.entryMethod, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get captureState => $composableBuilder(
      column: $table.captureState, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get captureStartedAt => $composableBuilder(
      column: $table.captureStartedAt,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get captureCompletedAt => $composableBuilder(
      column: $table.captureCompletedAt,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get deviceMetadataJson => $composableBuilder(
      column: $table.deviceMetadataJson,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get consentVersion => $composableBuilder(
      column: $table.consentVersion,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get consentTimestamp => $composableBuilder(
      column: $table.consentTimestamp,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get consentOperatorIdentifier => $composableBuilder(
      column: $table.consentOperatorIdentifier,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get mediaDeletedAt => $composableBuilder(
      column: $table.mediaDeletedAt,
      builder: (column) => ColumnFilters(column));

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

  Expression<bool> captureAssetsRefs(
      Expression<bool> Function($$CaptureAssetsTableFilterComposer f) f) {
    final $$CaptureAssetsTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.captureAssets,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$CaptureAssetsTableFilterComposer(
              $db: $db,
              $table: $db.captureAssets,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }

  Expression<bool> cameraResultsRefs(
      Expression<bool> Function($$CameraResultsTableFilterComposer f) f) {
    final $$CameraResultsTableFilterComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.cameraResults,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$CameraResultsTableFilterComposer(
              $db: $db,
              $table: $db.cameraResults,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }

  Expression<bool> measuredDetailRevisionsRefs(
      Expression<bool> Function($$MeasuredDetailRevisionsTableFilterComposer f)
          f) {
    final $$MeasuredDetailRevisionsTableFilterComposer composer =
        $composerBuilder(
            composer: this,
            getCurrentColumn: (t) => t.id,
            referencedTable: $db.measuredDetailRevisions,
            getReferencedColumn: (t) => t.visitId,
            builder: (joinBuilder,
                    {$addJoinBuilderToRootComposer,
                    $removeJoinBuilderFromRootComposer}) =>
                $$MeasuredDetailRevisionsTableFilterComposer(
                  $db: $db,
                  $table: $db.measuredDetailRevisions,
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

  ColumnOrderings<String> get localUuid => $composableBuilder(
      column: $table.localUuid, builder: (column) => ColumnOrderings(column));

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

  ColumnOrderings<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get entryMethod => $composableBuilder(
      column: $table.entryMethod, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get captureState => $composableBuilder(
      column: $table.captureState,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get captureStartedAt => $composableBuilder(
      column: $table.captureStartedAt,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get captureCompletedAt => $composableBuilder(
      column: $table.captureCompletedAt,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get deviceMetadataJson => $composableBuilder(
      column: $table.deviceMetadataJson,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get consentVersion => $composableBuilder(
      column: $table.consentVersion,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get consentTimestamp => $composableBuilder(
      column: $table.consentTimestamp,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get consentOperatorIdentifier => $composableBuilder(
      column: $table.consentOperatorIdentifier,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get mediaDeletedAt => $composableBuilder(
      column: $table.mediaDeletedAt,
      builder: (column) => ColumnOrderings(column));

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

  GeneratedColumn<String> get localUuid =>
      $composableBuilder(column: $table.localUuid, builder: (column) => column);

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

  GeneratedColumn<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => column);

  GeneratedColumn<int> get serverId =>
      $composableBuilder(column: $table.serverId, builder: (column) => column);

  GeneratedColumn<String> get entryMethod => $composableBuilder(
      column: $table.entryMethod, builder: (column) => column);

  GeneratedColumn<String> get captureState => $composableBuilder(
      column: $table.captureState, builder: (column) => column);

  GeneratedColumn<DateTime> get captureStartedAt => $composableBuilder(
      column: $table.captureStartedAt, builder: (column) => column);

  GeneratedColumn<DateTime> get captureCompletedAt => $composableBuilder(
      column: $table.captureCompletedAt, builder: (column) => column);

  GeneratedColumn<String> get deviceMetadataJson => $composableBuilder(
      column: $table.deviceMetadataJson, builder: (column) => column);

  GeneratedColumn<String> get consentVersion => $composableBuilder(
      column: $table.consentVersion, builder: (column) => column);

  GeneratedColumn<DateTime> get consentTimestamp => $composableBuilder(
      column: $table.consentTimestamp, builder: (column) => column);

  GeneratedColumn<String> get consentOperatorIdentifier => $composableBuilder(
      column: $table.consentOperatorIdentifier, builder: (column) => column);

  GeneratedColumn<DateTime> get mediaDeletedAt => $composableBuilder(
      column: $table.mediaDeletedAt, builder: (column) => column);

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

  Expression<T> captureAssetsRefs<T extends Object>(
      Expression<T> Function($$CaptureAssetsTableAnnotationComposer a) f) {
    final $$CaptureAssetsTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.captureAssets,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$CaptureAssetsTableAnnotationComposer(
              $db: $db,
              $table: $db.captureAssets,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }

  Expression<T> cameraResultsRefs<T extends Object>(
      Expression<T> Function($$CameraResultsTableAnnotationComposer a) f) {
    final $$CameraResultsTableAnnotationComposer composer = $composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $db.cameraResults,
        getReferencedColumn: (t) => t.visitId,
        builder: (joinBuilder,
                {$addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer}) =>
            $$CameraResultsTableAnnotationComposer(
              $db: $db,
              $table: $db.cameraResults,
              $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
              joinBuilder: joinBuilder,
              $removeJoinBuilderFromRootComposer:
                  $removeJoinBuilderFromRootComposer,
            ));
    return f(composer);
  }

  Expression<T> measuredDetailRevisionsRefs<T extends Object>(
      Expression<T> Function($$MeasuredDetailRevisionsTableAnnotationComposer a)
          f) {
    final $$MeasuredDetailRevisionsTableAnnotationComposer composer =
        $composerBuilder(
            composer: this,
            getCurrentColumn: (t) => t.id,
            referencedTable: $db.measuredDetailRevisions,
            getReferencedColumn: (t) => t.visitId,
            builder: (joinBuilder,
                    {$addJoinBuilderToRootComposer,
                    $removeJoinBuilderFromRootComposer}) =>
                $$MeasuredDetailRevisionsTableAnnotationComposer(
                  $db: $db,
                  $table: $db.measuredDetailRevisions,
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
        {bool childId,
        bool measurementsRefs,
        bool syncQueueRefs,
        bool captureAssetsRefs,
        bool cameraResultsRefs,
        bool measuredDetailRevisionsRefs})> {
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
            Value<String> localUuid = const Value.absent(),
            Value<DateTime> visitDate = const Value.absent(),
            Value<double> ageMonths = const Value.absent(),
            Value<String?> imagePath = const Value.absent(),
            Value<String?> sideImagePath = const Value.absent(),
            Value<String?> backImagePath = const Value.absent(),
            Value<String?> notes = const Value.absent(),
            Value<int?> ownerUserId = const Value.absent(),
            Value<int?> serverId = const Value.absent(),
            Value<String> entryMethod = const Value.absent(),
            Value<String?> captureState = const Value.absent(),
            Value<DateTime?> captureStartedAt = const Value.absent(),
            Value<DateTime?> captureCompletedAt = const Value.absent(),
            Value<String?> deviceMetadataJson = const Value.absent(),
            Value<String?> consentVersion = const Value.absent(),
            Value<DateTime?> consentTimestamp = const Value.absent(),
            Value<String?> consentOperatorIdentifier = const Value.absent(),
            Value<DateTime?> mediaDeletedAt = const Value.absent(),
          }) =>
              VisitsCompanion(
            id: id,
            childId: childId,
            localUuid: localUuid,
            visitDate: visitDate,
            ageMonths: ageMonths,
            imagePath: imagePath,
            sideImagePath: sideImagePath,
            backImagePath: backImagePath,
            notes: notes,
            ownerUserId: ownerUserId,
            serverId: serverId,
            entryMethod: entryMethod,
            captureState: captureState,
            captureStartedAt: captureStartedAt,
            captureCompletedAt: captureCompletedAt,
            deviceMetadataJson: deviceMetadataJson,
            consentVersion: consentVersion,
            consentTimestamp: consentTimestamp,
            consentOperatorIdentifier: consentOperatorIdentifier,
            mediaDeletedAt: mediaDeletedAt,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int childId,
            required String localUuid,
            Value<DateTime> visitDate = const Value.absent(),
            required double ageMonths,
            Value<String?> imagePath = const Value.absent(),
            Value<String?> sideImagePath = const Value.absent(),
            Value<String?> backImagePath = const Value.absent(),
            Value<String?> notes = const Value.absent(),
            Value<int?> ownerUserId = const Value.absent(),
            Value<int?> serverId = const Value.absent(),
            Value<String> entryMethod = const Value.absent(),
            Value<String?> captureState = const Value.absent(),
            Value<DateTime?> captureStartedAt = const Value.absent(),
            Value<DateTime?> captureCompletedAt = const Value.absent(),
            Value<String?> deviceMetadataJson = const Value.absent(),
            Value<String?> consentVersion = const Value.absent(),
            Value<DateTime?> consentTimestamp = const Value.absent(),
            Value<String?> consentOperatorIdentifier = const Value.absent(),
            Value<DateTime?> mediaDeletedAt = const Value.absent(),
          }) =>
              VisitsCompanion.insert(
            id: id,
            childId: childId,
            localUuid: localUuid,
            visitDate: visitDate,
            ageMonths: ageMonths,
            imagePath: imagePath,
            sideImagePath: sideImagePath,
            backImagePath: backImagePath,
            notes: notes,
            ownerUserId: ownerUserId,
            serverId: serverId,
            entryMethod: entryMethod,
            captureState: captureState,
            captureStartedAt: captureStartedAt,
            captureCompletedAt: captureCompletedAt,
            deviceMetadataJson: deviceMetadataJson,
            consentVersion: consentVersion,
            consentTimestamp: consentTimestamp,
            consentOperatorIdentifier: consentOperatorIdentifier,
            mediaDeletedAt: mediaDeletedAt,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) =>
                  (e.readTable(table), $$VisitsTableReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: (
              {childId = false,
              measurementsRefs = false,
              syncQueueRefs = false,
              captureAssetsRefs = false,
              cameraResultsRefs = false,
              measuredDetailRevisionsRefs = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [
                if (measurementsRefs) db.measurements,
                if (syncQueueRefs) db.syncQueue,
                if (captureAssetsRefs) db.captureAssets,
                if (cameraResultsRefs) db.cameraResults,
                if (measuredDetailRevisionsRefs) db.measuredDetailRevisions
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
                        typedResults: items),
                  if (captureAssetsRefs)
                    await $_getPrefetchedData<Visit, $VisitsTable,
                            CaptureAsset>(
                        currentTable: table,
                        referencedTable:
                            $$VisitsTableReferences._captureAssetsRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$VisitsTableReferences(db, table, p0)
                                .captureAssetsRefs,
                        referencedItemsForCurrentItem: (item,
                                referencedItems) =>
                            referencedItems.where((e) => e.visitId == item.id),
                        typedResults: items),
                  if (cameraResultsRefs)
                    await $_getPrefetchedData<Visit, $VisitsTable,
                            CameraResult>(
                        currentTable: table,
                        referencedTable:
                            $$VisitsTableReferences._cameraResultsRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$VisitsTableReferences(db, table, p0)
                                .cameraResultsRefs,
                        referencedItemsForCurrentItem: (item,
                                referencedItems) =>
                            referencedItems.where((e) => e.visitId == item.id),
                        typedResults: items),
                  if (measuredDetailRevisionsRefs)
                    await $_getPrefetchedData<Visit, $VisitsTable,
                            MeasuredDetailRevision>(
                        currentTable: table,
                        referencedTable: $$VisitsTableReferences
                            ._measuredDetailRevisionsRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$VisitsTableReferences(db, table, p0)
                                .measuredDetailRevisionsRefs,
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
        {bool childId,
        bool measurementsRefs,
        bool syncQueueRefs,
        bool captureAssetsRefs,
        bool cameraResultsRefs,
        bool measuredDetailRevisionsRefs})>;
typedef $$MeasurementsTableCreateCompanionBuilder = MeasurementsCompanion
    Function({
  Value<int> id,
  required int visitId,
  Value<double?> predictedHeightCm,
  Value<double?> predictedWeightKg,
  Value<double?> manualHeightCm,
  Value<double?> manualWeightKg,
  Value<double?> effectiveHeightCm,
  Value<double?> effectiveWeightKg,
  Value<String?> heightMethod,
  Value<String?> weightMethod,
  Value<double?> bmi,
  Value<String?> bmiStatus,
  Value<double?> hazZscore,
  Value<double?> whzZscore,
  Value<String?> hazStatus,
  Value<String?> whzStatus,
  Value<double?> confidenceScore,
  Value<double?> heightConfidence,
  Value<double?> weightConfidence,
  Value<double?> classificationConfidence,
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
  Value<String?> wastingMethod,
  Value<double?> muacCm,
  Value<String?> muacStatus,
  Value<String?> muacMethod,
  Value<bool?> muacAgeInRange,
  Value<double?> muacConfidence,
  Value<double?> muacUncertaintyLowerCm,
  Value<double?> muacUncertaintyUpperCm,
  Value<String?> muacModelVersion,
  Value<String?> muacCalibrationVersion,
  Value<bool?> muacIsDirectMeasurement,
  Value<bool?> muacRequiresConfirmation,
  Value<String?> muacReferralGuidance,
  Value<String?> combinedStatus,
  Value<String?> combinedTriggeredBy,
  Value<String?> combinedRationale,
  Value<String?> combinedMethod,
  Value<double?> combinedConfidenceScore,
  Value<String?> combinedProtocolVersion,
  Value<String?> poshanStatus,
  Value<String?> poshanTriggeredBy,
  Value<String?> classificationMethod,
  Value<String?> classificationRationale,
  Value<bool?> poshanComplete,
  Value<String?> measurementMode,
  Value<String?> oedema,
  Value<DateTime?> measuredAt,
  Value<int?> editorUserId,
  Value<String?> measuredNotes,
  Value<String?> whoAcuteStatus,
  Value<String?> whoAcuteTriggeredBy,
  Value<String?> whoAcuteRationale,
});
typedef $$MeasurementsTableUpdateCompanionBuilder = MeasurementsCompanion
    Function({
  Value<int> id,
  Value<int> visitId,
  Value<double?> predictedHeightCm,
  Value<double?> predictedWeightKg,
  Value<double?> manualHeightCm,
  Value<double?> manualWeightKg,
  Value<double?> effectiveHeightCm,
  Value<double?> effectiveWeightKg,
  Value<String?> heightMethod,
  Value<String?> weightMethod,
  Value<double?> bmi,
  Value<String?> bmiStatus,
  Value<double?> hazZscore,
  Value<double?> whzZscore,
  Value<String?> hazStatus,
  Value<String?> whzStatus,
  Value<double?> confidenceScore,
  Value<double?> heightConfidence,
  Value<double?> weightConfidence,
  Value<double?> classificationConfidence,
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
  Value<String?> wastingMethod,
  Value<double?> muacCm,
  Value<String?> muacStatus,
  Value<String?> muacMethod,
  Value<bool?> muacAgeInRange,
  Value<double?> muacConfidence,
  Value<double?> muacUncertaintyLowerCm,
  Value<double?> muacUncertaintyUpperCm,
  Value<String?> muacModelVersion,
  Value<String?> muacCalibrationVersion,
  Value<bool?> muacIsDirectMeasurement,
  Value<bool?> muacRequiresConfirmation,
  Value<String?> muacReferralGuidance,
  Value<String?> combinedStatus,
  Value<String?> combinedTriggeredBy,
  Value<String?> combinedRationale,
  Value<String?> combinedMethod,
  Value<double?> combinedConfidenceScore,
  Value<String?> combinedProtocolVersion,
  Value<String?> poshanStatus,
  Value<String?> poshanTriggeredBy,
  Value<String?> classificationMethod,
  Value<String?> classificationRationale,
  Value<bool?> poshanComplete,
  Value<String?> measurementMode,
  Value<String?> oedema,
  Value<DateTime?> measuredAt,
  Value<int?> editorUserId,
  Value<String?> measuredNotes,
  Value<String?> whoAcuteStatus,
  Value<String?> whoAcuteTriggeredBy,
  Value<String?> whoAcuteRationale,
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

  ColumnFilters<double> get effectiveHeightCm => $composableBuilder(
      column: $table.effectiveHeightCm,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get effectiveWeightKg => $composableBuilder(
      column: $table.effectiveWeightKg,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get heightMethod => $composableBuilder(
      column: $table.heightMethod, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get weightMethod => $composableBuilder(
      column: $table.weightMethod, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get bmi => $composableBuilder(
      column: $table.bmi, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get bmiStatus => $composableBuilder(
      column: $table.bmiStatus, builder: (column) => ColumnFilters(column));

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

  ColumnFilters<double> get heightConfidence => $composableBuilder(
      column: $table.heightConfidence,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get weightConfidence => $composableBuilder(
      column: $table.weightConfidence,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get classificationConfidence => $composableBuilder(
      column: $table.classificationConfidence,
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

  ColumnFilters<String> get wastingMethod => $composableBuilder(
      column: $table.wastingMethod, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get muacCm => $composableBuilder(
      column: $table.muacCm, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacStatus => $composableBuilder(
      column: $table.muacStatus, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacMethod => $composableBuilder(
      column: $table.muacMethod, builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get muacAgeInRange => $composableBuilder(
      column: $table.muacAgeInRange,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get muacConfidence => $composableBuilder(
      column: $table.muacConfidence,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get muacUncertaintyLowerCm => $composableBuilder(
      column: $table.muacUncertaintyLowerCm,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get muacUncertaintyUpperCm => $composableBuilder(
      column: $table.muacUncertaintyUpperCm,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacModelVersion => $composableBuilder(
      column: $table.muacModelVersion,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacCalibrationVersion => $composableBuilder(
      column: $table.muacCalibrationVersion,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get muacIsDirectMeasurement => $composableBuilder(
      column: $table.muacIsDirectMeasurement,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get muacRequiresConfirmation => $composableBuilder(
      column: $table.muacRequiresConfirmation,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get muacReferralGuidance => $composableBuilder(
      column: $table.muacReferralGuidance,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get combinedStatus => $composableBuilder(
      column: $table.combinedStatus,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get combinedTriggeredBy => $composableBuilder(
      column: $table.combinedTriggeredBy,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get combinedRationale => $composableBuilder(
      column: $table.combinedRationale,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get combinedMethod => $composableBuilder(
      column: $table.combinedMethod,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get combinedConfidenceScore => $composableBuilder(
      column: $table.combinedConfidenceScore,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get combinedProtocolVersion => $composableBuilder(
      column: $table.combinedProtocolVersion,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get poshanStatus => $composableBuilder(
      column: $table.poshanStatus, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get poshanTriggeredBy => $composableBuilder(
      column: $table.poshanTriggeredBy,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get classificationMethod => $composableBuilder(
      column: $table.classificationMethod,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get classificationRationale => $composableBuilder(
      column: $table.classificationRationale,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get poshanComplete => $composableBuilder(
      column: $table.poshanComplete,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get measurementMode => $composableBuilder(
      column: $table.measurementMode,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get oedema => $composableBuilder(
      column: $table.oedema, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get measuredAt => $composableBuilder(
      column: $table.measuredAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get editorUserId => $composableBuilder(
      column: $table.editorUserId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get measuredNotes => $composableBuilder(
      column: $table.measuredNotes, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get whoAcuteStatus => $composableBuilder(
      column: $table.whoAcuteStatus,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get whoAcuteTriggeredBy => $composableBuilder(
      column: $table.whoAcuteTriggeredBy,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get whoAcuteRationale => $composableBuilder(
      column: $table.whoAcuteRationale,
      builder: (column) => ColumnFilters(column));

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

  ColumnOrderings<double> get effectiveHeightCm => $composableBuilder(
      column: $table.effectiveHeightCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get effectiveWeightKg => $composableBuilder(
      column: $table.effectiveWeightKg,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get heightMethod => $composableBuilder(
      column: $table.heightMethod,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get weightMethod => $composableBuilder(
      column: $table.weightMethod,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get bmi => $composableBuilder(
      column: $table.bmi, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get bmiStatus => $composableBuilder(
      column: $table.bmiStatus, builder: (column) => ColumnOrderings(column));

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

  ColumnOrderings<double> get heightConfidence => $composableBuilder(
      column: $table.heightConfidence,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get weightConfidence => $composableBuilder(
      column: $table.weightConfidence,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get classificationConfidence => $composableBuilder(
      column: $table.classificationConfidence,
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

  ColumnOrderings<String> get wastingMethod => $composableBuilder(
      column: $table.wastingMethod,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get muacCm => $composableBuilder(
      column: $table.muacCm, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacStatus => $composableBuilder(
      column: $table.muacStatus, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacMethod => $composableBuilder(
      column: $table.muacMethod, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get muacAgeInRange => $composableBuilder(
      column: $table.muacAgeInRange,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get muacConfidence => $composableBuilder(
      column: $table.muacConfidence,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get muacUncertaintyLowerCm => $composableBuilder(
      column: $table.muacUncertaintyLowerCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get muacUncertaintyUpperCm => $composableBuilder(
      column: $table.muacUncertaintyUpperCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacModelVersion => $composableBuilder(
      column: $table.muacModelVersion,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacCalibrationVersion => $composableBuilder(
      column: $table.muacCalibrationVersion,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get muacIsDirectMeasurement => $composableBuilder(
      column: $table.muacIsDirectMeasurement,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get muacRequiresConfirmation => $composableBuilder(
      column: $table.muacRequiresConfirmation,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get muacReferralGuidance => $composableBuilder(
      column: $table.muacReferralGuidance,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get combinedStatus => $composableBuilder(
      column: $table.combinedStatus,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get combinedTriggeredBy => $composableBuilder(
      column: $table.combinedTriggeredBy,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get combinedRationale => $composableBuilder(
      column: $table.combinedRationale,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get combinedMethod => $composableBuilder(
      column: $table.combinedMethod,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get combinedConfidenceScore => $composableBuilder(
      column: $table.combinedConfidenceScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get combinedProtocolVersion => $composableBuilder(
      column: $table.combinedProtocolVersion,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get poshanStatus => $composableBuilder(
      column: $table.poshanStatus,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get poshanTriggeredBy => $composableBuilder(
      column: $table.poshanTriggeredBy,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get classificationMethod => $composableBuilder(
      column: $table.classificationMethod,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get classificationRationale => $composableBuilder(
      column: $table.classificationRationale,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get poshanComplete => $composableBuilder(
      column: $table.poshanComplete,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get measurementMode => $composableBuilder(
      column: $table.measurementMode,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get oedema => $composableBuilder(
      column: $table.oedema, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get measuredAt => $composableBuilder(
      column: $table.measuredAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get editorUserId => $composableBuilder(
      column: $table.editorUserId,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get measuredNotes => $composableBuilder(
      column: $table.measuredNotes,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get whoAcuteStatus => $composableBuilder(
      column: $table.whoAcuteStatus,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get whoAcuteTriggeredBy => $composableBuilder(
      column: $table.whoAcuteTriggeredBy,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get whoAcuteRationale => $composableBuilder(
      column: $table.whoAcuteRationale,
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

  GeneratedColumn<double> get effectiveHeightCm => $composableBuilder(
      column: $table.effectiveHeightCm, builder: (column) => column);

  GeneratedColumn<double> get effectiveWeightKg => $composableBuilder(
      column: $table.effectiveWeightKg, builder: (column) => column);

  GeneratedColumn<String> get heightMethod => $composableBuilder(
      column: $table.heightMethod, builder: (column) => column);

  GeneratedColumn<String> get weightMethod => $composableBuilder(
      column: $table.weightMethod, builder: (column) => column);

  GeneratedColumn<double> get bmi =>
      $composableBuilder(column: $table.bmi, builder: (column) => column);

  GeneratedColumn<String> get bmiStatus =>
      $composableBuilder(column: $table.bmiStatus, builder: (column) => column);

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

  GeneratedColumn<double> get heightConfidence => $composableBuilder(
      column: $table.heightConfidence, builder: (column) => column);

  GeneratedColumn<double> get weightConfidence => $composableBuilder(
      column: $table.weightConfidence, builder: (column) => column);

  GeneratedColumn<double> get classificationConfidence => $composableBuilder(
      column: $table.classificationConfidence, builder: (column) => column);

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

  GeneratedColumn<String> get wastingMethod => $composableBuilder(
      column: $table.wastingMethod, builder: (column) => column);

  GeneratedColumn<double> get muacCm =>
      $composableBuilder(column: $table.muacCm, builder: (column) => column);

  GeneratedColumn<String> get muacStatus => $composableBuilder(
      column: $table.muacStatus, builder: (column) => column);

  GeneratedColumn<String> get muacMethod => $composableBuilder(
      column: $table.muacMethod, builder: (column) => column);

  GeneratedColumn<bool> get muacAgeInRange => $composableBuilder(
      column: $table.muacAgeInRange, builder: (column) => column);

  GeneratedColumn<double> get muacConfidence => $composableBuilder(
      column: $table.muacConfidence, builder: (column) => column);

  GeneratedColumn<double> get muacUncertaintyLowerCm => $composableBuilder(
      column: $table.muacUncertaintyLowerCm, builder: (column) => column);

  GeneratedColumn<double> get muacUncertaintyUpperCm => $composableBuilder(
      column: $table.muacUncertaintyUpperCm, builder: (column) => column);

  GeneratedColumn<String> get muacModelVersion => $composableBuilder(
      column: $table.muacModelVersion, builder: (column) => column);

  GeneratedColumn<String> get muacCalibrationVersion => $composableBuilder(
      column: $table.muacCalibrationVersion, builder: (column) => column);

  GeneratedColumn<bool> get muacIsDirectMeasurement => $composableBuilder(
      column: $table.muacIsDirectMeasurement, builder: (column) => column);

  GeneratedColumn<bool> get muacRequiresConfirmation => $composableBuilder(
      column: $table.muacRequiresConfirmation, builder: (column) => column);

  GeneratedColumn<String> get muacReferralGuidance => $composableBuilder(
      column: $table.muacReferralGuidance, builder: (column) => column);

  GeneratedColumn<String> get combinedStatus => $composableBuilder(
      column: $table.combinedStatus, builder: (column) => column);

  GeneratedColumn<String> get combinedTriggeredBy => $composableBuilder(
      column: $table.combinedTriggeredBy, builder: (column) => column);

  GeneratedColumn<String> get combinedRationale => $composableBuilder(
      column: $table.combinedRationale, builder: (column) => column);

  GeneratedColumn<String> get combinedMethod => $composableBuilder(
      column: $table.combinedMethod, builder: (column) => column);

  GeneratedColumn<double> get combinedConfidenceScore => $composableBuilder(
      column: $table.combinedConfidenceScore, builder: (column) => column);

  GeneratedColumn<String> get combinedProtocolVersion => $composableBuilder(
      column: $table.combinedProtocolVersion, builder: (column) => column);

  GeneratedColumn<String> get poshanStatus => $composableBuilder(
      column: $table.poshanStatus, builder: (column) => column);

  GeneratedColumn<String> get poshanTriggeredBy => $composableBuilder(
      column: $table.poshanTriggeredBy, builder: (column) => column);

  GeneratedColumn<String> get classificationMethod => $composableBuilder(
      column: $table.classificationMethod, builder: (column) => column);

  GeneratedColumn<String> get classificationRationale => $composableBuilder(
      column: $table.classificationRationale, builder: (column) => column);

  GeneratedColumn<bool> get poshanComplete => $composableBuilder(
      column: $table.poshanComplete, builder: (column) => column);

  GeneratedColumn<String> get measurementMode => $composableBuilder(
      column: $table.measurementMode, builder: (column) => column);

  GeneratedColumn<String> get oedema =>
      $composableBuilder(column: $table.oedema, builder: (column) => column);

  GeneratedColumn<DateTime> get measuredAt => $composableBuilder(
      column: $table.measuredAt, builder: (column) => column);

  GeneratedColumn<int> get editorUserId => $composableBuilder(
      column: $table.editorUserId, builder: (column) => column);

  GeneratedColumn<String> get measuredNotes => $composableBuilder(
      column: $table.measuredNotes, builder: (column) => column);

  GeneratedColumn<String> get whoAcuteStatus => $composableBuilder(
      column: $table.whoAcuteStatus, builder: (column) => column);

  GeneratedColumn<String> get whoAcuteTriggeredBy => $composableBuilder(
      column: $table.whoAcuteTriggeredBy, builder: (column) => column);

  GeneratedColumn<String> get whoAcuteRationale => $composableBuilder(
      column: $table.whoAcuteRationale, builder: (column) => column);

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
            Value<double?> effectiveHeightCm = const Value.absent(),
            Value<double?> effectiveWeightKg = const Value.absent(),
            Value<String?> heightMethod = const Value.absent(),
            Value<String?> weightMethod = const Value.absent(),
            Value<double?> bmi = const Value.absent(),
            Value<String?> bmiStatus = const Value.absent(),
            Value<double?> hazZscore = const Value.absent(),
            Value<double?> whzZscore = const Value.absent(),
            Value<String?> hazStatus = const Value.absent(),
            Value<String?> whzStatus = const Value.absent(),
            Value<double?> confidenceScore = const Value.absent(),
            Value<double?> heightConfidence = const Value.absent(),
            Value<double?> weightConfidence = const Value.absent(),
            Value<double?> classificationConfidence = const Value.absent(),
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
            Value<String?> wastingMethod = const Value.absent(),
            Value<double?> muacCm = const Value.absent(),
            Value<String?> muacStatus = const Value.absent(),
            Value<String?> muacMethod = const Value.absent(),
            Value<bool?> muacAgeInRange = const Value.absent(),
            Value<double?> muacConfidence = const Value.absent(),
            Value<double?> muacUncertaintyLowerCm = const Value.absent(),
            Value<double?> muacUncertaintyUpperCm = const Value.absent(),
            Value<String?> muacModelVersion = const Value.absent(),
            Value<String?> muacCalibrationVersion = const Value.absent(),
            Value<bool?> muacIsDirectMeasurement = const Value.absent(),
            Value<bool?> muacRequiresConfirmation = const Value.absent(),
            Value<String?> muacReferralGuidance = const Value.absent(),
            Value<String?> combinedStatus = const Value.absent(),
            Value<String?> combinedTriggeredBy = const Value.absent(),
            Value<String?> combinedRationale = const Value.absent(),
            Value<String?> combinedMethod = const Value.absent(),
            Value<double?> combinedConfidenceScore = const Value.absent(),
            Value<String?> combinedProtocolVersion = const Value.absent(),
            Value<String?> poshanStatus = const Value.absent(),
            Value<String?> poshanTriggeredBy = const Value.absent(),
            Value<String?> classificationMethod = const Value.absent(),
            Value<String?> classificationRationale = const Value.absent(),
            Value<bool?> poshanComplete = const Value.absent(),
            Value<String?> measurementMode = const Value.absent(),
            Value<String?> oedema = const Value.absent(),
            Value<DateTime?> measuredAt = const Value.absent(),
            Value<int?> editorUserId = const Value.absent(),
            Value<String?> measuredNotes = const Value.absent(),
            Value<String?> whoAcuteStatus = const Value.absent(),
            Value<String?> whoAcuteTriggeredBy = const Value.absent(),
            Value<String?> whoAcuteRationale = const Value.absent(),
          }) =>
              MeasurementsCompanion(
            id: id,
            visitId: visitId,
            predictedHeightCm: predictedHeightCm,
            predictedWeightKg: predictedWeightKg,
            manualHeightCm: manualHeightCm,
            manualWeightKg: manualWeightKg,
            effectiveHeightCm: effectiveHeightCm,
            effectiveWeightKg: effectiveWeightKg,
            heightMethod: heightMethod,
            weightMethod: weightMethod,
            bmi: bmi,
            bmiStatus: bmiStatus,
            hazZscore: hazZscore,
            whzZscore: whzZscore,
            hazStatus: hazStatus,
            whzStatus: whzStatus,
            confidenceScore: confidenceScore,
            heightConfidence: heightConfidence,
            weightConfidence: weightConfidence,
            classificationConfidence: classificationConfidence,
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
            wastingMethod: wastingMethod,
            muacCm: muacCm,
            muacStatus: muacStatus,
            muacMethod: muacMethod,
            muacAgeInRange: muacAgeInRange,
            muacConfidence: muacConfidence,
            muacUncertaintyLowerCm: muacUncertaintyLowerCm,
            muacUncertaintyUpperCm: muacUncertaintyUpperCm,
            muacModelVersion: muacModelVersion,
            muacCalibrationVersion: muacCalibrationVersion,
            muacIsDirectMeasurement: muacIsDirectMeasurement,
            muacRequiresConfirmation: muacRequiresConfirmation,
            muacReferralGuidance: muacReferralGuidance,
            combinedStatus: combinedStatus,
            combinedTriggeredBy: combinedTriggeredBy,
            combinedRationale: combinedRationale,
            combinedMethod: combinedMethod,
            combinedConfidenceScore: combinedConfidenceScore,
            combinedProtocolVersion: combinedProtocolVersion,
            poshanStatus: poshanStatus,
            poshanTriggeredBy: poshanTriggeredBy,
            classificationMethod: classificationMethod,
            classificationRationale: classificationRationale,
            poshanComplete: poshanComplete,
            measurementMode: measurementMode,
            oedema: oedema,
            measuredAt: measuredAt,
            editorUserId: editorUserId,
            measuredNotes: measuredNotes,
            whoAcuteStatus: whoAcuteStatus,
            whoAcuteTriggeredBy: whoAcuteTriggeredBy,
            whoAcuteRationale: whoAcuteRationale,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int visitId,
            Value<double?> predictedHeightCm = const Value.absent(),
            Value<double?> predictedWeightKg = const Value.absent(),
            Value<double?> manualHeightCm = const Value.absent(),
            Value<double?> manualWeightKg = const Value.absent(),
            Value<double?> effectiveHeightCm = const Value.absent(),
            Value<double?> effectiveWeightKg = const Value.absent(),
            Value<String?> heightMethod = const Value.absent(),
            Value<String?> weightMethod = const Value.absent(),
            Value<double?> bmi = const Value.absent(),
            Value<String?> bmiStatus = const Value.absent(),
            Value<double?> hazZscore = const Value.absent(),
            Value<double?> whzZscore = const Value.absent(),
            Value<String?> hazStatus = const Value.absent(),
            Value<String?> whzStatus = const Value.absent(),
            Value<double?> confidenceScore = const Value.absent(),
            Value<double?> heightConfidence = const Value.absent(),
            Value<double?> weightConfidence = const Value.absent(),
            Value<double?> classificationConfidence = const Value.absent(),
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
            Value<String?> wastingMethod = const Value.absent(),
            Value<double?> muacCm = const Value.absent(),
            Value<String?> muacStatus = const Value.absent(),
            Value<String?> muacMethod = const Value.absent(),
            Value<bool?> muacAgeInRange = const Value.absent(),
            Value<double?> muacConfidence = const Value.absent(),
            Value<double?> muacUncertaintyLowerCm = const Value.absent(),
            Value<double?> muacUncertaintyUpperCm = const Value.absent(),
            Value<String?> muacModelVersion = const Value.absent(),
            Value<String?> muacCalibrationVersion = const Value.absent(),
            Value<bool?> muacIsDirectMeasurement = const Value.absent(),
            Value<bool?> muacRequiresConfirmation = const Value.absent(),
            Value<String?> muacReferralGuidance = const Value.absent(),
            Value<String?> combinedStatus = const Value.absent(),
            Value<String?> combinedTriggeredBy = const Value.absent(),
            Value<String?> combinedRationale = const Value.absent(),
            Value<String?> combinedMethod = const Value.absent(),
            Value<double?> combinedConfidenceScore = const Value.absent(),
            Value<String?> combinedProtocolVersion = const Value.absent(),
            Value<String?> poshanStatus = const Value.absent(),
            Value<String?> poshanTriggeredBy = const Value.absent(),
            Value<String?> classificationMethod = const Value.absent(),
            Value<String?> classificationRationale = const Value.absent(),
            Value<bool?> poshanComplete = const Value.absent(),
            Value<String?> measurementMode = const Value.absent(),
            Value<String?> oedema = const Value.absent(),
            Value<DateTime?> measuredAt = const Value.absent(),
            Value<int?> editorUserId = const Value.absent(),
            Value<String?> measuredNotes = const Value.absent(),
            Value<String?> whoAcuteStatus = const Value.absent(),
            Value<String?> whoAcuteTriggeredBy = const Value.absent(),
            Value<String?> whoAcuteRationale = const Value.absent(),
          }) =>
              MeasurementsCompanion.insert(
            id: id,
            visitId: visitId,
            predictedHeightCm: predictedHeightCm,
            predictedWeightKg: predictedWeightKg,
            manualHeightCm: manualHeightCm,
            manualWeightKg: manualWeightKg,
            effectiveHeightCm: effectiveHeightCm,
            effectiveWeightKg: effectiveWeightKg,
            heightMethod: heightMethod,
            weightMethod: weightMethod,
            bmi: bmi,
            bmiStatus: bmiStatus,
            hazZscore: hazZscore,
            whzZscore: whzZscore,
            hazStatus: hazStatus,
            whzStatus: whzStatus,
            confidenceScore: confidenceScore,
            heightConfidence: heightConfidence,
            weightConfidence: weightConfidence,
            classificationConfidence: classificationConfidence,
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
            wastingMethod: wastingMethod,
            muacCm: muacCm,
            muacStatus: muacStatus,
            muacMethod: muacMethod,
            muacAgeInRange: muacAgeInRange,
            muacConfidence: muacConfidence,
            muacUncertaintyLowerCm: muacUncertaintyLowerCm,
            muacUncertaintyUpperCm: muacUncertaintyUpperCm,
            muacModelVersion: muacModelVersion,
            muacCalibrationVersion: muacCalibrationVersion,
            muacIsDirectMeasurement: muacIsDirectMeasurement,
            muacRequiresConfirmation: muacRequiresConfirmation,
            muacReferralGuidance: muacReferralGuidance,
            combinedStatus: combinedStatus,
            combinedTriggeredBy: combinedTriggeredBy,
            combinedRationale: combinedRationale,
            combinedMethod: combinedMethod,
            combinedConfidenceScore: combinedConfidenceScore,
            combinedProtocolVersion: combinedProtocolVersion,
            poshanStatus: poshanStatus,
            poshanTriggeredBy: poshanTriggeredBy,
            classificationMethod: classificationMethod,
            classificationRationale: classificationRationale,
            poshanComplete: poshanComplete,
            measurementMode: measurementMode,
            oedema: oedema,
            measuredAt: measuredAt,
            editorUserId: editorUserId,
            measuredNotes: measuredNotes,
            whoAcuteStatus: whoAcuteStatus,
            whoAcuteTriggeredBy: whoAcuteTriggeredBy,
            whoAcuteRationale: whoAcuteRationale,
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
typedef $$CaptureAssetsTableCreateCompanionBuilder = CaptureAssetsCompanion
    Function({
  Value<int> id,
  required String assetUuid,
  required int visitId,
  required String role,
  Value<String?> localPath,
  Value<int?> serverId,
  Value<String?> serverObjectId,
  required DateTime capturedAt,
  Value<int?> selectedRank,
  Value<double?> poseScore,
  Value<double?> coverageScore,
  Value<double?> orientationScore,
  Value<double?> sharpnessScore,
  Value<double?> lightingScore,
  Value<double?> overallScore,
  Value<String?> qualityVerdict,
  Value<String?> rejectionReason,
  Value<String?> qualityThresholdVersion,
  Value<int?> imageWidth,
  Value<int?> imageHeight,
  Value<int?> exifOrientation,
  Value<int?> displayOrientation,
  Value<String?> deviceCameraMetadataJson,
  Value<String> syncState,
  Value<DateTime?> serverAcknowledgedAt,
});
typedef $$CaptureAssetsTableUpdateCompanionBuilder = CaptureAssetsCompanion
    Function({
  Value<int> id,
  Value<String> assetUuid,
  Value<int> visitId,
  Value<String> role,
  Value<String?> localPath,
  Value<int?> serverId,
  Value<String?> serverObjectId,
  Value<DateTime> capturedAt,
  Value<int?> selectedRank,
  Value<double?> poseScore,
  Value<double?> coverageScore,
  Value<double?> orientationScore,
  Value<double?> sharpnessScore,
  Value<double?> lightingScore,
  Value<double?> overallScore,
  Value<String?> qualityVerdict,
  Value<String?> rejectionReason,
  Value<String?> qualityThresholdVersion,
  Value<int?> imageWidth,
  Value<int?> imageHeight,
  Value<int?> exifOrientation,
  Value<int?> displayOrientation,
  Value<String?> deviceCameraMetadataJson,
  Value<String> syncState,
  Value<DateTime?> serverAcknowledgedAt,
});

final class $$CaptureAssetsTableReferences
    extends BaseReferences<_$AppDatabase, $CaptureAssetsTable, CaptureAsset> {
  $$CaptureAssetsTableReferences(
      super.$_db, super.$_table, super.$_typedResult);

  static $VisitsTable _visitIdTable(_$AppDatabase db) => db.visits.createAlias(
      $_aliasNameGenerator(db.captureAssets.visitId, db.visits.id));

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

class $$CaptureAssetsTableFilterComposer
    extends Composer<_$AppDatabase, $CaptureAssetsTable> {
  $$CaptureAssetsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get assetUuid => $composableBuilder(
      column: $table.assetUuid, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get role => $composableBuilder(
      column: $table.role, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get localPath => $composableBuilder(
      column: $table.localPath, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get serverObjectId => $composableBuilder(
      column: $table.serverObjectId,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get capturedAt => $composableBuilder(
      column: $table.capturedAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get selectedRank => $composableBuilder(
      column: $table.selectedRank, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get poseScore => $composableBuilder(
      column: $table.poseScore, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get coverageScore => $composableBuilder(
      column: $table.coverageScore, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get orientationScore => $composableBuilder(
      column: $table.orientationScore,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get sharpnessScore => $composableBuilder(
      column: $table.sharpnessScore,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get lightingScore => $composableBuilder(
      column: $table.lightingScore, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get overallScore => $composableBuilder(
      column: $table.overallScore, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get qualityVerdict => $composableBuilder(
      column: $table.qualityVerdict,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get rejectionReason => $composableBuilder(
      column: $table.rejectionReason,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get qualityThresholdVersion => $composableBuilder(
      column: $table.qualityThresholdVersion,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get imageWidth => $composableBuilder(
      column: $table.imageWidth, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get imageHeight => $composableBuilder(
      column: $table.imageHeight, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get exifOrientation => $composableBuilder(
      column: $table.exifOrientation,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get displayOrientation => $composableBuilder(
      column: $table.displayOrientation,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get deviceCameraMetadataJson => $composableBuilder(
      column: $table.deviceCameraMetadataJson,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get syncState => $composableBuilder(
      column: $table.syncState, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get serverAcknowledgedAt => $composableBuilder(
      column: $table.serverAcknowledgedAt,
      builder: (column) => ColumnFilters(column));

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

class $$CaptureAssetsTableOrderingComposer
    extends Composer<_$AppDatabase, $CaptureAssetsTable> {
  $$CaptureAssetsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get assetUuid => $composableBuilder(
      column: $table.assetUuid, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get role => $composableBuilder(
      column: $table.role, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get localPath => $composableBuilder(
      column: $table.localPath, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get serverObjectId => $composableBuilder(
      column: $table.serverObjectId,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get capturedAt => $composableBuilder(
      column: $table.capturedAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get selectedRank => $composableBuilder(
      column: $table.selectedRank,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get poseScore => $composableBuilder(
      column: $table.poseScore, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get coverageScore => $composableBuilder(
      column: $table.coverageScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get orientationScore => $composableBuilder(
      column: $table.orientationScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get sharpnessScore => $composableBuilder(
      column: $table.sharpnessScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get lightingScore => $composableBuilder(
      column: $table.lightingScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get overallScore => $composableBuilder(
      column: $table.overallScore,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get qualityVerdict => $composableBuilder(
      column: $table.qualityVerdict,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get rejectionReason => $composableBuilder(
      column: $table.rejectionReason,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get qualityThresholdVersion => $composableBuilder(
      column: $table.qualityThresholdVersion,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get imageWidth => $composableBuilder(
      column: $table.imageWidth, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get imageHeight => $composableBuilder(
      column: $table.imageHeight, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get exifOrientation => $composableBuilder(
      column: $table.exifOrientation,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get displayOrientation => $composableBuilder(
      column: $table.displayOrientation,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get deviceCameraMetadataJson => $composableBuilder(
      column: $table.deviceCameraMetadataJson,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get syncState => $composableBuilder(
      column: $table.syncState, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get serverAcknowledgedAt => $composableBuilder(
      column: $table.serverAcknowledgedAt,
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

class $$CaptureAssetsTableAnnotationComposer
    extends Composer<_$AppDatabase, $CaptureAssetsTable> {
  $$CaptureAssetsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get assetUuid =>
      $composableBuilder(column: $table.assetUuid, builder: (column) => column);

  GeneratedColumn<String> get role =>
      $composableBuilder(column: $table.role, builder: (column) => column);

  GeneratedColumn<String> get localPath =>
      $composableBuilder(column: $table.localPath, builder: (column) => column);

  GeneratedColumn<int> get serverId =>
      $composableBuilder(column: $table.serverId, builder: (column) => column);

  GeneratedColumn<String> get serverObjectId => $composableBuilder(
      column: $table.serverObjectId, builder: (column) => column);

  GeneratedColumn<DateTime> get capturedAt => $composableBuilder(
      column: $table.capturedAt, builder: (column) => column);

  GeneratedColumn<int> get selectedRank => $composableBuilder(
      column: $table.selectedRank, builder: (column) => column);

  GeneratedColumn<double> get poseScore =>
      $composableBuilder(column: $table.poseScore, builder: (column) => column);

  GeneratedColumn<double> get coverageScore => $composableBuilder(
      column: $table.coverageScore, builder: (column) => column);

  GeneratedColumn<double> get orientationScore => $composableBuilder(
      column: $table.orientationScore, builder: (column) => column);

  GeneratedColumn<double> get sharpnessScore => $composableBuilder(
      column: $table.sharpnessScore, builder: (column) => column);

  GeneratedColumn<double> get lightingScore => $composableBuilder(
      column: $table.lightingScore, builder: (column) => column);

  GeneratedColumn<double> get overallScore => $composableBuilder(
      column: $table.overallScore, builder: (column) => column);

  GeneratedColumn<String> get qualityVerdict => $composableBuilder(
      column: $table.qualityVerdict, builder: (column) => column);

  GeneratedColumn<String> get rejectionReason => $composableBuilder(
      column: $table.rejectionReason, builder: (column) => column);

  GeneratedColumn<String> get qualityThresholdVersion => $composableBuilder(
      column: $table.qualityThresholdVersion, builder: (column) => column);

  GeneratedColumn<int> get imageWidth => $composableBuilder(
      column: $table.imageWidth, builder: (column) => column);

  GeneratedColumn<int> get imageHeight => $composableBuilder(
      column: $table.imageHeight, builder: (column) => column);

  GeneratedColumn<int> get exifOrientation => $composableBuilder(
      column: $table.exifOrientation, builder: (column) => column);

  GeneratedColumn<int> get displayOrientation => $composableBuilder(
      column: $table.displayOrientation, builder: (column) => column);

  GeneratedColumn<String> get deviceCameraMetadataJson => $composableBuilder(
      column: $table.deviceCameraMetadataJson, builder: (column) => column);

  GeneratedColumn<String> get syncState =>
      $composableBuilder(column: $table.syncState, builder: (column) => column);

  GeneratedColumn<DateTime> get serverAcknowledgedAt => $composableBuilder(
      column: $table.serverAcknowledgedAt, builder: (column) => column);

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

class $$CaptureAssetsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $CaptureAssetsTable,
    CaptureAsset,
    $$CaptureAssetsTableFilterComposer,
    $$CaptureAssetsTableOrderingComposer,
    $$CaptureAssetsTableAnnotationComposer,
    $$CaptureAssetsTableCreateCompanionBuilder,
    $$CaptureAssetsTableUpdateCompanionBuilder,
    (CaptureAsset, $$CaptureAssetsTableReferences),
    CaptureAsset,
    PrefetchHooks Function({bool visitId})> {
  $$CaptureAssetsTableTableManager(_$AppDatabase db, $CaptureAssetsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$CaptureAssetsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$CaptureAssetsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$CaptureAssetsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<String> assetUuid = const Value.absent(),
            Value<int> visitId = const Value.absent(),
            Value<String> role = const Value.absent(),
            Value<String?> localPath = const Value.absent(),
            Value<int?> serverId = const Value.absent(),
            Value<String?> serverObjectId = const Value.absent(),
            Value<DateTime> capturedAt = const Value.absent(),
            Value<int?> selectedRank = const Value.absent(),
            Value<double?> poseScore = const Value.absent(),
            Value<double?> coverageScore = const Value.absent(),
            Value<double?> orientationScore = const Value.absent(),
            Value<double?> sharpnessScore = const Value.absent(),
            Value<double?> lightingScore = const Value.absent(),
            Value<double?> overallScore = const Value.absent(),
            Value<String?> qualityVerdict = const Value.absent(),
            Value<String?> rejectionReason = const Value.absent(),
            Value<String?> qualityThresholdVersion = const Value.absent(),
            Value<int?> imageWidth = const Value.absent(),
            Value<int?> imageHeight = const Value.absent(),
            Value<int?> exifOrientation = const Value.absent(),
            Value<int?> displayOrientation = const Value.absent(),
            Value<String?> deviceCameraMetadataJson = const Value.absent(),
            Value<String> syncState = const Value.absent(),
            Value<DateTime?> serverAcknowledgedAt = const Value.absent(),
          }) =>
              CaptureAssetsCompanion(
            id: id,
            assetUuid: assetUuid,
            visitId: visitId,
            role: role,
            localPath: localPath,
            serverId: serverId,
            serverObjectId: serverObjectId,
            capturedAt: capturedAt,
            selectedRank: selectedRank,
            poseScore: poseScore,
            coverageScore: coverageScore,
            orientationScore: orientationScore,
            sharpnessScore: sharpnessScore,
            lightingScore: lightingScore,
            overallScore: overallScore,
            qualityVerdict: qualityVerdict,
            rejectionReason: rejectionReason,
            qualityThresholdVersion: qualityThresholdVersion,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            exifOrientation: exifOrientation,
            displayOrientation: displayOrientation,
            deviceCameraMetadataJson: deviceCameraMetadataJson,
            syncState: syncState,
            serverAcknowledgedAt: serverAcknowledgedAt,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required String assetUuid,
            required int visitId,
            required String role,
            Value<String?> localPath = const Value.absent(),
            Value<int?> serverId = const Value.absent(),
            Value<String?> serverObjectId = const Value.absent(),
            required DateTime capturedAt,
            Value<int?> selectedRank = const Value.absent(),
            Value<double?> poseScore = const Value.absent(),
            Value<double?> coverageScore = const Value.absent(),
            Value<double?> orientationScore = const Value.absent(),
            Value<double?> sharpnessScore = const Value.absent(),
            Value<double?> lightingScore = const Value.absent(),
            Value<double?> overallScore = const Value.absent(),
            Value<String?> qualityVerdict = const Value.absent(),
            Value<String?> rejectionReason = const Value.absent(),
            Value<String?> qualityThresholdVersion = const Value.absent(),
            Value<int?> imageWidth = const Value.absent(),
            Value<int?> imageHeight = const Value.absent(),
            Value<int?> exifOrientation = const Value.absent(),
            Value<int?> displayOrientation = const Value.absent(),
            Value<String?> deviceCameraMetadataJson = const Value.absent(),
            Value<String> syncState = const Value.absent(),
            Value<DateTime?> serverAcknowledgedAt = const Value.absent(),
          }) =>
              CaptureAssetsCompanion.insert(
            id: id,
            assetUuid: assetUuid,
            visitId: visitId,
            role: role,
            localPath: localPath,
            serverId: serverId,
            serverObjectId: serverObjectId,
            capturedAt: capturedAt,
            selectedRank: selectedRank,
            poseScore: poseScore,
            coverageScore: coverageScore,
            orientationScore: orientationScore,
            sharpnessScore: sharpnessScore,
            lightingScore: lightingScore,
            overallScore: overallScore,
            qualityVerdict: qualityVerdict,
            rejectionReason: rejectionReason,
            qualityThresholdVersion: qualityThresholdVersion,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            exifOrientation: exifOrientation,
            displayOrientation: displayOrientation,
            deviceCameraMetadataJson: deviceCameraMetadataJson,
            syncState: syncState,
            serverAcknowledgedAt: serverAcknowledgedAt,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$CaptureAssetsTableReferences(db, table, e)
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
                        $$CaptureAssetsTableReferences._visitIdTable(db),
                    referencedColumn:
                        $$CaptureAssetsTableReferences._visitIdTable(db).id,
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

typedef $$CaptureAssetsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $CaptureAssetsTable,
    CaptureAsset,
    $$CaptureAssetsTableFilterComposer,
    $$CaptureAssetsTableOrderingComposer,
    $$CaptureAssetsTableAnnotationComposer,
    $$CaptureAssetsTableCreateCompanionBuilder,
    $$CaptureAssetsTableUpdateCompanionBuilder,
    (CaptureAsset, $$CaptureAssetsTableReferences),
    CaptureAsset,
    PrefetchHooks Function({bool visitId})>;
typedef $$CameraResultsTableCreateCompanionBuilder = CameraResultsCompanion
    Function({
  Value<int> id,
  required String resultUuid,
  Value<int?> serverId,
  required int visitId,
  required int version,
  Value<String?> supersedesResultUuid,
  Value<double?> estimatedHeightCm,
  Value<double?> estimatedWeightKg,
  Value<String?> heightSource,
  Value<String?> weightSource,
  Value<double?> estimatedHaz,
  Value<double?> estimatedWhz,
  Value<String?> estimatedStuntingStatus,
  Value<String?> estimatedWastingStatus,
  Value<String?> experimentalOverallCategory,
  Value<String?> componentProbabilitiesJson,
  Value<String?> bodyProportionFeaturesJson,
  Value<String?> captureQualitySummaryJson,
  required String method,
  required String modelVersion,
  required String manifestChecksum,
  required String trainingDataLabel,
  Value<bool> nonClinical,
  Value<DateTime> createdAt,
});
typedef $$CameraResultsTableUpdateCompanionBuilder = CameraResultsCompanion
    Function({
  Value<int> id,
  Value<String> resultUuid,
  Value<int?> serverId,
  Value<int> visitId,
  Value<int> version,
  Value<String?> supersedesResultUuid,
  Value<double?> estimatedHeightCm,
  Value<double?> estimatedWeightKg,
  Value<String?> heightSource,
  Value<String?> weightSource,
  Value<double?> estimatedHaz,
  Value<double?> estimatedWhz,
  Value<String?> estimatedStuntingStatus,
  Value<String?> estimatedWastingStatus,
  Value<String?> experimentalOverallCategory,
  Value<String?> componentProbabilitiesJson,
  Value<String?> bodyProportionFeaturesJson,
  Value<String?> captureQualitySummaryJson,
  Value<String> method,
  Value<String> modelVersion,
  Value<String> manifestChecksum,
  Value<String> trainingDataLabel,
  Value<bool> nonClinical,
  Value<DateTime> createdAt,
});

final class $$CameraResultsTableReferences
    extends BaseReferences<_$AppDatabase, $CameraResultsTable, CameraResult> {
  $$CameraResultsTableReferences(
      super.$_db, super.$_table, super.$_typedResult);

  static $VisitsTable _visitIdTable(_$AppDatabase db) => db.visits.createAlias(
      $_aliasNameGenerator(db.cameraResults.visitId, db.visits.id));

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

class $$CameraResultsTableFilterComposer
    extends Composer<_$AppDatabase, $CameraResultsTable> {
  $$CameraResultsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get resultUuid => $composableBuilder(
      column: $table.resultUuid, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get version => $composableBuilder(
      column: $table.version, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get supersedesResultUuid => $composableBuilder(
      column: $table.supersedesResultUuid,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get estimatedHeightCm => $composableBuilder(
      column: $table.estimatedHeightCm,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get estimatedWeightKg => $composableBuilder(
      column: $table.estimatedWeightKg,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get heightSource => $composableBuilder(
      column: $table.heightSource, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get weightSource => $composableBuilder(
      column: $table.weightSource, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get estimatedHaz => $composableBuilder(
      column: $table.estimatedHaz, builder: (column) => ColumnFilters(column));

  ColumnFilters<double> get estimatedWhz => $composableBuilder(
      column: $table.estimatedWhz, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get estimatedStuntingStatus => $composableBuilder(
      column: $table.estimatedStuntingStatus,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get estimatedWastingStatus => $composableBuilder(
      column: $table.estimatedWastingStatus,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get experimentalOverallCategory => $composableBuilder(
      column: $table.experimentalOverallCategory,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get componentProbabilitiesJson => $composableBuilder(
      column: $table.componentProbabilitiesJson,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get bodyProportionFeaturesJson => $composableBuilder(
      column: $table.bodyProportionFeaturesJson,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get captureQualitySummaryJson => $composableBuilder(
      column: $table.captureQualitySummaryJson,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get method => $composableBuilder(
      column: $table.method, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get modelVersion => $composableBuilder(
      column: $table.modelVersion, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get manifestChecksum => $composableBuilder(
      column: $table.manifestChecksum,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get trainingDataLabel => $composableBuilder(
      column: $table.trainingDataLabel,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<bool> get nonClinical => $composableBuilder(
      column: $table.nonClinical, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnFilters(column));

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

class $$CameraResultsTableOrderingComposer
    extends Composer<_$AppDatabase, $CameraResultsTable> {
  $$CameraResultsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get resultUuid => $composableBuilder(
      column: $table.resultUuid, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get version => $composableBuilder(
      column: $table.version, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get supersedesResultUuid => $composableBuilder(
      column: $table.supersedesResultUuid,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get estimatedHeightCm => $composableBuilder(
      column: $table.estimatedHeightCm,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get estimatedWeightKg => $composableBuilder(
      column: $table.estimatedWeightKg,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get heightSource => $composableBuilder(
      column: $table.heightSource,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get weightSource => $composableBuilder(
      column: $table.weightSource,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get estimatedHaz => $composableBuilder(
      column: $table.estimatedHaz,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<double> get estimatedWhz => $composableBuilder(
      column: $table.estimatedWhz,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get estimatedStuntingStatus => $composableBuilder(
      column: $table.estimatedStuntingStatus,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get estimatedWastingStatus => $composableBuilder(
      column: $table.estimatedWastingStatus,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get experimentalOverallCategory => $composableBuilder(
      column: $table.experimentalOverallCategory,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get componentProbabilitiesJson => $composableBuilder(
      column: $table.componentProbabilitiesJson,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get bodyProportionFeaturesJson => $composableBuilder(
      column: $table.bodyProportionFeaturesJson,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get captureQualitySummaryJson => $composableBuilder(
      column: $table.captureQualitySummaryJson,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get method => $composableBuilder(
      column: $table.method, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get modelVersion => $composableBuilder(
      column: $table.modelVersion,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get manifestChecksum => $composableBuilder(
      column: $table.manifestChecksum,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get trainingDataLabel => $composableBuilder(
      column: $table.trainingDataLabel,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<bool> get nonClinical => $composableBuilder(
      column: $table.nonClinical, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnOrderings(column));

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

class $$CameraResultsTableAnnotationComposer
    extends Composer<_$AppDatabase, $CameraResultsTable> {
  $$CameraResultsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get resultUuid => $composableBuilder(
      column: $table.resultUuid, builder: (column) => column);

  GeneratedColumn<int> get serverId =>
      $composableBuilder(column: $table.serverId, builder: (column) => column);

  GeneratedColumn<int> get version =>
      $composableBuilder(column: $table.version, builder: (column) => column);

  GeneratedColumn<String> get supersedesResultUuid => $composableBuilder(
      column: $table.supersedesResultUuid, builder: (column) => column);

  GeneratedColumn<double> get estimatedHeightCm => $composableBuilder(
      column: $table.estimatedHeightCm, builder: (column) => column);

  GeneratedColumn<double> get estimatedWeightKg => $composableBuilder(
      column: $table.estimatedWeightKg, builder: (column) => column);

  GeneratedColumn<String> get heightSource => $composableBuilder(
      column: $table.heightSource, builder: (column) => column);

  GeneratedColumn<String> get weightSource => $composableBuilder(
      column: $table.weightSource, builder: (column) => column);

  GeneratedColumn<double> get estimatedHaz => $composableBuilder(
      column: $table.estimatedHaz, builder: (column) => column);

  GeneratedColumn<double> get estimatedWhz => $composableBuilder(
      column: $table.estimatedWhz, builder: (column) => column);

  GeneratedColumn<String> get estimatedStuntingStatus => $composableBuilder(
      column: $table.estimatedStuntingStatus, builder: (column) => column);

  GeneratedColumn<String> get estimatedWastingStatus => $composableBuilder(
      column: $table.estimatedWastingStatus, builder: (column) => column);

  GeneratedColumn<String> get experimentalOverallCategory => $composableBuilder(
      column: $table.experimentalOverallCategory, builder: (column) => column);

  GeneratedColumn<String> get componentProbabilitiesJson => $composableBuilder(
      column: $table.componentProbabilitiesJson, builder: (column) => column);

  GeneratedColumn<String> get bodyProportionFeaturesJson => $composableBuilder(
      column: $table.bodyProportionFeaturesJson, builder: (column) => column);

  GeneratedColumn<String> get captureQualitySummaryJson => $composableBuilder(
      column: $table.captureQualitySummaryJson, builder: (column) => column);

  GeneratedColumn<String> get method =>
      $composableBuilder(column: $table.method, builder: (column) => column);

  GeneratedColumn<String> get modelVersion => $composableBuilder(
      column: $table.modelVersion, builder: (column) => column);

  GeneratedColumn<String> get manifestChecksum => $composableBuilder(
      column: $table.manifestChecksum, builder: (column) => column);

  GeneratedColumn<String> get trainingDataLabel => $composableBuilder(
      column: $table.trainingDataLabel, builder: (column) => column);

  GeneratedColumn<bool> get nonClinical => $composableBuilder(
      column: $table.nonClinical, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

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

class $$CameraResultsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $CameraResultsTable,
    CameraResult,
    $$CameraResultsTableFilterComposer,
    $$CameraResultsTableOrderingComposer,
    $$CameraResultsTableAnnotationComposer,
    $$CameraResultsTableCreateCompanionBuilder,
    $$CameraResultsTableUpdateCompanionBuilder,
    (CameraResult, $$CameraResultsTableReferences),
    CameraResult,
    PrefetchHooks Function({bool visitId})> {
  $$CameraResultsTableTableManager(_$AppDatabase db, $CameraResultsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$CameraResultsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$CameraResultsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$CameraResultsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<String> resultUuid = const Value.absent(),
            Value<int?> serverId = const Value.absent(),
            Value<int> visitId = const Value.absent(),
            Value<int> version = const Value.absent(),
            Value<String?> supersedesResultUuid = const Value.absent(),
            Value<double?> estimatedHeightCm = const Value.absent(),
            Value<double?> estimatedWeightKg = const Value.absent(),
            Value<String?> heightSource = const Value.absent(),
            Value<String?> weightSource = const Value.absent(),
            Value<double?> estimatedHaz = const Value.absent(),
            Value<double?> estimatedWhz = const Value.absent(),
            Value<String?> estimatedStuntingStatus = const Value.absent(),
            Value<String?> estimatedWastingStatus = const Value.absent(),
            Value<String?> experimentalOverallCategory = const Value.absent(),
            Value<String?> componentProbabilitiesJson = const Value.absent(),
            Value<String?> bodyProportionFeaturesJson = const Value.absent(),
            Value<String?> captureQualitySummaryJson = const Value.absent(),
            Value<String> method = const Value.absent(),
            Value<String> modelVersion = const Value.absent(),
            Value<String> manifestChecksum = const Value.absent(),
            Value<String> trainingDataLabel = const Value.absent(),
            Value<bool> nonClinical = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
          }) =>
              CameraResultsCompanion(
            id: id,
            resultUuid: resultUuid,
            serverId: serverId,
            visitId: visitId,
            version: version,
            supersedesResultUuid: supersedesResultUuid,
            estimatedHeightCm: estimatedHeightCm,
            estimatedWeightKg: estimatedWeightKg,
            heightSource: heightSource,
            weightSource: weightSource,
            estimatedHaz: estimatedHaz,
            estimatedWhz: estimatedWhz,
            estimatedStuntingStatus: estimatedStuntingStatus,
            estimatedWastingStatus: estimatedWastingStatus,
            experimentalOverallCategory: experimentalOverallCategory,
            componentProbabilitiesJson: componentProbabilitiesJson,
            bodyProportionFeaturesJson: bodyProportionFeaturesJson,
            captureQualitySummaryJson: captureQualitySummaryJson,
            method: method,
            modelVersion: modelVersion,
            manifestChecksum: manifestChecksum,
            trainingDataLabel: trainingDataLabel,
            nonClinical: nonClinical,
            createdAt: createdAt,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required String resultUuid,
            Value<int?> serverId = const Value.absent(),
            required int visitId,
            required int version,
            Value<String?> supersedesResultUuid = const Value.absent(),
            Value<double?> estimatedHeightCm = const Value.absent(),
            Value<double?> estimatedWeightKg = const Value.absent(),
            Value<String?> heightSource = const Value.absent(),
            Value<String?> weightSource = const Value.absent(),
            Value<double?> estimatedHaz = const Value.absent(),
            Value<double?> estimatedWhz = const Value.absent(),
            Value<String?> estimatedStuntingStatus = const Value.absent(),
            Value<String?> estimatedWastingStatus = const Value.absent(),
            Value<String?> experimentalOverallCategory = const Value.absent(),
            Value<String?> componentProbabilitiesJson = const Value.absent(),
            Value<String?> bodyProportionFeaturesJson = const Value.absent(),
            Value<String?> captureQualitySummaryJson = const Value.absent(),
            required String method,
            required String modelVersion,
            required String manifestChecksum,
            required String trainingDataLabel,
            Value<bool> nonClinical = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
          }) =>
              CameraResultsCompanion.insert(
            id: id,
            resultUuid: resultUuid,
            serverId: serverId,
            visitId: visitId,
            version: version,
            supersedesResultUuid: supersedesResultUuid,
            estimatedHeightCm: estimatedHeightCm,
            estimatedWeightKg: estimatedWeightKg,
            heightSource: heightSource,
            weightSource: weightSource,
            estimatedHaz: estimatedHaz,
            estimatedWhz: estimatedWhz,
            estimatedStuntingStatus: estimatedStuntingStatus,
            estimatedWastingStatus: estimatedWastingStatus,
            experimentalOverallCategory: experimentalOverallCategory,
            componentProbabilitiesJson: componentProbabilitiesJson,
            bodyProportionFeaturesJson: bodyProportionFeaturesJson,
            captureQualitySummaryJson: captureQualitySummaryJson,
            method: method,
            modelVersion: modelVersion,
            manifestChecksum: manifestChecksum,
            trainingDataLabel: trainingDataLabel,
            nonClinical: nonClinical,
            createdAt: createdAt,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$CameraResultsTableReferences(db, table, e)
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
                        $$CameraResultsTableReferences._visitIdTable(db),
                    referencedColumn:
                        $$CameraResultsTableReferences._visitIdTable(db).id,
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

typedef $$CameraResultsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $CameraResultsTable,
    CameraResult,
    $$CameraResultsTableFilterComposer,
    $$CameraResultsTableOrderingComposer,
    $$CameraResultsTableAnnotationComposer,
    $$CameraResultsTableCreateCompanionBuilder,
    $$CameraResultsTableUpdateCompanionBuilder,
    (CameraResult, $$CameraResultsTableReferences),
    CameraResult,
    PrefetchHooks Function({bool visitId})>;
typedef $$MeasuredDetailRevisionsTableCreateCompanionBuilder
    = MeasuredDetailRevisionsCompanion Function({
  Value<int> id,
  required String revisionUuid,
  Value<int?> serverId,
  required int visitId,
  required int revisionNumber,
  required String beforeJson,
  required String afterJson,
  Value<int?> editorUserId,
  Value<DateTime> createdAt,
  Value<String?> reason,
});
typedef $$MeasuredDetailRevisionsTableUpdateCompanionBuilder
    = MeasuredDetailRevisionsCompanion Function({
  Value<int> id,
  Value<String> revisionUuid,
  Value<int?> serverId,
  Value<int> visitId,
  Value<int> revisionNumber,
  Value<String> beforeJson,
  Value<String> afterJson,
  Value<int?> editorUserId,
  Value<DateTime> createdAt,
  Value<String?> reason,
});

final class $$MeasuredDetailRevisionsTableReferences extends BaseReferences<
    _$AppDatabase, $MeasuredDetailRevisionsTable, MeasuredDetailRevision> {
  $$MeasuredDetailRevisionsTableReferences(
      super.$_db, super.$_table, super.$_typedResult);

  static $VisitsTable _visitIdTable(_$AppDatabase db) => db.visits.createAlias(
      $_aliasNameGenerator(db.measuredDetailRevisions.visitId, db.visits.id));

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

class $$MeasuredDetailRevisionsTableFilterComposer
    extends Composer<_$AppDatabase, $MeasuredDetailRevisionsTable> {
  $$MeasuredDetailRevisionsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get revisionUuid => $composableBuilder(
      column: $table.revisionUuid, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get revisionNumber => $composableBuilder(
      column: $table.revisionNumber,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get beforeJson => $composableBuilder(
      column: $table.beforeJson, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get afterJson => $composableBuilder(
      column: $table.afterJson, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get editorUserId => $composableBuilder(
      column: $table.editorUserId, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get reason => $composableBuilder(
      column: $table.reason, builder: (column) => ColumnFilters(column));

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

class $$MeasuredDetailRevisionsTableOrderingComposer
    extends Composer<_$AppDatabase, $MeasuredDetailRevisionsTable> {
  $$MeasuredDetailRevisionsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get revisionUuid => $composableBuilder(
      column: $table.revisionUuid,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get serverId => $composableBuilder(
      column: $table.serverId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get revisionNumber => $composableBuilder(
      column: $table.revisionNumber,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get beforeJson => $composableBuilder(
      column: $table.beforeJson, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get afterJson => $composableBuilder(
      column: $table.afterJson, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get editorUserId => $composableBuilder(
      column: $table.editorUserId,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get reason => $composableBuilder(
      column: $table.reason, builder: (column) => ColumnOrderings(column));

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

class $$MeasuredDetailRevisionsTableAnnotationComposer
    extends Composer<_$AppDatabase, $MeasuredDetailRevisionsTable> {
  $$MeasuredDetailRevisionsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get revisionUuid => $composableBuilder(
      column: $table.revisionUuid, builder: (column) => column);

  GeneratedColumn<int> get serverId =>
      $composableBuilder(column: $table.serverId, builder: (column) => column);

  GeneratedColumn<int> get revisionNumber => $composableBuilder(
      column: $table.revisionNumber, builder: (column) => column);

  GeneratedColumn<String> get beforeJson => $composableBuilder(
      column: $table.beforeJson, builder: (column) => column);

  GeneratedColumn<String> get afterJson =>
      $composableBuilder(column: $table.afterJson, builder: (column) => column);

  GeneratedColumn<int> get editorUserId => $composableBuilder(
      column: $table.editorUserId, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  GeneratedColumn<String> get reason =>
      $composableBuilder(column: $table.reason, builder: (column) => column);

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

class $$MeasuredDetailRevisionsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $MeasuredDetailRevisionsTable,
    MeasuredDetailRevision,
    $$MeasuredDetailRevisionsTableFilterComposer,
    $$MeasuredDetailRevisionsTableOrderingComposer,
    $$MeasuredDetailRevisionsTableAnnotationComposer,
    $$MeasuredDetailRevisionsTableCreateCompanionBuilder,
    $$MeasuredDetailRevisionsTableUpdateCompanionBuilder,
    (MeasuredDetailRevision, $$MeasuredDetailRevisionsTableReferences),
    MeasuredDetailRevision,
    PrefetchHooks Function({bool visitId})> {
  $$MeasuredDetailRevisionsTableTableManager(
      _$AppDatabase db, $MeasuredDetailRevisionsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$MeasuredDetailRevisionsTableFilterComposer(
                  $db: db, $table: table),
          createOrderingComposer: () =>
              $$MeasuredDetailRevisionsTableOrderingComposer(
                  $db: db, $table: table),
          createComputedFieldComposer: () =>
              $$MeasuredDetailRevisionsTableAnnotationComposer(
                  $db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<String> revisionUuid = const Value.absent(),
            Value<int?> serverId = const Value.absent(),
            Value<int> visitId = const Value.absent(),
            Value<int> revisionNumber = const Value.absent(),
            Value<String> beforeJson = const Value.absent(),
            Value<String> afterJson = const Value.absent(),
            Value<int?> editorUserId = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<String?> reason = const Value.absent(),
          }) =>
              MeasuredDetailRevisionsCompanion(
            id: id,
            revisionUuid: revisionUuid,
            serverId: serverId,
            visitId: visitId,
            revisionNumber: revisionNumber,
            beforeJson: beforeJson,
            afterJson: afterJson,
            editorUserId: editorUserId,
            createdAt: createdAt,
            reason: reason,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required String revisionUuid,
            Value<int?> serverId = const Value.absent(),
            required int visitId,
            required int revisionNumber,
            required String beforeJson,
            required String afterJson,
            Value<int?> editorUserId = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<String?> reason = const Value.absent(),
          }) =>
              MeasuredDetailRevisionsCompanion.insert(
            id: id,
            revisionUuid: revisionUuid,
            serverId: serverId,
            visitId: visitId,
            revisionNumber: revisionNumber,
            beforeJson: beforeJson,
            afterJson: afterJson,
            editorUserId: editorUserId,
            createdAt: createdAt,
            reason: reason,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$MeasuredDetailRevisionsTableReferences(db, table, e)
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
                    referencedTable: $$MeasuredDetailRevisionsTableReferences
                        ._visitIdTable(db),
                    referencedColumn: $$MeasuredDetailRevisionsTableReferences
                        ._visitIdTable(db)
                        .id,
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

typedef $$MeasuredDetailRevisionsTableProcessedTableManager
    = ProcessedTableManager<
        _$AppDatabase,
        $MeasuredDetailRevisionsTable,
        MeasuredDetailRevision,
        $$MeasuredDetailRevisionsTableFilterComposer,
        $$MeasuredDetailRevisionsTableOrderingComposer,
        $$MeasuredDetailRevisionsTableAnnotationComposer,
        $$MeasuredDetailRevisionsTableCreateCompanionBuilder,
        $$MeasuredDetailRevisionsTableUpdateCompanionBuilder,
        (MeasuredDetailRevision, $$MeasuredDetailRevisionsTableReferences),
        MeasuredDetailRevision,
        PrefetchHooks Function({bool visitId})>;
typedef $$SyncOutboxTableCreateCompanionBuilder = SyncOutboxCompanion Function({
  Value<int> id,
  required int ownerUserId,
  required String visitUuid,
  required String entityType,
  required String entityUuid,
  Value<String> operation,
  Value<String?> dependencyEntityUuid,
  required String payloadJson,
  required String payloadChecksum,
  Value<String> status,
  Value<int> retryCount,
  Value<DateTime> createdAt,
  Value<DateTime?> lastAttemptAt,
  Value<DateTime?> acknowledgedAt,
  Value<String?> acknowledgementPayloadJson,
  Value<String?> errorMessage,
});
typedef $$SyncOutboxTableUpdateCompanionBuilder = SyncOutboxCompanion Function({
  Value<int> id,
  Value<int> ownerUserId,
  Value<String> visitUuid,
  Value<String> entityType,
  Value<String> entityUuid,
  Value<String> operation,
  Value<String?> dependencyEntityUuid,
  Value<String> payloadJson,
  Value<String> payloadChecksum,
  Value<String> status,
  Value<int> retryCount,
  Value<DateTime> createdAt,
  Value<DateTime?> lastAttemptAt,
  Value<DateTime?> acknowledgedAt,
  Value<String?> acknowledgementPayloadJson,
  Value<String?> errorMessage,
});

class $$SyncOutboxTableFilterComposer
    extends Composer<_$AppDatabase, $SyncOutboxTable> {
  $$SyncOutboxTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get visitUuid => $composableBuilder(
      column: $table.visitUuid, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get entityType => $composableBuilder(
      column: $table.entityType, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get entityUuid => $composableBuilder(
      column: $table.entityUuid, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get operation => $composableBuilder(
      column: $table.operation, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get dependencyEntityUuid => $composableBuilder(
      column: $table.dependencyEntityUuid,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get payloadJson => $composableBuilder(
      column: $table.payloadJson, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get payloadChecksum => $composableBuilder(
      column: $table.payloadChecksum,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get status => $composableBuilder(
      column: $table.status, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get lastAttemptAt => $composableBuilder(
      column: $table.lastAttemptAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get acknowledgedAt => $composableBuilder(
      column: $table.acknowledgedAt,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get acknowledgementPayloadJson => $composableBuilder(
      column: $table.acknowledgementPayloadJson,
      builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get errorMessage => $composableBuilder(
      column: $table.errorMessage, builder: (column) => ColumnFilters(column));
}

class $$SyncOutboxTableOrderingComposer
    extends Composer<_$AppDatabase, $SyncOutboxTable> {
  $$SyncOutboxTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
      column: $table.id, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get visitUuid => $composableBuilder(
      column: $table.visitUuid, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get entityType => $composableBuilder(
      column: $table.entityType, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get entityUuid => $composableBuilder(
      column: $table.entityUuid, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get operation => $composableBuilder(
      column: $table.operation, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get dependencyEntityUuid => $composableBuilder(
      column: $table.dependencyEntityUuid,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get payloadJson => $composableBuilder(
      column: $table.payloadJson, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get payloadChecksum => $composableBuilder(
      column: $table.payloadChecksum,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get status => $composableBuilder(
      column: $table.status, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
      column: $table.createdAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get lastAttemptAt => $composableBuilder(
      column: $table.lastAttemptAt,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get acknowledgedAt => $composableBuilder(
      column: $table.acknowledgedAt,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get acknowledgementPayloadJson => $composableBuilder(
      column: $table.acknowledgementPayloadJson,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get errorMessage => $composableBuilder(
      column: $table.errorMessage,
      builder: (column) => ColumnOrderings(column));
}

class $$SyncOutboxTableAnnotationComposer
    extends Composer<_$AppDatabase, $SyncOutboxTable> {
  $$SyncOutboxTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<int> get ownerUserId => $composableBuilder(
      column: $table.ownerUserId, builder: (column) => column);

  GeneratedColumn<String> get visitUuid =>
      $composableBuilder(column: $table.visitUuid, builder: (column) => column);

  GeneratedColumn<String> get entityType => $composableBuilder(
      column: $table.entityType, builder: (column) => column);

  GeneratedColumn<String> get entityUuid => $composableBuilder(
      column: $table.entityUuid, builder: (column) => column);

  GeneratedColumn<String> get operation =>
      $composableBuilder(column: $table.operation, builder: (column) => column);

  GeneratedColumn<String> get dependencyEntityUuid => $composableBuilder(
      column: $table.dependencyEntityUuid, builder: (column) => column);

  GeneratedColumn<String> get payloadJson => $composableBuilder(
      column: $table.payloadJson, builder: (column) => column);

  GeneratedColumn<String> get payloadChecksum => $composableBuilder(
      column: $table.payloadChecksum, builder: (column) => column);

  GeneratedColumn<String> get status =>
      $composableBuilder(column: $table.status, builder: (column) => column);

  GeneratedColumn<int> get retryCount => $composableBuilder(
      column: $table.retryCount, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  GeneratedColumn<DateTime> get lastAttemptAt => $composableBuilder(
      column: $table.lastAttemptAt, builder: (column) => column);

  GeneratedColumn<DateTime> get acknowledgedAt => $composableBuilder(
      column: $table.acknowledgedAt, builder: (column) => column);

  GeneratedColumn<String> get acknowledgementPayloadJson => $composableBuilder(
      column: $table.acknowledgementPayloadJson, builder: (column) => column);

  GeneratedColumn<String> get errorMessage => $composableBuilder(
      column: $table.errorMessage, builder: (column) => column);
}

class $$SyncOutboxTableTableManager extends RootTableManager<
    _$AppDatabase,
    $SyncOutboxTable,
    SyncOutboxData,
    $$SyncOutboxTableFilterComposer,
    $$SyncOutboxTableOrderingComposer,
    $$SyncOutboxTableAnnotationComposer,
    $$SyncOutboxTableCreateCompanionBuilder,
    $$SyncOutboxTableUpdateCompanionBuilder,
    (
      SyncOutboxData,
      BaseReferences<_$AppDatabase, $SyncOutboxTable, SyncOutboxData>
    ),
    SyncOutboxData,
    PrefetchHooks Function()> {
  $$SyncOutboxTableTableManager(_$AppDatabase db, $SyncOutboxTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$SyncOutboxTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$SyncOutboxTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$SyncOutboxTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> ownerUserId = const Value.absent(),
            Value<String> visitUuid = const Value.absent(),
            Value<String> entityType = const Value.absent(),
            Value<String> entityUuid = const Value.absent(),
            Value<String> operation = const Value.absent(),
            Value<String?> dependencyEntityUuid = const Value.absent(),
            Value<String> payloadJson = const Value.absent(),
            Value<String> payloadChecksum = const Value.absent(),
            Value<String> status = const Value.absent(),
            Value<int> retryCount = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<DateTime?> lastAttemptAt = const Value.absent(),
            Value<DateTime?> acknowledgedAt = const Value.absent(),
            Value<String?> acknowledgementPayloadJson = const Value.absent(),
            Value<String?> errorMessage = const Value.absent(),
          }) =>
              SyncOutboxCompanion(
            id: id,
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
            entityType: entityType,
            entityUuid: entityUuid,
            operation: operation,
            dependencyEntityUuid: dependencyEntityUuid,
            payloadJson: payloadJson,
            payloadChecksum: payloadChecksum,
            status: status,
            retryCount: retryCount,
            createdAt: createdAt,
            lastAttemptAt: lastAttemptAt,
            acknowledgedAt: acknowledgedAt,
            acknowledgementPayloadJson: acknowledgementPayloadJson,
            errorMessage: errorMessage,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int ownerUserId,
            required String visitUuid,
            required String entityType,
            required String entityUuid,
            Value<String> operation = const Value.absent(),
            Value<String?> dependencyEntityUuid = const Value.absent(),
            required String payloadJson,
            required String payloadChecksum,
            Value<String> status = const Value.absent(),
            Value<int> retryCount = const Value.absent(),
            Value<DateTime> createdAt = const Value.absent(),
            Value<DateTime?> lastAttemptAt = const Value.absent(),
            Value<DateTime?> acknowledgedAt = const Value.absent(),
            Value<String?> acknowledgementPayloadJson = const Value.absent(),
            Value<String?> errorMessage = const Value.absent(),
          }) =>
              SyncOutboxCompanion.insert(
            id: id,
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
            entityType: entityType,
            entityUuid: entityUuid,
            operation: operation,
            dependencyEntityUuid: dependencyEntityUuid,
            payloadJson: payloadJson,
            payloadChecksum: payloadChecksum,
            status: status,
            retryCount: retryCount,
            createdAt: createdAt,
            lastAttemptAt: lastAttemptAt,
            acknowledgedAt: acknowledgedAt,
            acknowledgementPayloadJson: acknowledgementPayloadJson,
            errorMessage: errorMessage,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ));
}

typedef $$SyncOutboxTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $SyncOutboxTable,
    SyncOutboxData,
    $$SyncOutboxTableFilterComposer,
    $$SyncOutboxTableOrderingComposer,
    $$SyncOutboxTableAnnotationComposer,
    $$SyncOutboxTableCreateCompanionBuilder,
    $$SyncOutboxTableUpdateCompanionBuilder,
    (
      SyncOutboxData,
      BaseReferences<_$AppDatabase, $SyncOutboxTable, SyncOutboxData>
    ),
    SyncOutboxData,
    PrefetchHooks Function()>;

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
  $$CaptureAssetsTableTableManager get captureAssets =>
      $$CaptureAssetsTableTableManager(_db, _db.captureAssets);
  $$CameraResultsTableTableManager get cameraResults =>
      $$CameraResultsTableTableManager(_db, _db.cameraResults);
  $$MeasuredDetailRevisionsTableTableManager get measuredDetailRevisions =>
      $$MeasuredDetailRevisionsTableTableManager(
          _db, _db.measuredDetailRevisions);
  $$SyncOutboxTableTableManager get syncOutbox =>
      $$SyncOutboxTableTableManager(_db, _db.syncOutbox);
}
