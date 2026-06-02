import 'package:flutter/material.dart';

class ChildFormScreen extends StatelessWidget {
  const ChildFormScreen({super.key, this.childId});
  final int? childId;

  @override
  Widget build(BuildContext context) =>
      const Scaffold(body: Center(child: Text('Child form (stub)')));
}
