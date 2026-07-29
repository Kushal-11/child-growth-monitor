import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../database/database.dart';
import '../../l10n/l10n_provider.dart';
import '../../providers/auth_provider.dart';
import '../../providers/database_provider.dart';
import '../shared/app_scaffold.dart';

class ChildrenListScreen extends ConsumerStatefulWidget {
  const ChildrenListScreen({super.key});

  @override
  ConsumerState<ChildrenListScreen> createState() => _ChildrenListScreenState();
}

class _ChildrenListScreenState extends ConsumerState<ChildrenListScreen> {
  String _searchQuery = '';

  @override
  Widget build(BuildContext context) {
    final userId = ref.watch(authProvider).user?.id;

    return AppScaffold(
      currentIndex: 1,
      child: Column(
        children: [
          // Header + search + new-child action
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  t('registered_children', ref),
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 12),
                TextField(
                  decoration: InputDecoration(
                    hintText: t('search_children_placeholder', ref),
                    prefixIcon: const Icon(Icons.search),
                    border: const OutlineInputBorder(),
                  ),
                  onChanged: (v) => setState(() => _searchQuery = v),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    key: const Key('new_child_btn'),
                    onPressed: () => context.go('/children/new'),
                    icon: const Icon(Icons.add),
                    label: Text(t('new_child', ref)),
                  ),
                ),
              ],
            ),
          ),

          // Children list (local, owner-scoped, offline-first)
          Expanded(
            child: userId == null
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Text(t('not_signed_in', ref)),
                    ),
                  )
                : StreamBuilder<List<ChildrenData>>(
                    stream: ref
                        .watch(childDaoProvider)
                        .watchForOwner(userId, search: _searchQuery),
                    builder: (context, snapshot) {
                      if (snapshot.hasError) {
                        return Center(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(
                              snapshot.error.toString(),
                              style: const TextStyle(color: Colors.red),
                            ),
                          ),
                        );
                      }
                      if (!snapshot.hasData) {
                        return const Center(child: CircularProgressIndicator());
                      }

                      final children = snapshot.data!;
                      if (children.isEmpty) {
                        return Center(
                          child: Text(
                            _searchQuery.isEmpty
                                ? t('empty_children', ref)
                                : t('search_no_results', ref),
                          ),
                        );
                      }

                      return ListView.separated(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        itemCount: children.length,
                        separatorBuilder: (_, __) => const Divider(height: 1),
                        itemBuilder: (context, index) {
                          final c = children[index];
                          return Dismissible(
                            key: ValueKey('child_${c.id}'),
                            direction: DismissDirection.endToStart,
                            background: Container(
                              alignment: Alignment.centerRight,
                              color: Colors.red,
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 20),
                              child: const Icon(Icons.archive,
                                  color: Colors.white),
                            ),
                            confirmDismiss: (_) => _confirmArchive(c),
                            onDismissed: (_) => ref
                                .read(childDaoProvider)
                                .setArchived(c.id, true),
                            child: ListTile(
                              title: Text(c.name),
                              subtitle: Text(
                                '${t('th_dob', ref)}: ${c.dateOfBirth}  •  '
                                '${t('th_sex', ref)}: ${c.sex}',
                              ),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () => context.go('/children/${c.id}'),
                            ),
                          );
                        },
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Future<bool> _confirmArchive(ChildrenData c) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(t('archive_child_title', ref)),
        content: Text('${t('archive_child_confirm', ref)}\n\n${c.name}'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(t('cancel', ref)),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: Text(t('archive_action', ref)),
          ),
        ],
      ),
    );
    return confirmed ?? false;
  }
}
