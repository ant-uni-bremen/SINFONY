import os
import glob

# Directory containing the result files
results_dir = 'models/speechcommands'

prefixes = ['working_memory_opt_', 'memory_opt_',
            'working_memory_', 'memory_', 'opt_']

# for filepath in glob.glob(os.path.join(results_dir, 'RES_*.npz')):
#     dirpath = os.path.dirname(filepath)
#     filename = os.path.basename(filepath)

#     # Remove RES_ prefix and .npz suffix
#     stem = filename.removeprefix('RES_').removesuffix('.npz')

#     # Check for known prefixes and move them to the end
#     moved_prefix = ''
#     for prefix in prefixes:
#         if stem.startswith(prefix):
#             stem = stem.removeprefix(prefix)
#             moved_prefix = '_' + prefix.removesuffix('_')
#             break

#     # RES_memory_filename.npz -> filename_results_memory.npz
#     new_filename = stem + '_results' + moved_prefix + '.npz'
#     new_filepath = os.path.join(dirpath, new_filename)

#     print(f'{filename} -> {new_filename}')
#     os.rename(filepath, new_filepath)

# print('✅ Done.')

for filepath in glob.glob(os.path.join(results_dir, '*_results.npz')):
    dirpath = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    stem = filename.removesuffix('_results.npz')

    # Check for known prefixes and move them to the end
    moved_prefix = ''
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem.removeprefix(prefix)
            moved_prefix = '_' + prefix.removesuffix('_')
            break

    if not moved_prefix:
        continue  # nothing to do, skip

    # filename_results.npz -> filename_results_memory.npz
    new_filename = stem + '_results' + moved_prefix + '.npz'
    new_filepath = os.path.join(dirpath, new_filename)

    print(f'{filename} -> {new_filename}')
    os.rename(filepath, new_filepath)

print('✅ Done.')
