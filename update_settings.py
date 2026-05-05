import os
import glob

# Find all yaml files
for filepath in glob.glob('**/*.yaml', recursive=True):
    with open(filepath, 'r') as f:
        content = f.read()

    replacement = (
        "simulation_filename_suffix: _results    # name suffix of simulation file\n"
        "  simulation_filename_prefix: ''          # name prefix of simulation file"
    )

    old_strings = [
        'simulation_filename_prefix: RES_',
        'fn_sim: RES_'
    ]

    updated = False
    for old_string in old_strings:
        if old_string in content:
            content = content.replace(old_string, replacement)
            updated = True

    if updated:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f'✅ Updated: {filepath}')

print('✅ Done.')
