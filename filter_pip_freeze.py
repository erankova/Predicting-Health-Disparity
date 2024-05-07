import subprocess

def clean_pip_freeze(output_file='requirements.txt'):
    """Generates a clean requirements file excluding local path references."""
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
    clean_lines = []

    for line in result.stdout.splitlines():
        # Exclude lines that include local file paths or editable installs
        if ' @ ' not in line and 'file://' not in line:
            clean_lines.append(line)

    with open(output_file, 'w') as file:
        file.write('\n'.join(clean_lines) + '\n')

clean_pip_freeze()
