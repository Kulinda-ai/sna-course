# FILE: annotate_files.py

import os

def annotate_py_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Check if first line already contains the filename
                header_comment = f"# FILE: {filename}"
                if lines and lines[0].strip() == header_comment:
                    continue  # Already annotated

                # Prepend filename comment and a blank line
                lines = [header_comment + '\n', '\n'] + lines

                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                print(f"✅ Annotated: {file_path}")

# 🔧 Replace with your project root directory
project_root = "./"  # Current directory
annotate_py_files(project_root)