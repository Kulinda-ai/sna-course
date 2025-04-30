# FILE: annotate_files.py
# 🔧 Utility script for automatically adding file header comments to .py files

import os

def annotate_py_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)

                # Read file contents
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Skip files already annotated
                header_comment = f"# FILE: {filename}"
                if lines and lines[0].strip() == header_comment:
                    continue

                # Insert filename header and spacing
                lines = [header_comment + '\n', '\n'] + lines

                # Overwrite file with updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                print(f"✅ Annotated: {file_path}")

# 🔧 Set the root directory where your .py files are located
project_root = "./"  # Use "." to annotate all Python files in the current folder and subfolders
annotate_py_files(project_root)
