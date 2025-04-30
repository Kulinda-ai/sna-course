# 🔧 Utility script for generating a single master .py file containing all code from a project

import os

def create_master_code_file(root_dir, output_file="all_code_combined.py"):
    """
    Combines all .py files under the specified root_dir into a single master script.
    Each file is separated and labeled by its relative path.
    """
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, root_dir)

                    # Annotate each file block
                    out.write(f"# === {rel_path} ===\n\n")

                    # Read and write content
                    with open(file_path, "r", encoding="utf-8") as f:
                        contents = f.read()
                    out.write(contents.strip() + "\n\n\n")

    print(f"✅ Combined code written to '{output_file}'")

# 🔧 Set the root directory of your project or codebase
project_root = "./"  # Change to desired base folder
create_master_code_file(project_root)
