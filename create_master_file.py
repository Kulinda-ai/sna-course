import os

def create_master_code_file(root_dir, output_file="all_code_combined.py"):
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, root_dir)

                    out.write(f"# === {rel_path} ===\n\n")

                    with open(file_path, "r", encoding="utf-8") as f:
                        contents = f.read()

                    out.write(contents.strip() + "\n\n\n")

    print(f"✅ Combined code written to '{output_file}'")

# 🔧 Set to the root directory of your project
project_root = "./"  # or your actual folder path
create_master_code_file(project_root)
