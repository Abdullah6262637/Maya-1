import os

def walk_and_bundle(root_dir, output_file, max_file_size_kb=500):
    ignored_dirs = {'.git', '.venv', 'venv', 'target', 'node_modules', '__pycache__', 'checkpoints'}
    ignored_exts = {'.pt', '.bin', '.db', '.log', '.zip', '.png', '.jpg', '.jpeg', '.gif', '.ico'}
    
    # We will output as a markdown file with file blocks
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("# Maya-1 Codebase Bundle\n")
        out.write("This file contains the key source code files of the Maya-1 project for LLM analysis.\n\n")
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune ignored directories in-place
            dirnames[:] = [d for d in dirnames if d not in ignored_dirs and not d.startswith('.')]
            
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Check extension
                ext = os.path.splitext(file)[1].lower()
                if ext in ignored_exts:
                    continue
                
                # Ignore specific large files
                if file == "ham_veri.txt" or file == "mock_data.bin":
                    continue
                    
                # Skip files that are too large
                try:
                    size_kb = os.path.getsize(file_path) / 1024
                    if size_kb > max_file_size_kb:
                        print(f"Skipping large file: {rel_path} ({size_kb:.1f} KB)")
                        continue
                except OSError:
                    continue
                
                # Attempt to read and write content
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    out.write(f"## File: `{rel_path}`\n")
                    # Determine programming language for syntax highlighting
                    lang = ""
                    if ext == ".py":
                        lang = "python"
                    elif ext == ".json":
                        lang = "json"
                    elif ext == ".lua":
                        lang = "lua"
                    elif ext == ".sh" or ext == ".ps1":
                        lang = "bash"
                    elif ext == ".md":
                        lang = "markdown"
                        
                    out.write(f"```{lang}\n")
                    out.write(content)
                    if not content.endswith('\n'):
                        out.write('\n')
                    out.write("```\n\n")
                    print(f"Bundled: {rel_path}")
                except Exception as e:
                    print(f"Error reading {rel_path}: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(current_dir, "maya_codebase_summary.md")
    print(f"Bundling codebase from {current_dir} into {output_path}...")
    walk_and_bundle(current_dir, output_path)
    print("Bundling complete! You can now upload 'maya_codebase_summary.md' directly to your AI analyzer.")
