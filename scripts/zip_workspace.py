import os
import zipfile

def zip_project():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    zip_path = os.path.join(base_dir, "maya_workspace.zip")
    
    # Files/directories to exclude
    exclude_dirs = {".venv", ".git", "__pycache__", "checkpoints", "prof_logs", "target", ".target", "node_modules"}
    exclude_files = {"ham_veri.txt", "mock_data.bin", "maya_workspace.zip", "temp_tr_news.csv", "temp_en_wiki.txt"}
    
    print(f"Creating zip archive at: {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                file_path = os.path.join(root, file)
                # Compute relative path for zip entry
                rel_path = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, rel_path)
                
    print(f"[SUCCESS] Zip file created successfully! Size: {os.path.getsize(zip_path) / 1024:.2f} KB")

if __name__ == "__main__":
    zip_project()
