#!/usr/bin/env python3
"""
Fix for Path import conflict in AVH_api_enhanced.py
Author: FETHl
Date: 2025-04-08 14:04:09
"""

import os
import re

def fix_path_imports(file_path):
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the import statement to use an alias for pathlib.Path
    content = content.replace(
        'from pathlib import Path',
        'from pathlib import Path as PathLib'
    )
    
    # Replace all instances of Path(...) when used for file paths
    # We'll look for patterns like Path(BASE_IMAGE_PATH) or similar
    patterns = [
        r'Path\((BASE_IMAGE_PATH|"[^"]+"|\'[^\']+\'|[^)]+\.path)\)',
        r'Path\((BASE_IMAGE_PATH|"[^"]+"|\'[^\']+\'|[^)]+\.path)\) / ',
        r'Path\(f"[^"]+")',
        r'Path\(f\'[^\']+\')'
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, r'PathLib(\1)', content)
    
    # Also find constructs like x = Path() and similar
    content = re.sub(r'= Path\(\)', r'= PathLib()', content)
    
    # Only change those Path instances that refer to file system paths, not FastAPI Path params
    
    # Save the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Update version info
    update_version_info(file_path)
    
    print(f"Fixed Path import conflicts in {file_path}")

def update_version_info(file_path):
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update version function with current timestamp
    version_pattern = r'@app.get\("/version"\)\s*def get_version\(\):\s*return \{[^}]+\}'
    new_version = """@app.get("/version")
def get_version():
    return {
        "version": "2.0.0", 
        "api_version": "2.0.0",
        "sam_version": "1.0", 
        "user": "FETHl",
        "date": "2025-04-08 14:04:09"
    }"""
    
    content = re.sub(version_pattern, new_version, content)
    
    # Save the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    file_path = "AVH_api_enhanced.py"
    if os.path.exists(file_path):
        fix_path_imports(file_path)
    else:
        print(f"Error: {file_path} not found in the current directory.")