import re
import json
import ast
from typing import Dict, Any

def parse_lua_file(file_path: str) -> Dict[str, Any]:
    """
    Parses a Lua configuration file containing nested tables and returns a Python dictionary.
    Safe and robust with zero binary dependencies.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Remove single-line comments starting with '--'
    content = re.sub(r'--.*$', '', content, flags=re.MULTILINE)

    # 2. Remove assignment wrappers like 'local config =' and 'return config'
    content = re.sub(r'local\s+\w+\s*=\s*', '', content)
    content = re.sub(r'return\s+\w+', '', content)
    content = content.strip()

    # 3. Convert Lua key-value assignment 'key = value' into JSON format '"key": value'
    # Captures alphanumeric keys preceding an '=' sign
    content = re.sub(r'(\b\w+\b)\s*=\s*', r'"\1": ', content)

    # 4. Replace semicolons (valid in Lua tables) with commas
    content = content.replace(';', ',')

    # 5. Clean up trailing commas before closing braces: ', }' -> '}'
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*\]', ']', content)

    # 6. Parse JSON/Dict literal
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # Fallback to ast.literal_eval which is more forgiving of single quotes and raw numbers
            return ast.literal_eval(content)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse Lua table to Python dict.\n"
                f"Error: {e}\n"
                f"Processed buffer:\n{content}"
            )

if __name__ == "__main__":
    import sys
    import os
    
    # Test script execution
    test_path = os.path.join(os.path.dirname(__file__), "config.lua")
    if os.path.exists(test_path):
        print(f"Testing Lua config parser on: {test_path}")
        cfg = parse_lua_file(test_path)
        print("Successfully parsed Lua configuration:")
        print(json.dumps(cfg, indent=2))
    else:
        print("config.lua not found in current directory.")
