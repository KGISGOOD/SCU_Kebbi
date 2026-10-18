import os
import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Replace backslash-doublequote with doublequote
    # But we must be careful not to replace in places like "\n" (which is backslash-n)
    # Actually, we only want to fix the docstring quotes that were escaped.
    # The pattern we see is: at the start of the file, we have \\"\\"\\" (three times)
    # and similarly at the end of the docstring.
    # We can replace all occurrences of \\\\" with "? But note that in the file, the backslash is a single backslash.
    # In the string we read, a backslash is represented as '\' (two characters: backslash and backslash?).
    # Actually, when we read the file, the string contains a backslash character (ASCII 92) followed by a double quote.
    # So we want to replace the two-character sequence: backslash then double quote, with just double quote.
    # We can do: content = content.replace('\\\\"', '\\"')? Wait, let's think.
    # In the Python string, to represent a backslash we write '\\', and to represent a double quote we write '\\"'.
    # So the sequence backslash-doublequote in the file is represented in the string as '\\\\\\"'? 
    # Actually, no: when we read the file, if the file contains the two bytes: 0x5C (backslash) and 0x22 (double quote),
    # then the Python string will contain the characters: '\' and '"'. So we can write: '\' + '"' -> '\\"'.
    # So we want to replace the substring '\\"' with '"'.
    new_content = content.replace('\\"', '"')
    # However, this will also replace escaped quotes inside strings like in docstrings of functions? 
    # For example, in the function docstring we have: """Load and parse all supported files...
    # After replacement, it will become: """Load and parse all supported files... (which is correct)
    # But also, in the code we have strings like: "Compute MD5 hash of file content." 
    # In the file, that string is written as: "Compute MD5 hash of file content." (with a backslash before the quote)
    # After replacement, it will become: "Compute MD5 hash of file content." (which is also correct because we want the string to have quotes)
    # Wait, no: we want the string literals in the code to have double quotes. So we want the file to contain: 
    #   docstring = """..."""
    #   and normal strings like: s = "hello"
    # In the file we currently have: 
    #   docstring = \\"\\"\\"...\\"\\"\\"   (backslash before each quote)
    #   normal strings: \\"Compute MD5 hash of file content.\\"
    # After replacing \\" with ", we get:
    #   docstring: """..."""   (correct)
    #   normal strings: "Compute MD5 hash of file content."   (correct)
    # So the replacement is safe.
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'Fixed: {filepath}')

# Walk through the directories
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
