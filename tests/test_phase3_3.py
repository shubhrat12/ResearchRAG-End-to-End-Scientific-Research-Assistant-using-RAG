"""
Simple test script for phase3.3 implementation.
"""

import sys
import os

# Add the current directory to the path to ensure we can import the modules
sys.path.append(os.path.abspath('.'))

try:
    # Try importing with underscore notation
    print("Trying to import with underscore notation...")
    from essentials.phase3_3.vector_store import ChromaVectorStore
    print("Import with underscore notation succeeded!")
except Exception as e:
    print(f"Import with underscore notation failed: {e}")

# Print the current module search path
print("\nPython module search path:")
for path in sys.path:
    print(f"- {path}")

# List the essentials directory
print("\nContents of essentials directory:")
try:
    print(os.listdir("essentials"))
except Exception as e:
    print(f"Error listing directory: {e}") 