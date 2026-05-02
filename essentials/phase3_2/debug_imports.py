import sys
import os

print("Debug script running...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    from essentials.phase3_1.models import Chunk
    print("Successfully imported Chunk")
except Exception as e:
    print(f"Error importing Chunk: {str(e)}")

try:
    from essentials.phase3_2.vector_store import VectorStore
    print("Successfully imported VectorStore")
except Exception as e:
    print(f"Error importing VectorStore: {str(e)}")

print("Debug script completed") 