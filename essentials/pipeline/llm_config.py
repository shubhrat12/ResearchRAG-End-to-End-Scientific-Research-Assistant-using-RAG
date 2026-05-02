# Configuration for Local LLM Integration

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Path to the quantized local model file
MODEL_PATH = "D:/Langchain Project/models/llm_gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Quantization type (e.g., q4_0)
QUANTIZATION_TYPE = 'q4_0'

# Context window size
CONTEXT_WINDOW = 4096

# Temperature for generation
TEMPERATURE = 0.7 