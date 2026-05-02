from functools import lru_cache
from typing import List
import logging
from pathlib import Path

logger = logging.getLogger("prompt.budget")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_tokenizer = None
_tokenizer_name = None

# --- Tokenizer setup ---
def set_tokenizer(tokenizer, name=None):
    global _tokenizer, _tokenizer_name
    _tokenizer = tokenizer
    _tokenizer_name = name or str(tokenizer)
    logger.info(f"[token_budget] Tokenizer set to: {_tokenizer_name}")

# --- Token encoding ---
def _encode(text: str) -> List[int]:
    if _tokenizer is not None:
        return _tokenizer.encode(text, add_special_tokens=False)
    try:
        from tiktoken import get_encoding
        _enc = get_encoding("cl100k_base")
        return _enc.encode(text, disallowed_special=())
    except ModuleNotFoundError:
        from llama_cpp import Llama
        _llama = Llama(model_path="__DUMMY__", n_ctx=4)
        return _llama.tokenize(f" {text}".encode())

@lru_cache(maxsize=None)
def count_tokens(text: str) -> int:
    """Fast token count with memoisation."""
    return len(_encode(text))

def trim_to_tokens(text: str, limit: int) -> str:
    """Return `text` trimmed (word-safe) to `limit` tokens."""
    tokens = _encode(text)
    if len(tokens) <= limit:
        return text
    # Re-encode word by word until just below limit.
    words = text.split()
    out, running = [], []
    for w in words:
        running.append(w)
        if len(_encode(" ".join(running))) > limit:
            break
        out.append(w)
    return " ".join(out)

# --- Test function for alignment ---
def test_tokenizer_alignment(llm_tokenizer):
    test_str = "This is a test string for tokenizer alignment."
    our_count = count_tokens(test_str)
    llm_count = len(llm_tokenizer.encode(test_str, add_special_tokens=False))
    print(f"[token_budget] Our count: {our_count}, LLM count: {llm_count}")
    assert our_count == llm_count, f"Tokenizer mismatch: {our_count} vs {llm_count}" 