from functools import lru_cache
import tiktoken
import logging
logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def _enc(model_name: str = "mistral-7b"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"tiktoken does not recognize model '{model_name}', falling back to 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

def n_tokens(txt: str, model_name="mistral-7b") -> int:
    """Fast deterministic token count (used everywhere)."""
    try:
        return len(_enc(model_name).encode(txt))
    except Exception as e:
        logger.warning(f"Token counting failed for model '{model_name}': {e}. Returning 0.")
        return 0 