from typing import List
from essentials.phase3_2.retrieval import Retriever
# Import or define mmr_rerank as needed
# from essentials.phase3_2.retrieval import mmr_rerank  # Uncomment if available
from utils.token_utils import n_tokens
import logging

# You may need to pass in retriever and mmr_rerank from the main pipeline
# For now, assume retriever is instantiated elsewhere and passed in

logger = logging.getLogger(__name__)

def retrieve_chunks(query: str, k: int = 15, retriever=None, mmr_rerank=None, debug: bool = False, debug_dump_path: str = None) -> List:
    """
    High-recall retrieval → re-rank (MMR) → de-dup.
    Returns chunks sorted by importance.
    """
    logger.info(f"[DEBUG] Running retrieval for: {query}")
    if retriever is None:
        raise ValueError("Retriever instance must be provided.")
    if mmr_rerank is None:
        retrieved = retriever.retrieve(query, k=k)
    else:
        raw = retriever.retrieve(query, k=k)
        retrieved = mmr_rerank(query, raw)
    retrieved = [ch for ch in retrieved if isinstance(ch, dict) and len(ch.get('text', '').strip()) > 20]
    logger.info(f"[DEBUG] Retrieved {len(retrieved)} chunks")
    for i, r in enumerate(retrieved):
        text = r.get('text', r.get('page_content', ''))
        logger.info(f"Chunk {i} (tokens: {n_tokens(text)}): {text[:100]}...")
        if not text:
            logger.warning(f"Chunk {i} missing or empty text.")
    if not retrieved:
        logger.warning(f"No valid chunks retrieved for the question: '{query}'")
        logger.warning(f"Returning fallback chunk. Query token count: {n_tokens(query)}")
        return [{"id": "fallback", "text": "The document contains no relevant results for this query.", "score": 0.0, "metadata": {}}]
    if debug and debug_dump_path:
        try:
            with open(debug_dump_path, "w", encoding="utf-8") as f:
                for i, r in enumerate(retrieved[:20]):
                    text = r.get('text', r.get('page_content', ''))
                    f.write(f"Chunk {i}: {text[:200]}\n\n")
            logger.info(f"Dumped top-{min(20, len(retrieved))} retrieved texts to {debug_dump_path}")
        except Exception as e:
            logger.error(f"Failed to dump debug texts: {e}")
    seen, deduped = set(), []
    for ch in retrieved:
        if isinstance(ch, str):
            ch = {"text": ch}
        t = ch.get('page_content', ch.get('text', '')) if isinstance(ch, dict) else str(ch)
        t = t.strip()
        if t not in seen:
            deduped.append(ch)
            seen.add(t)
    return deduped 