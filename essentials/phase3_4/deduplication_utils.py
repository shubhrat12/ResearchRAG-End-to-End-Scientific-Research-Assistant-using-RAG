"""
Deduplication Utilities for Phase 3.4.

This module provides utilities for detecting and handling duplicate or
highly similar content in retrieved chunks to improve context quality.
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Callable
import re
import logging
import numpy as np
from essentials.phase3_1.models import Chunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two text chunks.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    # Tokenize by splitting on whitespace and punctuation
    def tokenize(text):
        # Convert to lowercase and split by non-alphanumeric characters
        return set(re.findall(r'\w+', text.lower()))
        
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # Calculate Jaccard similarity: intersection / union
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def contains_substring(text1: str, text2: str, min_length: int = 50) -> bool:
    """Check if one text contains a substantial substring of another.
    
    Args:
        text1: First text
        text2: Second text
        min_length: Minimum substring length to consider
        
    Returns:
        True if substantial overlap exists
    """
    # If one text is much shorter than the other, check if it's contained
    if len(text1) < len(text2) - min_length:
        return text1.lower() in text2.lower()
    elif len(text2) < len(text1) - min_length:
        return text2.lower() in text1.lower()
    
    return False

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score (-1 to 1)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
        
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
    embedding_key: str = "embedding",
    use_embeddings: bool = True,
    use_text: bool = True
) -> List[Dict[str, Any]]:
    """Deduplicate chunks based on similarity.
    
    Args:
        chunks: List of chunks to deduplicate
        similarity_threshold: Threshold above which chunks are considered duplicates
        embedding_key: Key for embeddings in chunk dictionaries
        use_embeddings: Whether to use embeddings for similarity comparison
        use_text: Whether to use text for similarity comparison
        
    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return []
        
    # Keep track of which chunks to include
    include_indices = set()
    duplicate_pairs = []
    
    # First pass: identify duplicates
    for i in range(len(chunks)):
        # If we haven't seen this chunk yet, add it
        if i not in include_indices:
            include_indices.add(i)
            
            # Check for duplicates with all other chunks
            for j in range(i + 1, len(chunks)):
                is_duplicate = False
                
                # Check text similarity if requested
                if use_text:
                    text_i = chunks[i].get("text", "")
                    text_j = chunks[j].get("text", "")
                    
                    # Check for substring containment
                    if contains_substring(text_i, text_j):
                        is_duplicate = True
                    else:
                        # Check Jaccard similarity
                        text_similarity = jaccard_similarity(text_i, text_j)
                        if text_similarity > similarity_threshold:
                            is_duplicate = True
                
                # Check embedding similarity if requested and available
                if use_embeddings and embedding_key in chunks[i] and embedding_key in chunks[j]:
                    emb_i = chunks[i][embedding_key]
                    emb_j = chunks[j][embedding_key]
                    
                    emb_similarity = cosine_similarity(emb_i, emb_j)
                    if emb_similarity > similarity_threshold:
                        is_duplicate = True
                
                # If duplicate, record the pair but don't add j to include_indices
                if is_duplicate:
                    # Always keep the chunk with the higher score
                    score_i = chunks[i].get("score", 0)
                    score_j = chunks[j].get("score", 0)
                    
                    if score_j > score_i:
                        # Replace i with j in the include set
                        include_indices.remove(i)
                        include_indices.add(j)
                        duplicate_pairs.append((i, j))
                        break
                    else:
                        duplicate_pairs.append((j, i))
    
    # Create the deduplicated list
    deduplicated = [chunks[i] for i in sorted(include_indices)]
    
    logger.info(f"Deduplicated {len(chunks)} chunks to {len(deduplicated)} chunks")
    
    return deduplicated

def deduplicate_chunk_objects(
    chunks: List[Chunk], 
    similarity_threshold: float = 0.7
) -> List[Chunk]:
    """Deduplicate a list of Chunk objects based on text similarity.
    
    Args:
        chunks: List of Chunk objects
        similarity_threshold: Threshold above which chunks are considered duplicates
        
    Returns:
        Deduplicated list of Chunk objects
    """
    if not chunks:
        return []
        
    # Convert to dictionaries for processing
    chunk_dicts = [
        {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata or {}}
        for chunk in chunks
    ]
    
    # Deduplicate
    deduplicated_dicts = deduplicate_chunks(
        chunk_dicts,
        similarity_threshold=similarity_threshold,
        use_embeddings=False,  # No embeddings in Chunk objects
        use_text=True
    )
    
    # Convert back to Chunk objects
    deduplicated_ids = [d["id"] for d in deduplicated_dicts]
    return [chunk for chunk in chunks if chunk.id in deduplicated_ids]

def diversify_chunks(chunks: List[Dict[str, Any]], max_per_source: int = 2) -> List[Dict[str, Any]]:
    """Diversify chunks by limiting the number from each source.
    
    Args:
        chunks: List of chunks to diversify
        max_per_source: Maximum number of chunks to include from each source
        
    Returns:
        Diversified list of chunks
    """
    if not chunks:
        return []
        
    # Group chunks by source
    sources = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        if source not in sources:
            sources[source] = []
        sources[source].append(chunk)
    
    # Sort each source group by score
    for source in sources:
        sources[source].sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take at most max_per_source from each source
    result = []
    for source in sources:
        result.extend(sources[source][:max_per_source])
    
    # Sort the final list by score
    result.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return result

def find_duplicates_in_text(text: str, min_length: int = 20, min_occurrences: int = 2) -> List[Tuple[str, int]]:
    """Find repeated phrases in text.
    
    Args:
        text: Text to analyze
        min_length: Minimum phrase length to consider
        min_occurrences: Minimum number of occurrences to report
        
    Returns:
        List of (phrase, count) tuples
    """
    # This is a simplified approach using n-grams
    words = text.lower().split()
    if len(words) < min_length:
        return []
    
    # Generate n-grams and count occurrences
    n_grams = {}
    max_n = min(len(words) // 2, 10)  # Don't go beyond 10-grams or half the text
    
    for n in range(min_length // 4, max_n + 1):  # Approximate min_length in words
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if len(phrase) >= min_length:
                n_grams[phrase] = n_grams.get(phrase, 0) + 1
    
    # Filter by minimum occurrences
    duplicates = [(phrase, count) for phrase, count in n_grams.items() if count >= min_occurrences]
    
    # Sort by count (descending)
    duplicates.sort(key=lambda x: x[1], reverse=True)
    
    return duplicates

def remove_duplicated_sentences(text: str) -> str:
    """Remove duplicated sentences from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with duplicated sentences removed
    """
    if not text:
        return ""
        
    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Track unique sentences
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        # Normalize for comparison
        normalized = sentence.lower().strip()
        if normalized and normalized not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(normalized)
    
    # Rejoin text
    return " ".join(unique_sentences) 