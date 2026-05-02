"""
Context Builder for Phase 3.4.

This module takes retrieved chunks from the vector store and assembles them into a
prompt-ready context block with intelligent handling of context window limits,
deduplication, and metadata integration.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import logging
import re
from collections import defaultdict
import numpy as np
from essentials.phase3_1.models import Chunk
from utils.token_utils import n_tokens
from pathlib import Path
import yaml
from essentials.utils.token_budget import count_tokens, trim_to_tokens
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BUDGET = yaml.safe_load(Path(PROJECT_ROOT / "essentials/config/prompt_budget.yaml").read_text())
MAX_CTX         = _BUDGET["max_ctx"]
ANS_RESERVE     = _BUDGET["answer_reserve"]
SLICE_TOKENS    = _BUDGET["slices"]

class ContextBuilder:
    """Builds context from retrieved chunks for use in prompts."""
    
    def __init__(
        self,
        max_tokens: int = 4096,
        token_counter: Optional[Callable[[str], int]] = None,
        include_metadata: bool = True,
        deduplicate: bool = True,
        diversify: bool = True,
        coherence_check: bool = True,
        preserve_order: bool = False,
        debug: bool = False  # New debug flag
    ):
        """Initialize the context builder.
        
        Args:
            max_tokens: Maximum number of tokens allowed in the context
            token_counter: Function to count tokens in text (if None, uses heuristic)
            include_metadata: Whether to include chunk metadata in the context
            deduplicate: Whether to remove similar chunks
            diversify: Whether to promote information diversity
            coherence_check: Whether to check for topic coherence
            preserve_order: Whether to preserve original order (overrides sorting by relevance)
            debug: Whether to log detailed skip reasons
        """
        self.max_tokens = max_tokens
        self.token_counter = token_counter or self._estimate_tokens
        self.include_metadata = include_metadata
        self.deduplicate = deduplicate
        self.diversify = diversify
        self.coherence_check = coherence_check
        self.preserve_order = preserve_order
        self.debug = debug
        
        # Track which chunks are used in the final context
        self.used_chunks = []
        self.used_citation_ids = set()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 characters per token for English text
        return len(text) // 4 + 1
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as a string.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Formatted metadata string
        """
        if not metadata:
            return ""
        
        # Format based on available metadata fields
        formatted = []
        
        # Source information
        if "source" in metadata:
            formatted.append(f"Source: {metadata['source']}")
            
        # Section information
        if "section" in metadata:
            formatted.append(f"Section: {metadata['section']}")
        elif "section_title" in metadata:
            formatted.append(f"Section: {metadata['section_title']}")
            
        # Page information
        if "page" in metadata:
            formatted.append(f"Page: {metadata['page']}")
            
        # Figure/table information
        if "content_type" in metadata:
            if str(metadata["content_type"]).lower() in ["figure", "table", "chart", "graph"]:
                content_type = metadata["content_type"]
                content_id = metadata.get("content_id", "")
                formatted.append(f"{content_type}: {content_id}")
        
        # Citation information
        if "citation" in metadata:
            formatted.append(f"Citation: {metadata['citation']}")
        
        # Return formatted string or empty string if no metadata
        if formatted:
            return "[" + " | ".join(formatted) + "]"
        return ""
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks using Jaccard similarity.
        
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
    
    def _is_duplicate(self, new_text: str, threshold: float = 0.7) -> bool:
        """Check if a text is too similar to already included chunks.
        
        Args:
            new_text: Text to check
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if duplicate, False otherwise
        """
        for chunk in self.used_chunks:
            chunk_text = chunk.get("text", "")
            if self._calculate_similarity(new_text, chunk_text) > threshold:
                return True
        return False
    
    def _check_coherence(self, current_text: str, new_text: str) -> bool:
        """Check if adding a new chunk maintains topic coherence.
        
        Args:
            current_text: Current context text
            new_text: New text to add
            
        Returns:
            True if coherent, False otherwise
        """
        # This is a simplified coherence check
        # In a production system, you might use more sophisticated methods
        
        # Extract key terms from both texts
        current_terms = set(re.findall(r'\b\w{4,}\b', current_text.lower()))
        new_terms = set(re.findall(r'\b\w{4,}\b', new_text.lower()))
        
        # Check if there's sufficient term overlap
        if len(current_terms) == 0:
            return True  # No current context yet
            
        overlap = len(current_terms.intersection(new_terms))
        min_overlap = min(2, len(current_terms) // 10)  # At least 10% overlap or 2 terms
        
        return overlap >= min_overlap
    
    def _ensure_diversity(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder chunks to ensure diversity of information.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Reordered list of chunks
        """
        if not chunks:
            return []
            
        # Group chunks by section/source
        sections = defaultdict(list)
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            section = metadata.get("section", metadata.get("section_title", "unknown"))
            sections[section].append(chunk)
            
        # If only one section, return original order
        if len(sections) <= 1:
            return chunks
            
        # Reorder to alternate between sections
        result = []
        remaining = list(sections.values())
        
        while any(remaining):
            for i in range(len(remaining)):
                if remaining[i]:
                    result.append(remaining[i].pop(0))
                    
            # Remove empty sections
            remaining = [s for s in remaining if s]
            
        return result
    
    def _format_chunk(self, chunk: Dict[str, Any], include_citations: bool = True) -> Tuple[str, int]:
        """Format a chunk for inclusion in the context.
        
        Args:
            chunk: Chunk to format
            include_citations: Whether to include citations
            
        Returns:
            Tuple of (formatted text, token count)
        """
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        formatted_parts = []
        
        # Add metadata header if requested
        if self.include_metadata:
            metadata_str = self._format_metadata(metadata)
            if metadata_str:
                formatted_parts.append(metadata_str)
        
        # Add the main text
        formatted_parts.append(text)
        
        # Add citation if available and requested
        if include_citations and "id" in chunk:
            chunk_id = chunk["id"]
            if chunk_id not in self.used_citation_ids:
                self.used_citation_ids.add(chunk_id)
                citation = f"[{len(self.used_citation_ids)}]"
                formatted_parts.append(citation)
        
        # Join all parts
        formatted_text = "\n".join(formatted_parts)
        
        # Calculate token count
        token_count = self.token_counter(formatted_text)
        
        return formatted_text, token_count
    
    def _infer_query_intent(self, query: str) -> str:
        """Infer the general intent of the query."""
        query = query.lower()
        if any(keyword in query for keyword in ["why", "how", "benefit"]):
            return "motivation"
        elif any(keyword in query for keyword in ["what is", "define"]):
            return "definition"
        elif any(keyword in query for keyword in ["how does", "describe the method"]):
            return "methodology"
        elif any(keyword in query for keyword in ["findings", "accuracy", "performance"]):
            return "results"
        elif any(keyword in query for keyword in ["prior work", "related work"]):
            return "background"
        elif any(keyword in query for keyword in ["challenges", "drawbacks"]):
            return "limitations"
        return "general"

    def _score_chunk_relevance_by_metadata(self, chunk: Dict[str, Any], query_terms: List[str]) -> float:
        """Boost chunk scores based on metadata relevance to the query terms."""
        metadata_fields = [
            str(chunk.get("metadata", {}).get("section", "")).lower(),
            str(chunk.get("metadata", {}).get("section_title", "")).lower(),
            str(chunk.get("metadata", {}).get("page", "")).lower(),
            str(chunk.get("metadata", {}).get("content_type", "")).lower()
        ]
        for term in query_terms:
            if any(term in field for field in metadata_fields):
                return 1.2  # Boost score by 20%
        return 0.0

    def _keyword_match_bonus(self, chunk, query_terms):
        # Bonus if any query term is in metadata fields
        metadata = chunk.get("metadata", {})
        fields = [
            str(metadata.get("section", "")).lower(),
            str(metadata.get("section_title", "")).lower(),
            str(metadata.get("page", "")).lower(),
            str(metadata.get("content_type", "")).lower(),
        ]
        for term in query_terms:
            if any(term in field for field in fields):
                return 0.5
        return 0.0

    def _is_figure_query(self, query: str) -> Optional[str]:
        """Detect if the query is about a specific figure. Returns the figure number as a string if found."""
        if not query:
            return None
        match = re.search(r"figure\s*(\d+)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _find_figure_chunk(self, chunks: List[Dict[str, Any]], figure_number: str) -> Optional[Dict[str, Any]]:
        """Find the chunk corresponding to the given figure number."""
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            # Check for figure number in metadata or text
            if str(metadata.get("content_type", "")).lower() == "figure":
                # Try to match figure number in content_id or text
                if (
                    str(metadata.get("content_id", "")).strip() == figure_number
                    or re.search(rf"figure\s*{figure_number}\b", chunk.get("text", ""), re.IGNORECASE)
                ):
                    return chunk
        return None

    def _find_nearby_context(self, chunks: List[Dict[str, Any]], figure_chunk: Dict[str, Any], window: int = 2) -> List[Dict[str, Any]]:
        """Find section/context chunks near the figure chunk (e.g., previous and next N chunks)."""
        if not figure_chunk:
            return []
        idx = None
        for i, chunk in enumerate(chunks):
            if chunk is figure_chunk:
                idx = i
                break
        if idx is None:
            return []
        # Get previous and next 'window' chunks that are not figures
        context_chunks = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            j = idx + offset
            if 0 <= j < len(chunks):
                meta = chunks[j].get("metadata", {})
                if str(meta.get("content_type", "")).lower() != "figure":
                    context_chunks.append(chunks[j])
        return context_chunks

    def _budgeted_concat(self, parts: dict) -> str:
        """Trim each slice to its token allotment, then concatenate in order."""
        ordered_keys = ["sys", "q", "fig", "ctx", "ref"]
        buf = []
        running_tokens = 0
        for k in ordered_keys:
            target = SLICE_TOKENS[k]
            txt    = parts.get(k, "")
            txt    = trim_to_tokens(txt, target)
            running_tokens += count_tokens(txt)
            buf.append(txt.strip())
        # final sanity check
        assert running_tokens + ANS_RESERVE <= MAX_CTX, (
            f"Prompt {running_tokens}t + reserve exceeds window {MAX_CTX}"
        )
        return "\n\n".join(buf).strip()

    def build_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build context from retrieved chunks with figure-aware logic if query is about a figure."""
        if not retrieved_chunks:
            logger.warning("No chunks provided for context building")
            return {
                "context": "",
                "chunks_used": 0,
                "tokens_used": 0,
                "citations": {},
                "warning": "No chunks provided"
            }

        # Figure-aware logic
        figure_number = self._is_figure_query(query)
        if figure_number:
            figure_chunk = self._find_figure_chunk(retrieved_chunks, figure_number)
            if figure_chunk:
                # Get nearby context
                context_chunks = self._find_nearby_context(retrieved_chunks, figure_chunk, window=2)
                # Format: figure chunk first, then context
                ordered_chunks = [figure_chunk] + context_chunks
                # Remove duplicates
                seen = set()
                unique_chunks = []
                for c in ordered_chunks:
                    cid = id(c)
                    if cid not in seen:
                        unique_chunks.append(c)
                        seen.add(cid)
                # Build context as usual
                context_parts = []
                total_tokens = 0
                self.used_chunks = []
                self.used_citation_ids = set()
                citations = {}
                current_context = ""
                for chunk in unique_chunks:
                    chunk_text = chunk.get("text", "")
                    if self.deduplicate and self._is_duplicate(chunk_text):
                        continue
                    formatted_chunk, chunk_tokens = self._format_chunk(chunk)
                    if total_tokens + chunk_tokens > self.max_tokens:
                        break
                    context_parts.append(formatted_chunk)
                    total_tokens += chunk_tokens
                    current_context += " " + chunk_text
                    self.used_chunks.append(chunk)
                    if "id" in chunk:
                        citations[str(len(self.used_citation_ids))] = chunk.get("id")
                context = "\n\n".join(context_parts)
                return {
                    "context": context,
                    "chunks_used": len(self.used_chunks),
                    "tokens_used": total_tokens,
                    "citations": citations,
                    "figure_number": figure_number,
                    "figure_found": True
                }
            # If figure not found, fall back to default
        # Default logic (unchanged)
        # Log the number of retrieved chunks
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for context building")

        # Reset tracking
        self.used_chunks = []
        self.used_citation_ids = set()
        citations = {}

        # Tokenize query into terms
        query_terms = re.findall(r'\w+', query.lower()) if query else []

        # Compute adjusted scores and keyword bonuses
        for chunk in retrieved_chunks:
            rel_score = chunk.get("score", 0)
            meta_bonus = self._score_chunk_relevance_by_metadata(chunk, query_terms)
            kw_bonus = self._keyword_match_bonus(chunk, query_terms)
            chunk["keyword_bonus"] = kw_bonus
            chunk["adjusted_score"] = rel_score + meta_bonus + kw_bonus

        # Log adjusted scores
        logger.info("Adjusted chunk scores based on metadata and keyword relevance")

        # Sort: keyword matches first, then by adjusted score
        if not self.preserve_order:
            retrieved_chunks.sort(key=lambda x: (x["keyword_bonus"], x["adjusted_score"]), reverse=True)

        # Log sorted chunks
        logger.info(f"Sorted chunks: {retrieved_chunks}")

        # Optionally ensure diversity in chunks
        if self.diversify:
            retrieved_chunks = self._ensure_diversity(retrieved_chunks)

        context_parts = []
        total_tokens = 0
        current_context = ""

        # Process each chunk
        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "")

            # Log chunk text
            logger.info(f"Processing chunk: {chunk_text}")

            # Skip if deduplication is enabled and this is a duplicate
            if self.deduplicate and self._is_duplicate(chunk_text):
                if self.debug:
                    logger.info(f"Skipping chunk (duplicate): {chunk_text[:80]}...")
                continue

            # Check coherence if enabled
            if self.coherence_check and current_context and not self._check_coherence(current_context, chunk_text):
                if self.debug:
                    logger.info(f"Skipping chunk (incoherent): {chunk_text[:80]}...")
                continue

            # Format the chunk
            formatted_chunk, chunk_tokens = self._format_chunk(chunk)

            # Check if adding this chunk would exceed the token limit
            if total_tokens + chunk_tokens > self.max_tokens:
                logger.info("Token limit reached, stopping context assembly")
                # If we haven't added any chunks yet, add this one truncated
                if not context_parts:
                    # Truncate to fit
                    truncated_chunk = formatted_chunk[:int(self.max_tokens * 4)]  # Using our heuristic
                    context_parts.append(truncated_chunk)
                    total_tokens = self.token_counter(truncated_chunk)
                break

            # Add the chunk
            context_parts.append(formatted_chunk)
            total_tokens += chunk_tokens
            current_context += " " + chunk_text

            # Track the chunk and citation
            self.used_chunks.append(chunk)
            if "id" in chunk:
                citations[str(len(self.used_citation_ids))] = chunk.get("id")

        # Join all parts with double newlines
        context = "\n\n".join(context_parts)

        # Log a preview of the built context
        logger.info(f"\n🚨 Built Context Preview:\n{context[:1000]}")

        # Log the final context length and preview
        logger.info(f"\n🧠 Final context length: {len(context)} characters")
        logger.info(f"\n🧠 Final context preview:\n{context[:1000]}")

        return {
            "context": context,
            "chunks_used": len(self.used_chunks),
            "tokens_used": total_tokens,
            "citations": citations
        }
    
    def build_from_chunks(
        self, 
        chunks: List[Chunk], 
        scores: Optional[List[float]] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build context directly from Chunk objects.
        
        Args:
            chunks: List of Chunk objects
            scores: Optional list of relevance scores (parallel to chunks)
            query: Optional query for coherence checking
            
        Returns:
            Dictionary with built context and metadata
        """
        # Convert Chunk objects to dictionaries
        retrieved_chunks = []
        for i, chunk in enumerate(chunks):
            score = scores[i] if scores and i < len(scores) else 1.0
            retrieved_chunks.append({
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata or {},
                "score": score
            })
        
        return self.build_context(retrieved_chunks, query)

    @staticmethod
    def merge_figure_and_section_chunks(figure_chunks: List[Dict[str, Any]], section_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge figure chunks (from convert_figures_to_chunks) and section/content chunks into a single list.
        This should be used before passing to build_context.
        Args:
            figure_chunks: List of figure chunk dicts (with content_type: 'figure')
            section_chunks: List of section/content chunk dicts
        Returns:
            List of all chunks, ready for context building
        """
        # Optionally, you could sort or interleave, but default is just concatenate
        return figure_chunks + section_chunks

    def build_context_v2(self, chunks, sys_msg: str, user_q: str, llm) -> str:
        """Budget-aware context builder with slice-aware trimming."""
        # Partition chunks into slices
        figure_section = []
        normal_section = []
        refs_section = []
        for ch in chunks:
            meta = ch.get("metadata", {})
            ctype = str(meta.get("content_type", "")).lower()
            if ctype == "figure":
                figure_section.append(ch.get("page_content", ch.get("text", "")))
            elif ctype == "reference":
                refs_section.append(ch.get("page_content", ch.get("text", "")))
            else:
                normal_section.append(ch.get("page_content", ch.get("text", "")))
        # Join sections
        figure_section = "\n".join(figure_section)
        normal_section = "\n".join(normal_section)
        refs_section = "\n".join(refs_section)
        # Compose prompt
        prompt = self._budgeted_concat({
            "sys": sys_msg,
            "q": user_q,
            "fig": figure_section,
            "ctx": normal_section,
            "ref": refs_section,
        })
        return prompt

def trim_text_to_token_limit(text: str, max_tokens: int, token_counter: Optional[Callable[[str], int]] = None) -> str:
    """Utility function to trim text to a token limit.
    
    Args:
        text: Text to trim
        max_tokens: Maximum number of tokens
        token_counter: Function to count tokens (if None, uses heuristic)
        
    Returns:
        Trimmed text
    """
    if not text:
        return ""
        
    counter = token_counter or (lambda t: len(t) // 4 + 1)
    
    # If already under limit, return as is
    if counter(text) <= max_tokens:
        return text
    
    # Trim by paragraphs first
    paragraphs = text.split("\n\n")
    result = []
    tokens_used = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = counter(paragraph)
        if tokens_used + paragraph_tokens <= max_tokens:
            result.append(paragraph)
            tokens_used += paragraph_tokens
        else:
            # For the last paragraph, trim by sentences
            sentences = paragraph.split(". ")
            for sentence in sentences:
                sentence_tokens = counter(sentence + ". ")
                if tokens_used + sentence_tokens <= max_tokens:
                    result.append(sentence + ".")
                    tokens_used += sentence_tokens
                else:
                    break
            break
    
    return "\n\n".join(result) 