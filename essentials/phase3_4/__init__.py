"""
Phase 3.4: Context Building for RAG

This module provides components for building context from retrieved chunks
and creating prompt templates for the LangChain-GPT RAG pipeline.
"""

from essentials.phase3_4.context_builder import ContextBuilder, trim_text_to_token_limit
from essentials.phase3_4.deduplication_utils import (
    jaccard_similarity, contains_substring, cosine_similarity,
    deduplicate_chunks, deduplicate_chunk_objects, diversify_chunks,
    find_duplicates_in_text, remove_duplicated_sentences
)
from essentials.phase3_4.prompt_templates import (
    PromptTemplate, PromptTemplateLibrary, QueryType
)

__all__ = [
    'ContextBuilder',
    'trim_text_to_token_limit',
    'jaccard_similarity',
    'contains_substring',
    'cosine_similarity',
    'deduplicate_chunks',
    'deduplicate_chunk_objects',
    'diversify_chunks',
    'find_duplicates_in_text',
    'remove_duplicated_sentences',
    'PromptTemplate',
    'PromptTemplateLibrary',
    'QueryType'
] 