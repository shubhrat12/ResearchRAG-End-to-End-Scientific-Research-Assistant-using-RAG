# chunking.py

from typing import List, Union, Dict, Tuple, Optional
from essentials.phase3_1.models import Chunk, Section
import re
from essentials.utils.token_budget import count_tokens
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Implement the chunking functions
def chunk_fixed(text: str, chunk_size: int, overlap: int = 0) -> List[Chunk]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = ' '.join(words[start:end]).strip()
        if chunk_text:
            chunk = Chunk(id=Chunk.generate_id(), text=chunk_text, source="fixed", metadata={"index": len(chunks), "total": None})
            chunks.append(chunk)
        start += chunk_size - overlap
    # Update total chunks metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk.metadata["total"] = total_chunks
    return chunks

def chunk_by_sentence(text: str) -> List[Chunk]:
    sentences = re.split(r'(?<=[.!?])[\s\n]+', text)
    chunks = [Chunk(id=Chunk.generate_id(), text=sentence.strip(), source="sentence") for sentence in sentences if sentence.strip()]
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata = {"index": i, "total": total_chunks}
    return chunks

def split_text_to_token_chunks(text: str, max_tokens: int = 300) -> List[str]:
    """Split text into chunks of up to max_tokens tokens, splitting at sentence boundaries if possible."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_tokens = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(' '.join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    if current:
        chunks.append(' '.join(current))
    return chunks

def chunk_by_paragraph(text: str) -> List[Chunk]:
    paragraphs = text.split('\n\n')
    all_chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        subchunks = split_text_to_token_chunks(paragraph, max_tokens=300)
        for subchunk in subchunks:
            all_chunks.append(subchunk)
    total_chunks = len(all_chunks)
    chunk_objs = [Chunk(id=Chunk.generate_id(), text=subchunk, source="paragraph", metadata={"index": i, "total": total_chunks}) for i, subchunk in enumerate(all_chunks)]
    return chunk_objs

def chunk_by_sections(sections: List[Section]) -> List[Chunk]:
    chunks = []
    for i, section in enumerate(sections):
        # Check if this is an abstract or conclusion section for special handling
        is_special_section = False
        section_title = section.title.lower() if section.title else ""
        if "abstract" in section_title or "conclusion" in section_title:
            is_special_section = True
            
        # Create the chunk with section information
        chunk = Chunk(
            id=Chunk.generate_id(), 
            text=section.content, 
            source="section", 
            metadata={
                "title": section.title,
                "index": i,
                "is_special_section": is_special_section
            }
        )
        chunks.append(chunk)
    
    # Update total chunks metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk.metadata["total"] = total_chunks
    
    return chunks

def detect_tables(text: str) -> List[Tuple[int, int, str]]:
    """Detect tables in text using patterns.
    Returns a list of (start_pos, end_pos, table_text)"""
    # Simple table detection using common patterns
    tables = []
    
    # Pattern 1: Text between rows of dashes/pipes (more relaxed pattern)
    table_pattern1 = r'(\+[-+]+\+|\|[-|]+\||\+-+\+)[^\n]*\n(.*?\n)+?(\+[-+]+\+|\|[-|]+\||\+-+\+)'
    # Pattern 2: Text with multiple tab/space separated columns (more relaxed)
    table_pattern2 = r'(?:\S+[ \t]+){2,}\S+[ \t]*\n(?:\S+[ \t]+){2,}\S+[ \t]*\n(?:\S+[ \t]+){2,}\S+'
    # Pattern 3: Simple ASCII table pattern
    table_pattern3 = r'[+|][-+|]*[+|][\s\S]*?[+|][-+|]*[+|]'
    
    for pattern in [table_pattern1, table_pattern2, table_pattern3]:
        for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
            tables.append((match.start(), match.end(), match.group(0)))
    
    return tables

def extract_mathematical_formulas(text: str) -> List[Tuple[int, int, str]]:
    """Extract mathematical formulas from text.
    Returns a list of (start_pos, end_pos, formula_text)"""
    formulas = []
    
    # Pattern 1: LaTeX style equations between $$ or $ markers
    pattern1 = r'\$\$(.*?)\$\$|\$(.*?)\$'
    # Pattern 2: Equation environments
    pattern2 = r'\\begin\{equation\}(.*?)\\end\{equation\}'
    
    for pattern in [pattern1, pattern2]:
        for match in re.finditer(pattern, text, re.DOTALL):
            formulas.append((match.start(), match.end(), match.group(0)))
    
    return formulas

def extract_citations(text: str) -> List[Tuple[int, int, str]]:
    """Extract citations from text.
    Returns a list of (start_pos, end_pos, citation_text)"""
    citations = []
    
    # Pattern 1: Harvard style citations [Author, Year]
    pattern1 = r'\[([A-Za-z\s]+,\s*\d{4}(?:;\s*[A-Za-z\s]+,\s*\d{4})*)\]'
    # Pattern 2: Numeric citations [1] or [1,2,3]
    pattern2 = r'\[(\d+(?:,\s*\d+)*)\]'
    # Pattern 3: Author-year parenthetical citations (Author et al., Year)
    pattern3 = r'\(([A-Za-z\s]+(?:et al\.)?(?:,|\s)\s*\d{4}(?:;\s*[A-Za-z\s]+(?:et al\.)?(?:,|\s)\s*\d{4})*)\)'
    
    for pattern in [pattern1, pattern2, pattern3]:
        for match in re.finditer(pattern, text):
            citations.append((match.start(), match.end(), match.group(0)))
    
    return citations

def chunk_table_content(text: str) -> List[Chunk]:
    """Create chunks from tables in the text."""
    tables = detect_tables(text)
    chunks = []
    
    for i, (_, _, table_text) in enumerate(tables):
        chunk = Chunk(
            id=Chunk.generate_id(),
            text=table_text,
            source="table",
            metadata={
                "index": i,
                "total": len(tables),
                "content_type": "table"
            }
        )
        chunks.append(chunk)
    
    return chunks

def chunk_mathematical_formulas(text: str) -> List[Chunk]:
    """Create chunks from mathematical formulas in the text."""
    formulas = extract_mathematical_formulas(text)
    chunks = []
    
    for i, (_, _, formula_text) in enumerate(formulas):
        chunk = Chunk(
            id=Chunk.generate_id(),
            text=formula_text,
            source="formula",
            metadata={
                "index": i,
                "total": len(formulas),
                "content_type": "formula"
            }
        )
        chunks.append(chunk)
    
    return chunks

def chunk_with_citations(text: str, chunk_size: int = 100, overlap: int = 20) -> List[Chunk]:
    """Create chunks that preserve citation context."""
    citations = extract_citations(text)
    words = text.split()
    chunks = []
    
    # If no citations, fall back to standard chunking
    if not citations:
        return chunk_fixed(text, chunk_size, overlap)
    
    # Create positions map for words
    word_positions = []
    pos = 0
    for word in words:
        word_positions.append((pos, pos + len(word)))
        pos += len(word) + 1  # +1 for space
    
    # Find chunks that preserve citations
    start_idx = 0
    while start_idx < len(words):
        # Determine end index based on chunk size
        end_idx = min(start_idx + chunk_size, len(words))
        
        # Get text positions
        chunk_start_pos = word_positions[start_idx][0] if start_idx < len(word_positions) else 0
        chunk_end_pos = word_positions[end_idx-1][1] if end_idx-1 < len(word_positions) else len(text)
        
        # Check if we need to extend to include a citation
        for cit_start, cit_end, _ in citations:
            # If citation starts in this chunk but ends after it
            if chunk_start_pos <= cit_start < chunk_end_pos and cit_end > chunk_end_pos:
                # Find word index that includes the citation end
                for i, (wp_start, wp_end) in enumerate(word_positions[end_idx:], end_idx):
                    if wp_end >= cit_end:
                        end_idx = i + 1  # Include this word
                        break
                chunk_end_pos = word_positions[min(end_idx-1, len(word_positions)-1)][1]
        
        # Create the chunk
        chunk_text = ' '.join(words[start_idx:end_idx])
        if chunk_text:
            # Find citations in this chunk
            chunk_citations = []
            for _, _, cit_text in citations:
                if cit_text in chunk_text:
                    chunk_citations.append(cit_text)
            
            chunk = Chunk(
                id=Chunk.generate_id(),
                text=chunk_text,
                source="citation_aware",
                metadata={
                    "index": len(chunks),
                    "total": None,  # Will update later
                    "citations": chunk_citations,
                    "has_citations": len(chunk_citations) > 0
                }
            )
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start_idx += max(1, chunk_size - overlap)
    
    # Update total chunks metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk.metadata["total"] = total_chunks
    
    return chunks

def process_special_sections(sections: List[Section]) -> List[Chunk]:
    """Create special chunks for abstract and conclusion sections."""
    special_chunks = []
    
    for section in sections:
        section_title = section.title.lower() if section.title else ""
        if "abstract" in section_title or "conclusion" in section_title:
            # Process these sections with special attention
            chunk = Chunk(
                id=Chunk.generate_id(),
                text=section.content,
                source="special_section",
                metadata={
                    "title": section.title,
                    "section_type": "abstract" if "abstract" in section_title else "conclusion",
                    "importance": "high"
                }
            )
            special_chunks.append(chunk)
    
    return special_chunks

def chunk_document(text_or_sections: Union[str, List[Section]], strategy: str = "fixed", **kwargs) -> List[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(text_or_sections, **kwargs)
    elif strategy == "sentence":
        return chunk_by_sentence(text_or_sections)
    elif strategy == "paragraph":
        return chunk_by_paragraph(text_or_sections)
    elif strategy == "section":
        return chunk_by_sections(text_or_sections)
    elif strategy == "table":
        return chunk_table_content(text_or_sections)
    elif strategy == "formula":
        return chunk_mathematical_formulas(text_or_sections)
    elif strategy == "citation_aware":
        return chunk_with_citations(text_or_sections, **kwargs)
    elif strategy == "special_sections":
        return process_special_sections(text_or_sections)
    else:
        raise ValueError(f"Unknown strategy: {strategy}") 