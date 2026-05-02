import logging
from src.langchain_gpt.document_processing.grobid_client import GrobidClient
import fitz  # PyMuPDF
from essentials.phase3_1.chunking import split_text_to_token_chunks, Chunk
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO)

def parse_pdf(pdf_path):
    logging.info(f"Parsing PDF with Grobid: {pdf_path}")
    grobid_client = GrobidClient()
    document = grobid_client.pdf_to_document(pdf_path)
    
    metadata = {
        'title': document.metadata.title,
        'authors': document.metadata.authors,
        'references': document.metadata.references
    }
    
    # --- Post-process Grobid chunks with improved token-based chunking ---
    improved_chunks = []
    for orig_chunk in document.chunks:
        # Use the improved chunker on each chunk's text
        subchunks = split_text_to_token_chunks(orig_chunk.text, max_tokens=300)
        for i, subtext in enumerate(subchunks):
            improved_chunks.append(
                Chunk(
                    id=Chunk.generate_id(),
                    text=subtext,
                    source=orig_chunk.metadata.get('source', 'grobid'),
                    metadata={**(orig_chunk.metadata or {}), 'subchunk_index': i, 'parent_chunk_id': getattr(orig_chunk, 'id', None)}
                )
            )

    # Update sections to use improved chunks
    sections = [
        {
            'heading': chunk.metadata.get('heading', 'No Heading'),
            'text': chunk.text,
            'number': chunk.metadata.get('page', -1),
            'chunk': chunk
        }
        for chunk in improved_chunks
    ]
    
    # Extract pages using PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_data = {
            'number': page.number,
            'text': page.get_text(),
            'figures': []  # Empty list for figures
        }
        pages.append(page_data)
    doc.close()
    
    return {
        'title': metadata['title'],
        'authors': metadata['authors'],
        'references': metadata['references'],
        'sections': sections,
        'pages': pages,
        'metadata': metadata,
        'chunks': improved_chunks  # Use improved chunks
    } 