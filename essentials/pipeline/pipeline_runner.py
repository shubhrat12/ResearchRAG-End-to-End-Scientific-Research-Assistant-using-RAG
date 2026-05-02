import os, logging, warnings
os.environ["DISABLE_TQDM"] = "1"      # kill tqdm bars
import logging
from pathlib import Path
import sys
from tqdm import tqdm

# Disable tqdm globally to avoid closed file error in subprocess
tqdm.disable = True
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(Path(__file__).resolve().parent.parent.parent / "pipeline.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
import sys
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import traceback
from pathlib import Path
from essentials.pipeline.pdf_to_document import parse_pdf
from essentials.pipeline.section_classifier import classify_sections
from essentials.pipeline.ranking_engine import score_sections
from essentials.pipeline.summarizer_wrapper import summarize_sections
from essentials.pipeline.semantic_search_connector import fetch_additional_documents
from essentials.phase4_1.llm_runner import LLMRunner
from essentials.phase3_4.context_builder import ContextBuilder
from essentials.phase2_2.entity_extraction import ScientificEntityExtractor
from essentials.phase2_2.relation_extraction import PatternRelationExtractor
from essentials.phase2_2.claim_detection import ScientificClaimDetector
from essentials.phase2_3.bert_query_expansion import BERTQueryExpander
from essentials.phase2_3.field_specific_search import CitationNetwork
from essentials.phase2_3.field_specific_search import FieldSpecificSearchOptimizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import LayoutLMForTokenClassification
from essentials.phase2_3 import fix_figure_detection
from essentials.phase2_3.fix_figure_detection import process_figure_detection_data
from essentials.pipeline.figure_detector import detect_figures_in_pdf, convert_figures_to_chunks
import fitz  # PyMuPDF
import json
import chromadb
import argparse
from chromadb.config import Settings
from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding
try:
    from essentials.phase3_2.scientific_embeddings import ScientificEmbedding
    SCIENTIFIC_AVAILABLE = True
except ImportError:
    ScientificEmbedding = None
    SCIENTIFIC_AVAILABLE = False
from essentials.phase3_2.basic_embeddings import BasicEmbedding
import numpy as np
from essentials.phase3_1.models import Chunk
import sys
import uuid
from retrieval.safe_retriever import retrieve_chunks
from utils.token_utils import n_tokens
from llm_utils import safe_generate
from essentials.phase3_2.retrieval import Retriever
import yaml
from essentials.utils.token_budget import set_tokenizer, count_tokens, trim_to_tokens
import re
import shutil
import time

# Load models
section_classifier_model = AutoModelForSequenceClassification.from_pretrained(str(Path(__file__).resolve().parent.parent.parent / 'models/section_classifier/best_model'))
section_classifier_tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parent.parent.parent / 'models/section_classifier/best_model'))

figure_detector_model = LayoutLMForTokenClassification.from_pretrained(str(Path(__file__).resolve().parent.parent.parent / 'models/figure_detector/best_model'))
from transformers import LayoutLMTokenizerFast
figure_detector_tokenizer = LayoutLMTokenizerFast.from_pretrained(str(Path(__file__).resolve().parent.parent.parent / 'models/figure_detector/best_model'))

reference_parser_model = AutoModelForSequenceClassification.from_pretrained(str(Path(__file__).resolve().parent.parent.parent / 'models/reference_parser/best_model'))
reference_parser_tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parent.parent.parent / 'models/reference_parser/best_model'))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BUDGET = yaml.safe_load((PROJECT_ROOT / "essentials/config/prompt_budget.yaml").read_text())

# Define max chunk tokens as a constant
MAX_CHUNK_TOKENS = 1000
MAX_CONTEXT_TOKENS = 2000

# Helper to detect metadata-specific questions
METADATA_KEYWORDS = [
    "author", "title", "figure", "reference", "doi", "journal", "year", "publisher", "affiliation", "section", "table", "caption"
]
def is_metadata_question(q: str) -> bool:
    q_lower = q.lower()
    return any(kw in q_lower for kw in METADATA_KEYWORDS)

def select_embedder():
    if ScientificEmbedding is not None:
        logging.info("Using ScientificEmbedding for embedding.")
        return ScientificEmbedding()
    else:
        logging.warning("ScientificEmbedding not available. Falling back to AdvancedEmbedding.")
        return AdvancedEmbedding(enable_cache=True, enable_compression=True, target_dimensions=100)

def build_safe_prompt(metadata, system_prompt, question, context, max_tokens=4096, reserved_for_answer=600):
    from essentials.utils.token_budget import count_tokens, trim_to_tokens
    # Initial token counts
    meta_tokens = count_tokens(metadata)
    sys_tokens = count_tokens(system_prompt)
    q_tokens = count_tokens(question)
    ctx_tokens = count_tokens(context)
    max_prompt_tokens = max_tokens - reserved_for_answer
    # Start with full segments
    trimmed_metadata = metadata
    trimmed_sys = system_prompt
    trimmed_context = context
    # 1. Trim context first
    available_for_ctx = max_prompt_tokens - (meta_tokens + sys_tokens + q_tokens)
    if available_for_ctx < ctx_tokens:
        trimmed_context = trim_to_tokens(context, max(0, available_for_ctx))
        ctx_tokens = count_tokens(trimmed_context)
    # 2. If still over budget, trim metadata (to at least 200 tokens, but must include title and 2-3 references)
    total = count_tokens(trimmed_metadata) + count_tokens(trimmed_sys) + count_tokens(trimmed_context) + q_tokens
    if total > max_prompt_tokens:
        # Try to extract title and up to 3 references from metadata
        import re
        title_match = re.search(r'(Title of the paper:.*?\n)', trimmed_metadata)
        title_str = title_match.group(1) if title_match else ''
        refs_match = re.search(r'(References:\n)([\s\S]*)', trimmed_metadata)
        refs_str = ''
        if refs_match:
            refs_header = refs_match.group(1)
            refs_body = refs_match.group(2)
            refs_lines = refs_body.strip().split('\n')
            num_refs_to_include = 2
            refs_str = refs_header + '\n'.join(refs_lines[:num_refs_to_include]) + '\n'

        # Compose minimal metadata
        minimal_metadata = title_str + refs_str
        # If still too long, trim to 200 tokens
        trimmed_metadata = trim_to_tokens(minimal_metadata, 200)
        meta_tokens = count_tokens(trimmed_metadata)
        total = meta_tokens + count_tokens(trimmed_sys) + count_tokens(trimmed_context) + q_tokens
    # 3. If still over budget, trim system prompt to at least 20 tokens
    if total > max_prompt_tokens:
        min_sys_tokens = 20
        trimmed_sys = trim_to_tokens(system_prompt, min_sys_tokens)
        sys_tokens = count_tokens(trimmed_sys)
        total = meta_tokens + sys_tokens + count_tokens(trimmed_context) + q_tokens
    # Final log
    # Compose
    return trimmed_metadata + trimmed_sys + trimmed_context + question

def main():
    parser = argparse.ArgumentParser(description="Scientific PDF Pipeline (defaults to interactive query mode if --mode is not provided)")
    parser.add_argument('--mode', choices=['index', 'query'], default=None, help='Pipeline mode: index or query (default: query if not provided)')
    parser.add_argument('--pdf', type=str, help='Path to PDF (required for index mode)')
    parser.add_argument('--question', type=str, help='Question to ask (query mode)')
    parser.add_argument('--db_path', type=str, default='db/paper_index', help='Path to persistent vector DB')
    args = parser.parse_args()

    # If --mode is not provided, default to interactive query mode
    if args.mode is None:
        args.mode = 'query'
        args.question = None

    if args.mode == 'index':
        if not args.pdf:
            return
        try:
            # Clean re-index: remove db_path if exists
            if os.path.exists(args.db_path):
                shutil.rmtree(args.db_path)
            
            logging.info(f"Starting pipeline for {args.pdf}")
            
            # Step 1: Parse PDF
            logging.info("Step 1: Parsing PDF")
            document = parse_pdf(str(args.pdf))
            logging.info(f"Parsed {len(document['sections'])} sections")
            
            # Extract additional metadata
            document_title = document.get('title', 'Unknown')
            authors_list = document.get('metadata', {}).get('authors', ['Unknown'])
            references_list = document.get('metadata', {}).get('references', ['Unknown'])

            # --- Build metadata block for LLM prompt (moved up) ---
            formatted_references = "\n".join([
                f"- {ref.get('title', 'Unknown')} by {', '.join(ref.get('authors', []))} ({ref.get('year', '')})" + (f", DOI: {ref.get('doi')}" if ref.get('doi') else "")
                for ref in references_list
            ])
            metadata_block = f"Title of the paper: {document_title}\nAuthors: {', '.join(authors_list)}\nReferences:\n{formatted_references}\n"

            # Log extracted metadata
            logging.info(f"Title: {document_title}")
            logging.info(f"Authors: {', '.join(authors_list)}")
            logging.info(f"References:")
            for ref in references_list:
                if isinstance(ref, dict):
                    logging.info(f"- {ref.get('reference', 'Unknown')}")
                else:
                    logging.info(f"- {ref}")
            
            # Initialize extractors and detectors
            entity_extractor = ScientificEntityExtractor()
            relation_extractor = PatternRelationExtractor()
            claim_detector = ScientificClaimDetector()
            query_expander = BERTQueryExpander()
            citation_network = CitationNetwork()
            search_optimizer = FieldSpecificSearchOptimizer(field='biology')

            # Step 1.5: Extract Entities
            logging.info("Step 1.5: Extracting entities")
            for section in document['sections']:
                entities = entity_extractor.extract_entities(section['text'])
                section['entities'] = entities
            logging.info("Extracted entities for all sections")

            # Step 1.6: Extract Relations
            logging.info("Step 1.6: Extracting relations")
            for section in document['sections']:
                relations = relation_extractor.extract_relations(section['text'])
                section['relations'] = relations
            logging.info("Extracted relations for all sections")

            # Step 1.7: Detect Claims
            logging.info("Step 1.7: Detecting claims")
            for section in document['sections']:
                claims = claim_detector.extract_claims(section['text'])
                section['claims'] = claims
            logging.info("Detected claims for all sections")

            # Step 2: Classify Sections with the new model
            logging.info("Step 2: Classifying sections with the fine-tuned model")
            for section in document['sections']:
                inputs = section_classifier_tokenizer(section['text'], return_tensors='pt')
                outputs = section_classifier_model(**inputs)
                section['label'] = outputs.logits.argmax(-1).item()
            logging.info(f"Classified {len(document['sections'])} sections with the fine-tuned model")

            # Step 2.5: Extract text and bounding boxes from PDF
            logging.info("Step 2.5: Extracting text and bounding boxes from PDF")

            def extract_text_and_boxes(pdf_path):
                doc = fitz.open(pdf_path)
                extracted_data = []

                for page in doc:
                    words = page.get_text("words")
                    page_data = {
                        "words": [],
                        "boxes": []
                    }
                    for w in words:
                        x0, y0, x1, y1, text, *_ = w
                        if text.strip():
                            page_data["words"].append(text)
                            page_data["boxes"].append([int(x0), int(y0), int(x1), int(y1)])
                    extracted_data.append(page_data)

                doc.close()
                return extracted_data

            # Extract data from PDF
            extracted_data = extract_text_and_boxes(args.pdf)

            # Save extracted data to a temporary JSON file
            temp_json_path = str(PROJECT_ROOT / "temp_extracted_data.json")
            with open(temp_json_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=2)

            # Pre-process the data
            preprocessed_data_path = process_figure_detection_data(temp_json_path)

            # Ensure document['pages'] is populated
            if not document.get('pages'):
                logging.info("Pages are missing or empty, extracting using PyMuPDF")
                doc = fitz.open(args.pdf)
                pages = []
                for page in doc:
                    page_data = {
                        'number': page.number,
                        'text': page.get_text(),
                        'figures': []  # Empty list for figures
                    }
                    pages.append(page_data)
                doc.close()
                document['pages'] = pages

            # Update figure detection logic
            detected_figures_per_page = detect_figures_in_pdf(args.pdf, figure_detector_tokenizer, figure_detector_model)
            for i, figures in enumerate(detected_figures_per_page):
                document['pages'][i]['figures'] = figures
            logging.info(f"Detected figures: {document['pages']}")

            # Convert detected figures to context-builder-compatible chunks
            figure_chunks = convert_figures_to_chunks(detected_figures_per_page, args.pdf)

            # Prepare section/content chunks (from section['chunk'])
            section_chunks = []
            for section in document['sections']:
                if 'chunk' in section:
                    section_chunks.append({
                        "id": getattr(section['chunk'], 'id', None),
                        "text": getattr(section['chunk'], 'text', None),
                        "metadata": getattr(section['chunk'], 'metadata', None),
                        "score": section.get('score', 1.0)
                    })

            # Merge figure and section chunks
            all_chunks = ContextBuilder.merge_figure_and_section_chunks(figure_chunks, section_chunks)
            print("=== Chunk Previews ===")
            for ch in all_chunks:
                text = ch.get('text', ch.get('page_content', ''))

            # Step 2.6: Parse and clean references
            logging.info("Step 2.6: Ensuring references are structured and clean")
            # Ensure references_list is a list of dicts with required fields
            cleaned_references = []
            for ref in references_list:
                if not isinstance(ref, dict):
                    continue
                cleaned_ref = {
                    'title': ref.get('title', '').strip(),
                    'authors': ref.get('authors', []) if isinstance(ref.get('authors', []), list) else [],
                    'year': ref.get('year', ''),
                    'journal': ref.get('journal', ''),
                    'doi': ref.get('doi', ''),
                }
                # Only add if at least title is present
                if cleaned_ref['title']:
                    cleaned_references.append(cleaned_ref)
            # Store back in document['metadata']['references']
            document['metadata']['references'] = cleaned_references
            references_list = cleaned_references
            logging.info(f"Structured {len(references_list)} references for downstream use.")

            # Step 3: Score Document
            logging.info("Step 3: Scoring sections")
            scored_sections = score_sections(document['sections'])
            logging.info(f"Scored {len(scored_sections)} sections")
            
            # Step 4: Summarize Sections
            logging.info("Step 4: Summarizing sections")
            summaries = summarize_sections(scored_sections)
            logging.info(f"Generated {len(summaries)} summaries")
            
            # Step 5: Fetch Additional Documents if needed
            logging.info("Step 5: Fetching additional documents")
            related_papers = fetch_additional_documents(document['title'])
            logging.info(f"Found {len(related_papers)} related papers")
            
            # Output section titles and their summaries
            for section, summary in zip(scored_sections, summaries):
                logging.info(f"Section: {section['label']} - {section['heading']}")
                logging.info(f"Summary: {summary}")
            
            logging.info("Pipeline completed successfully.")

            # Instantiate LLMRunner
            llm_runner = LLMRunner()

            # Print each chunk's preview
            for section in document['sections']:
                if 'chunk' in section:
                    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                        pass

            # Initialize ContextBuilder with modified settings
            context_builder = ContextBuilder(deduplicate=False, diversify=False, coherence_check=False, max_tokens=4096, debug=True)

            # --- Limit and debug context chunks before LLM preview/call ---
           
            # Separate figure, reference, and normal chunks
            figure_chunks_sel = [ch for ch in all_chunks if str(ch.get('metadata', {}).get('content_type', '')).lower() == 'figure']
            reference_chunks_sel = [ch for ch in all_chunks if str(ch.get('metadata', {}).get('content_type', '')).lower() == 'reference']
            normal_chunks_sel = [ch for ch in all_chunks if ch not in figure_chunks_sel and ch not in reference_chunks_sel]
            # Always include at least 2 figure chunks if available
            selected_chunks = []
            running_tokens = 0
            dropped_chunks = 0
            # Add up to 2 figure chunks first
            fig_tokens = 0
            for ch in figure_chunks_sel[:2]:
                # Only pass the caption (assume in 'text' or 'page_content')
                caption = ch.get('text', ch.get('page_content', ''))
                # Remove all metadata except caption
                ch_caption_only = {'text': caption, 'metadata': {}}
                ch_tokens = count_tokens(caption)
                if ch_tokens > MAX_CHUNK_TOKENS:
                    logging.warning(f"Chunk exceeds MAX_CHUNK_TOKENS ({ch_tokens} > {MAX_CHUNK_TOKENS}), trimming.")
                    caption = trim_to_tokens(caption, MAX_CHUNK_TOKENS)
                    ch_tokens = count_tokens(caption)
                ch_caption_only['text'] = caption
                selected_chunks.append(ch_caption_only)
                running_tokens += ch_tokens
                fig_tokens += ch_tokens
            # Add all reference chunks (never skip for being short)
            ref_tokens = 0
            for ch in reference_chunks_sel[:3]:
                ch_text = ch.get('page_content', ch.get('text', ch.get('text', '')))
                ch_tokens = count_tokens(ch_text)
                if ch_tokens > MAX_CHUNK_TOKENS:
                    logging.warning(f"Chunk exceeds MAX_CHUNK_TOKENS ({ch_tokens} > {MAX_CHUNK_TOKENS}), trimming.")
                    ch_text = trim_to_tokens(ch_text, MAX_CHUNK_TOKENS)
                    ch_tokens = count_tokens(ch_text)
                if 'page_content' in ch:
                    ch['page_content'] = ch_text
                else:
                    ch['text'] = ch_text
                selected_chunks.append(ch)
                running_tokens += ch_tokens
                ref_tokens += ch_tokens
            # Add remaining figure chunks (after 2), then normal chunks
            for ch in figure_chunks_sel[2:] + normal_chunks_sel:
                ch_text = ch.get('page_content', ch.get('text', ch.get('text', '')))
                ch_tokens = count_tokens(ch_text)
                # Never skip figure/reference for being short (<10 tokens)
                if ch_tokens < 10 and str(ch.get('metadata', {}).get('content_type', '')).lower() not in ['figure', 'reference']:
                    continue
                if ch_tokens > MAX_CHUNK_TOKENS:
                    logging.warning(f"Chunk exceeds MAX_CHUNK_TOKENS ({ch_tokens} > {MAX_CHUNK_TOKENS}), trimming.")
                    ch_text = trim_to_tokens(ch_text, MAX_CHUNK_TOKENS)
                    ch_tokens = count_tokens(ch_text)
                if running_tokens + ch_tokens > MAX_CONTEXT_TOKENS:
                    dropped_chunks += 1
                    continue
                if 'page_content' in ch:
                    ch['page_content'] = ch_text
                else:
                    ch['text'] = ch_text
                selected_chunks.append(ch)
                running_tokens += ch_tokens
            
            SYSTEM_TXT = metadata_block + "\nYou are an expert scientific assistant. Use the following extracted sections from a scientific paper to answer the user's question as accurately as possible."
            dummy_question = 'Summarize the main contributions of this paper.'
            if not selected_chunks:
                context = ""
            else:
                context = context_builder.build_context_v2(selected_chunks, SYSTEM_TXT, dummy_question, llm_runner)
                context = "Context:\n" + context
                total_tokens = count_tokens(context)
                
                if total_tokens > MAX_CONTEXT_TOKENS:
                    context = trim_to_tokens(context, MAX_CONTEXT_TOKENS)
                    total_tokens = count_tokens(context)

            # Pass metadata block + context to LLMRunner.generate
            # system_prompt = "\nYou are an expert scientific assistant. Use the following extracted sections from a scientific paper to answer the user's question as accurately as possible.\n\nPaper Context:\n"
            # question_str = ""
            # full_prompt = build_safe_prompt(metadata_block, system_prompt, question_str, context, max_tokens=4096, reserved_for_answer=600)
            # response = llm_runner.generate(full_prompt, '', title=document_title, authors=authors_list, references=references_list)
            # logging.info(f"Generated Response: {response}")

            # Step 5.1: Expand Queries
            logging.info("Step 5.1: Expanding queries")
            expanded_title = query_expander.expand_query([document['title']])
            logging.info(f"Expanded Title: {expanded_title}")

            # Step 5.2: Analyze Citation Network
            logging.info("Step 5.2: Analyzing citation network")
            for paper in related_papers:
                citation_network.add_paper(paper['id'], paper['title'])
                for citation in paper.get('citations', []):
                    citation_network.add_citation(paper['id'], citation)
            logging.info(citation_network.get_network_info())

            # Step 5.3: Rank Results
            logging.info("Step 5.3: Ranking results")
            relevance_scores = search_optimizer.calculate_relevance_scores(document['title'], [paper['title'] for paper in related_papers])
            logging.info(f"Relevance Scores: {relevance_scores}")

            # Step 3.5: Index chunks in ChromaDB
            # Extract all section['chunk']s
            chunk_list = [section['chunk'] for section in document['sections'] if 'chunk' in section]
            # Ensure all chunks have id, text, metadata
            safe_chunks = []
            for chunk in chunk_list:
                chunk_id = getattr(chunk, 'id', None)
                chunk_text = getattr(chunk, 'text', None)
                chunk_metadata = getattr(chunk, 'metadata', None)
                if not chunk_id:
                    chunk_id = chunk_metadata.get('document_id') if chunk_metadata else str(uuid.uuid4())
                if not chunk_text:
                    continue
                if not chunk_metadata:
                    chunk_metadata = {}
                # Wrap as a simple object with id, text, metadata
                class SimpleChunk:
                    def __init__(self, id, text, metadata):
                        self.id = id
                        self.text = text
                        self.metadata = metadata
                safe_chunks.append(SimpleChunk(chunk_id, chunk_text, chunk_metadata))
            embedder = select_embedder()
            # Embed chunks with diagnostics
            embedded_chunks = embedder.embed_chunks(safe_chunks)
            # Log embedding diagnostics (already logged inside embed_chunks)
            # Prepare ChromaDB
            collection_name = "paper_chunks"
            db_path = "D:/Langchain Project/db/paper_index"
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(collection_name)
            logger.info(f"✅ ChromaDB collection contains {collection.count()} documents.")
            def clean_metadata(meta):
                # Remove keys with None values, or convert None to ""
                if not meta:
                    return {}
                return {k: ("" if v is None else v) for k, v in meta.items() if v is not None}
            # Store chunks and embeddings
            for chunk, emb in zip(safe_chunks, embedded_chunks):
                collection.add(
                    documents=[chunk.text],
                    embeddings=[emb['embedding']],
                    metadatas=[clean_metadata(chunk.metadata)],
                    ids=[chunk.id]
                )
            logger.info(f"Collection now has {collection.count()} entries.")
            print("Successfully inserted chunks into ChromaDB")
            logging.info("ChromaDB persistence is handled automatically by the client version in use.")
            logging.info(f"Using ChromaDB path: {db_path}")

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            logging.error(traceback.format_exc())
            with open(str(PROJECT_ROOT / "indexing_failed.txt"), "w", encoding="utf-8") as f:
                f.write("Pipeline failed:\n")
                f.write(str(e) + "\n")
                f.write(traceback.format_exc())
            input("Press Enter to exit...")
            sys.exit(1)

    elif args.mode == 'query':
        try:
            # Load persistent vector DB
            collection_name = "paper_chunks"
            db_path = "D:/Langchain Project/db/paper_index"
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(collection_name)
            logger.info(f"✅ ChromaDB collection contains {collection.count()} documents.")
            logging.info(f"Using ChromaDB path: {db_path}")
            # Log and check collection count before querying
            doc_count = collection.count()
            logging.info(f"ChromaDB collection '{collection_name}' has {doc_count} documents before query.")
            if doc_count == 0:
                logging.warning("No documents found in ChromaDB collection before querying. Please re-run in index mode to populate the collection.")

            # Load paper metadata from metadata.json in db_path
            metadata_path = str(Path(db_path) / "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    paper_metadata = json.load(f)
                document_title = paper_metadata.get("title", "Unknown")
                authors_list = paper_metadata.get("authors", ["Unknown"])
                references_list = paper_metadata.get("references", [])
            else:
                document_title = "Unknown"
                authors_list = ["Unknown"]
                references_list = []

            # Choose embedder
            embedder = select_embedder()
            llm_runner = LLMRunner()
            context_builder = ContextBuilder(deduplicate=False, diversify=False, coherence_check=False, max_tokens=4096, debug=True)

            retriever = Retriever(vector_store=collection, embedding_model=embedder)

            def ask_and_answer(question):
                # Set the tokenizer for token_budget to match the LLM
                if hasattr(llm_runner, 'tokenizer'):
                    set_tokenizer(llm_runner.tokenizer, getattr(llm_runner, 'model_name', None))
                # Build metadata block for SYSTEM_TXT
                formatted_references = "\n".join([
                    f"- {ref.get('title', 'Unknown')} by {', '.join(ref.get('authors', []))} ({ref.get('year', '')})" + (f", DOI: {ref.get('doi')}") if ref.get('doi') else ""
                    for ref in references_list
                ])
                metadata_block = f"Title of the paper: {document_title}\nAuthors: {', '.join(authors_list)}\nReferences:\n{formatted_references}\nYou are an expert scientific assistant. Use the following extracted sections from a scientific paper to answer the user's question as accurately as possible."
                SYSTEM_TXT = metadata_block
                # --- Detect question intent ---
                question_lower = question.lower()
                wants_figures = any(keyword in question_lower for keyword in ["figure", "caption", "diagram", "image", "chart"])
                wants_references = any(keyword in question_lower for keyword in ["reference", "citation", "bibliography"])
                # --- Retrieve only the top 3 chunks (for metadata fallback logic) ---
                retrieval_start = time.time()
                raw_chunks = retrieve_chunks(question, k=3, retriever=retriever)[:3]
                retrieval_time = time.time() - retrieval_start
                logging.info(f"Vector DB query step took %.2f seconds", retrieval_time)
                # --- Chunk selection logic ---
                # Get all chunks from the DB for this question
                figure_chunks_sel = []
                reference_chunks_sel = []
                normal_chunks_sel = []
                if wants_figures:
                    # Only include figure chunks
                    figure_chunks_sel = [ch for ch in raw_chunks if str(ch.get('metadata', {}).get('content_type', '')).lower() == 'figure']
                if wants_references:
                    # Only include reference chunks
                    reference_chunks_sel = [ch for ch in raw_chunks if str(ch.get('metadata', {}).get('content_type', '')).lower() == 'reference']
                if not wants_figures and not wants_references:
                    # Prioritize normal text chunks
                    # Dynamically rank normal text chunks by similarity
                    normal_chunks_sel = retriever.retrieve(question, k=10)
                selected_chunks = []
                running_tokens = 0
                dropped_chunks = 0
                fig_tokens = 0
                ref_tokens = 0
                # Set context token limit for this function
                MAX_CONTEXT_TOKENS = 3000
                # Add figure chunks if needed
                if wants_figures:
                    for ch in figure_chunks_sel[:2]:
                        caption = ch.get('text', ch.get('page_content', ''))
                        ch_caption_only = {'text': caption, 'metadata': {}}
                        ch_tokens = count_tokens(caption)
                        if ch_tokens > MAX_CHUNK_TOKENS:
                            logging.warning(f"Chunk exceeds MAX_CHUNK_TOKENS ({ch_tokens} > {MAX_CHUNK_TOKENS}), trimming.")
                            caption = trim_to_tokens(caption, MAX_CHUNK_TOKENS)
                            ch_tokens = count_tokens(caption)
                        ch_caption_only['text'] = caption
                        selected_chunks.append(ch_caption_only)
                        running_tokens += ch_tokens
                        fig_tokens += ch_tokens
                # Add reference chunks if needed
                if wants_references:
                    for ch in reference_chunks_sel[:3]:
                        ch_text = ch.get('page_content', ch.get('text', ch.get('text', '')))
                        ch_tokens = count_tokens(ch_text)
                        if ch_tokens > MAX_CHUNK_TOKENS:
                            logging.warning(f"Chunk exceeds MAX_CHUNK_TOKENS ({ch_tokens} > {MAX_CHUNK_TOKENS}), trimming.")
                            ch_text = trim_to_tokens(ch_text, MAX_CHUNK_TOKENS)
                            ch_tokens = count_tokens(ch_text)
                        if 'page_content' in ch:
                            ch['page_content'] = ch_text
                        else:
                            ch['text'] = ch_text
                        selected_chunks.append(ch)
                        running_tokens += ch_tokens
                        ref_tokens += ch_tokens
                # Add normal text chunks if neither figures nor references are requested
                if not wants_figures and not wants_references:
                    for ch in normal_chunks_sel:
                        ch_text = ch.get('page_content', ch.get('text', ch.get('text', '')))
                        ch_tokens = count_tokens(ch_text)
                        if ch_tokens < 10:
                            continue
                        if ch_tokens > MAX_CHUNK_TOKENS:
                            logging.warning(f"Chunk exceeds MAX_CHUNK_TOKENS ({ch_tokens} > {MAX_CHUNK_TOKENS}), trimming.")
                            ch_text = trim_to_tokens(ch_text, MAX_CHUNK_TOKENS)
                            ch_tokens = count_tokens(ch_text)
                        if running_tokens + ch_tokens > MAX_CONTEXT_TOKENS:
                            dropped_chunks += 1
                            continue
                        if 'page_content' in ch:
                            ch['page_content'] = ch_text
                        else:
                            ch['text'] = ch_text
                        selected_chunks.append(ch)
                        running_tokens += ch_tokens
                # --- Fallback logging ---
                if len(selected_chunks) == 0:
                    logging.warning("⚠️ No valid chunks retrieved. Falling back to LLM.")
                # --- Post-retrieval debug logging ---
                for idx, ch in enumerate(selected_chunks):
                    chunk_id = ch.get('id', ch.get('metadata', {}).get('id', f'chunk{idx}'))
                    chunk_text = ch.get('text', ch.get('page_content', ''))
                    logging.info(f"Retrieved chunk [{chunk_id}]: \"{chunk_text[:100]}...\"")
                # --- Aggressive emergency trimming ---
                prompt_context = context_builder.build_context_v2(selected_chunks, SYSTEM_TXT, question, llm_runner)
                # Warn if context is very small
                if count_tokens(prompt_context) < 100:
                    logging.warning("⚠️ Context may be incomplete. Answer may be vague.")
                system_prompt = SYSTEM_TXT + "\n\nPaper Context:\n"
                question_str = question
                prompt = build_safe_prompt("", system_prompt, question_str, prompt_context, max_tokens=4096, reserved_for_answer=600)
                # LLM timing
                llm_start = time.time()
                try:
                    response = llm_runner.generate(prompt)
                except Exception as e:
                    raise
                llm_time = time.time() - llm_start
                logging.info("LLM response step took %.2f seconds", llm_time)
                # This line is required for Streamlit to display the answer
                print("LLM Answer:", response)
           

            if args.question:
                ask_and_answer(args.question)
            else:
                while True:
                    q = input("Enter your question (or type 'exit' to quit): ")
                    if q.strip().lower() == 'exit':
                        break
                    ask_and_answer(q)
        except Exception as e:
            logging.error(f"Query mode failed: {str(e)}")
            logging.error(traceback.format_exc())
            sys.exit(1)

if __name__ == "__main__":
    main() 