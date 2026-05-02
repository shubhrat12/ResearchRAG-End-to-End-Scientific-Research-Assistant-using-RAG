"""
Inference pipeline for enhanced PDF processing.

This module implements a unified inference pipeline that uses all fine-tuned models
for processing scientific papers.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Import models
from .models.section_classifier import SectionClassifier
from .models.figure_detector import FigureDetector
from .models.reference_parser import ReferenceParser

# Import document processing modules
from ..document_processing.pdf_extractor_factory import PDFExtractorFactory
from ..document_processing.pdf_extractors.hybrid_extractor import HybridExtractor

# Set up logging
logger = logging.getLogger(__name__)


class EnhancedPDFProcessor:
    """
    Enhanced PDF processing pipeline using transfer learning models.
    
    This pipeline combines section classification, figure detection, and reference parsing
    to create a structured representation of a scientific paper.
    """
    
    def __init__(
        self,
        section_classifier_path: str = "models/section_classifier",
        figure_detector_path: str = "models/figure_detector",
        reference_parser_path: str = "models/reference_parser",
        output_dir: str = "data/processed/structured",
        use_gpu: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            section_classifier_path: Path to the section classifier model
            figure_detector_path: Path to the figure detector model
            reference_parser_path: Path to the reference parser model
            output_dir: Directory to save processed outputs
            use_gpu: Whether to use GPU acceleration if available
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device based on availability
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models(
            section_classifier_path, 
            figure_detector_path,
            reference_parser_path
        )
        
        # Initialize PDF extractor
        self.pdf_extractor = PDFExtractorFactory.create("hybrid")
        if not isinstance(self.pdf_extractor, HybridExtractor):
            logger.warning("Using fallback extractor instead of HybridExtractor")
        
        logger.info("Enhanced PDF processor initialized")
    
    def _init_models(
        self,
        section_classifier_path: str,
        figure_detector_path: str,
        reference_parser_path: str
    ) -> None:
        """
        Initialize all models.
        
        Args:
            section_classifier_path: Path to the section classifier model
            figure_detector_path: Path to the figure detector model
            reference_parser_path: Path to the reference parser model
        """
        # Initialize section classifier
        logger.info("Initializing section classifier")
        self.section_classifier = SectionClassifier(
            model_dir=section_classifier_path,
            device=self.device
        )
        
        try:
            self.section_classifier.load_model()
            logger.info("Section classifier loaded successfully")
        except FileNotFoundError:
            logger.warning(
                f"Section classifier model not found at {section_classifier_path}. "
                "Using a new untrained model."
            )
        
        # Initialize figure detector
        logger.info("Initializing figure detector")
        self.figure_detector = FigureDetector(
            model_dir=figure_detector_path,
            device=self.device
        )
        
        try:
            self.figure_detector.load_model()
            logger.info("Figure detector loaded successfully")
        except FileNotFoundError:
            logger.warning(
                f"Figure detector model not found at {figure_detector_path}. "
                "Using a new untrained model."
            )
        
        # Initialize reference parser
        logger.info("Initializing reference parser")
        self.reference_parser = ReferenceParser(
            model_dir=reference_parser_path,
            device=self.device
        )
        
        try:
            self.reference_parser.load_model()
            logger.info("Reference parser loaded successfully")
        except FileNotFoundError:
            logger.warning(
                f"Reference parser model not found at {reference_parser_path}. "
                "Using a new untrained model."
            )
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Process a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with structured document information
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text and layout from PDF
        extraction_result = self.pdf_extractor.extract(str(pdf_path))
        
        # Process extracted text
        document = {}
        
        # Extract metadata
        document['metadata'] = extraction_result.get('metadata', {})
        
        # Process sections
        document['sections'] = self._process_sections(extraction_result)
        
        # Process figures and tables
        document['figures'] = self._process_figures(extraction_result)
        
        # Process references
        document['references'] = self._process_references(extraction_result)
        
        # Link in-text citations to references
        document['citation_links'] = self._link_citations(
            extraction_result, 
            document['references']
        )
        
        # Save processed document
        output_path = self.output_dir / f"{pdf_path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump(document, f, indent=2)
        
        logger.info(f"Document processed and saved to {output_path}")
        
        return document
    
    def _process_sections(self, extraction_result: Dict) -> List[Dict]:
        """
        Process and classify document sections.
        
        Args:
            extraction_result: Extracted document content
            
        Returns:
            List of processed sections with classification
        """
        sections = []
        
        # Extract text sections from the document
        body_text = extraction_result.get('body_text', [])
        
        # Classify each section
        section_texts = [section.get('text', '') for section in body_text]
        section_names = [section.get('section', '') for section in body_text]
        
        # Predict section types using the classifier
        if section_texts:
            try:
                section_predictions = self.section_classifier.predict(section_texts)
                
                for i, (section, prediction) in enumerate(zip(body_text, section_predictions)):
                    sections.append({
                        'section_id': i,
                        'section_name': section.get('section', ''),
                        'section_type': prediction['section_type'],
                        'confidence': prediction['confidence'],
                        'text': section.get('text', ''),
                        'start_page': section.get('start_page', 0),
                        'end_page': section.get('end_page', 0)
                    })
                
                logger.info(f"Classified {len(sections)} sections")
            except Exception as e:
                logger.error(f"Error classifying sections: {str(e)}")
                # Fallback: Use original sections without classification
                for i, section in enumerate(body_text):
                    sections.append({
                        'section_id': i,
                        'section_name': section.get('section', ''),
                        'section_type': 'unknown',
                        'confidence': 0.0,
                        'text': section.get('text', ''),
                        'start_page': section.get('start_page', 0),
                        'end_page': section.get('end_page', 0)
                    })
        
        return sections
    
    def _process_figures(self, extraction_result: Dict) -> List[Dict]:
        """
        Process and detect figures in the document.
        
        Args:
            extraction_result: Extracted document content
            
        Returns:
            List of detected figures and tables
        """
        figures = []
        
        # Extract layout information
        layout_info = extraction_result.get('layout', {})
        pages = layout_info.get('pages', [])
        
        # Process each page
        for page_idx, page in enumerate(pages):
            # Extract boxes from the page
            words = page.get('words', [])
            boxes = page.get('boxes', [])
            
            if not words or not boxes or len(words) != len(boxes):
                logger.warning(f"Invalid layout information for page {page_idx}")
                continue
                
            # Create document data for prediction
            doc_data = {
                'document_id': extraction_result.get('paper_id', f"doc_{page_idx}"),
                'page_num': page_idx,
                'words': words,
                'boxes': boxes
            }
            
            # Detect layout elements
            try:
                predictions = self.figure_detector.predict([doc_data])
                
                if predictions:
                    # Extract figure and table elements
                    pred = predictions[0]
                    
                    # Extract figures
                    for fig_idx, figure_element in enumerate(pred['grouped_predictions'].get('figure', [])):
                        figures.append({
                            'element_id': f"fig_{page_idx}_{fig_idx}",
                            'element_type': 'figure',
                            'page': page_idx,
                            'box': figure_element['box'],
                            'confidence': figure_element['confidence']
                        })
                    
                    # Extract tables
                    for table_idx, table_element in enumerate(pred['grouped_predictions'].get('table', [])):
                        figures.append({
                            'element_id': f"table_{page_idx}_{table_idx}",
                            'element_type': 'table',
                            'page': page_idx,
                            'box': table_element['box'],
                            'confidence': table_element['confidence']
                        })
                        
                logger.info(f"Detected elements on page {page_idx}: "
                          f"{len([f for f in figures if f['page'] == page_idx])}")
            except Exception as e:
                logger.error(f"Error detecting figures on page {page_idx}: {str(e)}")
        
        return figures
    
    def _process_references(self, extraction_result: Dict) -> List[Dict]:
        """
        Process and parse references in the document.
        
        Args:
            extraction_result: Extracted document content
            
        Returns:
            List of parsed references
        """
        references = []
        
        # Extract references from the document
        ref_entries = extraction_result.get('bib_entries', {})
        
        # Create reference strings
        ref_strings = []
        ref_ids = []
        
        for ref_id, ref_data in ref_entries.items():
            title = ref_data.get('title', '')
            authors = [a.get('first', '') + ' ' + a.get('last', '') for a in ref_data.get('authors', [])]
            year = ref_data.get('year', '')
            venue = ref_data.get('venue', '')
            
            # Create simple reference string
            ref_string = ""
            if authors:
                if len(authors) == 1:
                    ref_string += authors[0]
                elif len(authors) == 2:
                    ref_string += f"{authors[0]} and {authors[1]}"
                else:
                    ref_string += f"{authors[0]} et al."
            
            if year:
                ref_string += f" ({year})"
            
            if title:
                ref_string += f" {title}"
            
            if venue:
                ref_string += f". {venue}"
            
            ref_strings.append(ref_string)
            ref_ids.append(ref_id)
        
        # Parse references using the reference parser
        if ref_strings:
            try:
                parsed_refs = self.reference_parser.predict(ref_strings)
                
                for i, (ref_id, ref_string, parsed_ref) in enumerate(zip(ref_ids, ref_strings, parsed_refs)):
                    references.append({
                        'ref_id': ref_id,
                        'ref_index': i,
                        'ref_string': ref_string,
                        'components': parsed_ref['components'],
                        'original_data': ref_entries[ref_id] if ref_id in ref_entries else {}
                    })
                
                logger.info(f"Parsed {len(references)} references")
            except Exception as e:
                logger.error(f"Error parsing references: {str(e)}")
                # Fallback: Use original reference data
                for i, (ref_id, ref_string) in enumerate(zip(ref_ids, ref_strings)):
                    if ref_id in ref_entries:
                        references.append({
                            'ref_id': ref_id,
                            'ref_index': i,
                            'ref_string': ref_string,
                            'components': {},
                            'original_data': ref_entries[ref_id]
                        })
        
        return references
    
    def _link_citations(self, extraction_result: Dict, references: List[Dict]) -> Dict:
        """
        Link in-text citations to references.
        
        Args:
            extraction_result: Extracted document content
            references: Parsed references
            
        Returns:
            Dictionary mapping citation strings to reference indices
        """
        # Extract citation markers from body text
        citation_markers = []
        
        for section in extraction_result.get('body_text', []):
            text = section.get('text', '')
            cite_spans = section.get('cite_spans', [])
            
            for cite_span in cite_spans:
                start = cite_span.get('start', 0)
                end = cite_span.get('end', 0)
                ref_id = cite_span.get('ref_id', '')
                
                if start < end and start >= 0 and end <= len(text):
                    citation_text = text[start:end]
                    citation_markers.append({
                        'text': citation_text,
                        'ref_id': ref_id
                    })
        
        # Try to link citations using the reference parser
        citation_texts = [marker['text'] for marker in citation_markers]
        citation_links = {}
        
        if citation_texts and references:
            try:
                links = self.reference_parser.link_citations(citation_texts, references)
                
                # Convert to ref_ids
                for i, marker in enumerate(citation_markers):
                    if marker['text'] in links:
                        ref_index = links[marker['text']]
                        if 0 <= ref_index < len(references):
                            citation_links[marker['text']] = references[ref_index]['ref_id']
                
                logger.info(f"Linked {len(citation_links)} citations to references")
            except Exception as e:
                logger.error(f"Error linking citations: {str(e)}")
                # Fallback: Use original ref_ids if available
                for marker in citation_markers:
                    if marker['ref_id']:
                        citation_links[marker['text']] = marker['ref_id']
        
        return citation_links
    
    def process_directory(self, input_dir: Union[str, Path]) -> List[Dict]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Input directory containing PDF files
            
        Returns:
            List of processed documents
        """
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Invalid directory: {input_dir}")
        
        logger.info(f"Processing PDFs in directory: {input_dir}")
        
        # Get all PDF files in the directory
        pdf_files = list(input_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(pdf_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        return results 