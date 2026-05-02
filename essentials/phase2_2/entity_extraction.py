"""
Entity Extraction module for Scientific NLP Pipeline (Phase 2.2)

This module uses SciSpacy models to extract scientific entities from text,
focusing on biomedical and scientific entities.
"""

import os
import json
import logging
import spacy
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Define PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Create logs directory
logs_dir = PROJECT_ROOT / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "entity_extraction.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("entity_extraction")

class ScientificEntityExtractor:
    """Extract scientific entities from text using SciSpacy models."""
    
    def __init__(self, model_name: str = "en_core_sci_sm"):
        """
        Initialize the entity extractor with a specific SciSpacy model.
        
        Args:
            model_name: Name of the SciSpacy model to use. Default is "en_core_sci_sm".
                Other options include "en_ner_bionlp13cg_md" for biomedical entities.
        """
        logger.info(f"Initializing ScientificEntityExtractor with model: {model_name}")
        
        try:
            # Try to load the model
            self.nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except OSError:
            # If model not found, try to download it
            logger.warning(f"Model {model_name} not found. Attempting to download...")
            try:
                os.system(f"python -m spacy download {model_name}")
                self.nlp = spacy.load(model_name)
                logger.info(f"Successfully downloaded and loaded model: {model_name}")
            except Exception as e:
                # If download fails, try to install scispacy and models
                logger.warning(f"Could not download {model_name}. Attempting to install scispacy...")
                os.system("pip install scispacy")
                
                if model_name == "en_core_sci_sm":
                    os.system("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")
                elif model_name == "en_ner_bionlp13cg_md":
                    os.system("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz")
                
                self.nlp = spacy.load(model_name)
                logger.info(f"Successfully installed and loaded model: {model_name}")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from a single text document.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of entity dictionaries, each containing:
                - text: The entity text
                - label: The entity type/label
                - start: Start character position
                - end: End character position
                - sent_idx: Index of the sentence containing the entity
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided to extract_entities")
            return []
        
        try:
            doc = self.nlp(text)
            
            # Extract entities with sentence context
            entities = []
            for sent_idx, sent in enumerate(doc.sents):
                for ent in sent.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char - sent.start_char,  # Relative to sentence
                        "end": ent.end_char - sent.start_char,  # Relative to sentence
                        "sent_idx": sent_idx,
                        "sentence": sent.text
                    }
                    entities.append(entity_info)
            
            return entities
        
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def process_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document to extract entities from its sections.
        
        Args:
            doc_data: Document data containing sections with text
            
        Returns:
            Document with added entity information for each section
        """
        if not isinstance(doc_data, dict):
            logger.warning(f"Invalid document data format: {type(doc_data)}")
            return {}
        
        # Create a copy to avoid modifying the original
        result = doc_data.copy()
        
        # Extract text field
        text = doc_data.get("text", "")
        section_type = doc_data.get("section_type", "unknown")
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Add to result
        result["entities"] = entities
        result["entity_count"] = len(entities)
        
        return result
    
    def process_dataset(self, 
                        input_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None,
                        max_docs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a dataset of documents to extract entities.
        
        Args:
            input_path: Path to the input JSON file containing documents
            output_path: Optional path to save the processed documents
            max_docs: Maximum number of documents to process (for testing)
            
        Returns:
            List of processed documents with extracted entities
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return []
        
        try:
            # Load documents
            with open(input_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            
            if max_docs:
                documents = documents[:max_docs]
                
            logger.info(f"Processing {len(documents)} documents for entity extraction")
            
            # Process each document
            processed_docs = []
            for doc in tqdm(documents, desc="Extracting entities"):
                processed_doc = self.process_document(doc)
                processed_docs.append(processed_doc)
            
            # Save to output file if specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_docs, f, indent=2)
                
                logger.info(f"Saved processed documents with entities to {output_path}")
            
            return processed_docs
        
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    extractor = ScientificEntityExtractor()
    
    # Define paths
    input_file = PROJECT_ROOT / "data/transfer_learning/prepared/section_classification/train/data.json"
    output_file = PROJECT_ROOT / "data/derived/phase2.2_output/entities_output.json"
    
    # Process a small subset for testing
    processed_docs = extractor.process_dataset(
        input_path=input_file,
        output_path=output_file,
        max_docs=10  # Process only 10 docs for quick testing
    )
    
    # Print statistics
    entity_counts = [doc.get("entity_count", 0) for doc in processed_docs]
    total_entities = sum(entity_counts)
    
    print(f"Processed {len(processed_docs)} documents")
    print(f"Extracted {total_entities} entities in total")
    print(f"Average {total_entities / len(processed_docs):.1f} entities per document") 