"""
Relation Extraction module for Scientific NLP Pipeline (Phase 2.2)

This module implements simple rule-based relation extraction from scientific text
using SpaCy's dependency parsing capabilities.
"""

import json
import logging
import spacy
import re
import os
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
        logging.FileHandler(logs_dir / "relation_extraction.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("relation_extraction")

class PatternRelationExtractor:
    """Extract relations from text using dependency parsing and patterns."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the relation extractor with a SpaCy model.
        
        Args:
            model_name: Name of the SpaCy model to use.
        """
        logger.info(f"Initializing PatternRelationExtractor with model: {model_name}")
        
        try:
            # Load SpaCy model
            self.nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except OSError as e:
            # Try to download the model if not found
            logger.warning(f"Model {model_name} not found. Attempting to download...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            logger.info(f"Successfully downloaded and loaded model: {model_name}")
        
        # Define relation patterns
        self.relation_patterns = [
            # "uses [method]" pattern
            {
                "name": "uses_method",
                "description": "Identifies methods or techniques used in research",
                "verbs": ["use", "utilize", "employ", "apply", "implement", "leverage", "adopt"],
                "dependencies": ["dobj", "pobj"]
            },
            # "measured [quantity]" pattern
            {
                "name": "measured_quantity",
                "description": "Identifies quantities or metrics that were measured",
                "verbs": ["measure", "quantify", "calculate", "evaluate", "assess", "estimate", "determine", "compute"],
                "dependencies": ["dobj", "pobj"]
            },
            # "caused [effect]" pattern
            {
                "name": "caused_effect",
                "description": "Identifies cause-effect relationships",
                "verbs": ["cause", "lead", "result", "trigger", "induce", "generate", "produce", "affect"],
                "dependencies": ["dobj", "pobj", "prep"]
            },
            # "compared [thing1] and [thing2]" pattern
            {
                "name": "compared_things",
                "description": "Identifies comparisons between entities",
                "verbs": ["compare", "contrast", "differentiate", "distinguish", "evaluate"],
                "dependencies": ["dobj", "pobj", "conj"]
            }
        ]
    
    def _extract_relation_arguments(self, verb_token, pattern):
        """
        Extract arguments for a relation based on dependency patterns.
        
        Args:
            verb_token: The verb token that triggers the relation
            pattern: The relation pattern to match
            
        Returns:
            List of argument dictionaries
        """
        arguments = []
        
        # Look for direct objects and objects of prepositions
        for child in verb_token.children:
            # Check if this is a relevant dependency
            if child.dep_ in pattern["dependencies"]:
                # For direct objects
                if child.dep_ == "dobj":
                    # Get the full noun phrase
                    span = self._get_span_for_token(child)
                    arguments.append({
                        "role": "object",
                        "text": span.text,
                        "root": child.text
                    })
                
                # For prepositional objects, follow the prep -> pobj path
                elif child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            span = self._get_span_for_token(grandchild)
                            arguments.append({
                                "role": f"prep_{child.text}",  # e.g., "prep_with"
                                "text": span.text,
                                "root": grandchild.text
                            })
                
                # For conjunctions (in comparison patterns)
                elif child.dep_ == "conj":
                    span = self._get_span_for_token(child)
                    arguments.append({
                        "role": "conj",
                        "text": span.text,
                        "root": child.text
                    })
        
        # If the verb has a subject, add it
        subjects = []
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                span = self._get_span_for_token(child)
                subjects.append({
                    "role": "subject",
                    "text": span.text,
                    "root": child.text
                })
        
        # Return all arguments including subjects
        return subjects + arguments
    
    def _get_span_for_token(self, token):
        """
        Get the noun phrase span for a token, if it exists.
        
        Args:
            token: The token to find a span for
            
        Returns:
            The span containing the token
        """
        # First check if token is part of a noun chunk
        doc = token.doc
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk
        
        # Otherwise just return the token itself as a span
        return doc[token.i:token.i+1]
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relations from a text based on predefined patterns.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of relation dictionaries
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided to extract_relations")
            return []
        
        try:
            # Parse the text
            doc = self.nlp(text)
            
            # Store found relations
            relations = []
            
            # Process each sentence
            for sent_idx, sent in enumerate(doc.sents):
                # Check each token for potential relation triggers
                for token in sent:
                    # Check if token is a verb
                    if token.pos_ == "VERB":
                        # Check against each relation pattern
                        for pattern in self.relation_patterns:
                            # Check if the lemma matches one of our target verbs
                            if token.lemma_ in pattern["verbs"]:
                                # Extract the arguments
                                args = self._extract_relation_arguments(token, pattern)
                                
                                # Only add relations that have at least one argument
                                if args:
                                    relation = {
                                        "type": pattern["name"],
                                        "verb": token.text,
                                        "verb_lemma": token.lemma_,
                                        "arguments": args,
                                        "sent_idx": sent_idx,
                                        "sentence": sent.text,
                                        "confidence": 0.7  # Default confidence score
                                    }
                                    relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting relations: {str(e)}")
            return []
    
    def process_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document to extract relations from its text.
        
        Args:
            doc_data: Document data containing text
            
        Returns:
            Document with added relation information
        """
        if not isinstance(doc_data, dict):
            logger.warning(f"Invalid document data format: {type(doc_data)}")
            return {}
        
        # Create a copy to avoid modifying the original
        result = doc_data.copy()
        
        # Extract text field
        text = doc_data.get("text", "")
        
        # Extract relations
        relations = self.extract_relations(text)
        
        # Add to result
        result["relations"] = relations
        result["relation_count"] = len(relations)
        
        return result
    
    def process_dataset(self, 
                        input_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None,
                        max_docs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a dataset of documents to extract relations.
        
        Args:
            input_path: Path to the input JSON file containing documents
            output_path: Optional path to save the processed documents
            max_docs: Maximum number of documents to process (for testing)
            
        Returns:
            List of processed documents with extracted relations
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
                
            logger.info(f"Processing {len(documents)} documents for relation extraction")
            
            # Process each document
            processed_docs = []
            for doc in tqdm(documents, desc="Extracting relations"):
                processed_doc = self.process_document(doc)
                processed_docs.append(processed_doc)
            
            # Save to output file if specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_docs, f, indent=2)
                
                logger.info(f"Saved processed documents with relations to {output_path}")
            
            return processed_docs
        
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    extractor = PatternRelationExtractor()
    
    # Define paths
    input_file = PROJECT_ROOT / "data/transfer_learning/prepared/section_classification/train/data.json"
    output_file = PROJECT_ROOT / "data/derived/phase2.2_output/relations_output.json"
    
    # Process a small subset for testing
    processed_docs = extractor.process_dataset(
        input_path=input_file,
        output_path=output_file,
        max_docs=10  # Process only 10 docs for quick testing
    )
    
    # Print statistics
    relation_counts = [doc.get("relation_count", 0) for doc in processed_docs]
    total_relations = sum(relation_counts)
    
    print(f"Processed {len(processed_docs)} documents")
    print(f"Extracted {total_relations} relations in total")
    print(f"Average {total_relations / len(processed_docs):.1f} relations per document") 