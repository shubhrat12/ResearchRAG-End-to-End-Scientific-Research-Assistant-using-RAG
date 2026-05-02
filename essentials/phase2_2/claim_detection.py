"""
Claim Detection module for Scientific NLP Pipeline (Phase 2.2)

This module implements rule-based and pattern-based approaches to identify
scientific claims in research papers, focusing on results/discussion sections.
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
        logging.FileHandler(logs_dir / "claim_detection.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("claim_detection")

class ScientificClaimDetector:
    """Detect scientific claims in research papers using heuristics and patterns."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the claim detector with a SpaCy model.
        
        Args:
            model_name: Name of the SpaCy model to use.
        """
        logger.info(f"Initializing ScientificClaimDetector with model: {model_name}")
        
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
        
        # Define claim signal patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize patterns for claim detection."""
        # Claim signal verbs (modal, reporting, etc.)
        self.claim_verbs = [
            "suggest", "demonstrate", "show", "indicate", "reveal", "confirm",
            "establish", "prove", "conclude", "argue", "report", "find", "observe",
            "discover", "determine", "imply", "highlight", "provide", "identify",
            "illustrate", "support", "validate"
        ]
        
        # Evidence signal patterns
        self.evidence_patterns = [
            r"\(\s*[A-Za-z]+\s*et\s*al\.,?\s*\d{4}\s*\)",  # (Author et al., 2020)
            r"\(\s*[A-Za-z]+\s*and\s*[A-Za-z]+,?\s*\d{4}\s*\)",  # (Author and Author, 2020)
            r"\[\s*\d+\s*\]",  # [1] or [23]
            r"according\s+to",  # according to
            r"based\s+on",  # based on
            r"consistent\s+with",  # consistent with
            r"in\s+line\s+with",  # in line with
            r"as\s+demonstrated\s+by",  # as demonstrated by
            r"as\s+shown\s+by",  # as shown by
            r"as\s+reported\s+by"  # as reported by
        ]
        
        # Compile regex patterns
        self.evidence_regex = re.compile("|".join(self.evidence_patterns), re.IGNORECASE)
        
        # Hedging/uncertainty phrases
        self.hedging_phrases = [
            "may", "might", "could", "can", "possibly", "potentially", "likely",
            "suggest", "indicate", "appear", "seem", "tend to", "generally",
            "often", "sometimes", "frequently", "rarely", "occasionally", "usually",
            "typically", "perhaps", "presumably", "approximately", "about", "around",
            "estimate", "hypothesize", "speculate", "assume", "probable", "possible"
        ]
        
        # Words indicating findings/results
        self.finding_words = [
            "result", "finding", "outcome", "observation", "data", "evidence", 
            "analysis", "experiment", "study", "investigation", "evaluation"
        ]
        
        # Claim confidence boosters
        self.confidence_boosters = [
            "clearly", "definitely", "strongly", "significantly", "substantially",
            "markedly", "notably", "dramatically", "considerably", "undoubtedly",
            "unequivocally", "certainly", "evidently", "indeed"
        ]
    
    def _contains_hedge_words(self, text):
        """
        Check if text contains hedging words.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if hedging is present, and the hedge words found
        """
        found_hedges = []
        for hedge in self.hedging_phrases:
            # Look for the hedge phrase as a whole word
            pattern = r'\b' + re.escape(hedge) + r'\b'
            if re.search(pattern, text.lower()):
                found_hedges.append(hedge)
        
        return len(found_hedges) > 0, found_hedges
    
    def _has_citation_evidence(self, text):
        """
        Check if text contains citation evidence.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if citations are present
        """
        return bool(self.evidence_regex.search(text))
    
    def _get_confidence_score(self, contains_hedges, has_citation, contains_boosters, claim_verb_strength):
        """
        Calculate confidence score for a claim.
        
        Args:
            contains_hedges: Whether the text contains hedging language
            has_citation: Whether the text has citations
            contains_boosters: Whether the text contains confidence boosters
            claim_verb_strength: Strength of the claim verb (0-1)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base score
        score = 0.5
        
        # Adjust for hedging (decreases confidence)
        if contains_hedges:
            score -= 0.1
        
        # Adjust for citations (increases confidence)
        if has_citation:
            score += 0.15
        
        # Adjust for boosters (increases confidence)
        if contains_boosters:
            score += 0.1
        
        # Adjust for verb strength
        score += claim_verb_strength * 0.25
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _contains_confidence_boosters(self, text):
        """
        Check if text contains confidence boosting words.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if boosters are present, and the boosters found
        """
        found_boosters = []
        for booster in self.confidence_boosters:
            pattern = r'\b' + re.escape(booster) + r'\b'
            if re.search(pattern, text.lower()):
                found_boosters.append(booster)
        
        return len(found_boosters) > 0, found_boosters
    
    def _get_claim_verb_strength(self, verb):
        """
        Get a strength score for a claim verb.
        
        Args:
            verb: The verb to evaluate
            
        Returns:
            Strength score between 0 and 1
        """
        # Strong verbs (more definitive)
        strong_verbs = ["demonstrate", "prove", "confirm", "establish", "show"]
        
        # Medium verbs
        medium_verbs = ["indicate", "reveal", "find", "observe", "discover", "determine"]
        
        # Weaker verbs (more tentative)
        weak_verbs = ["suggest", "imply", "may", "might", "could"]
        
        if verb in strong_verbs:
            return 1.0
        elif verb in medium_verbs:
            return 0.7
        elif verb in weak_verbs:
            return 0.4
        else:
            return 0.5  # Default for other verbs
    
    def extract_claims(self, text: str, section_type: str = None) -> List[Dict[str, Any]]:
        """
        Extract claims from a text.
        
        Args:
            text: The text to analyze
            section_type: The type of section (e.g., "results", "discussion")
            
        Returns:
            List of claim dictionaries
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided to extract_claims")
            return []
        
        # Prioritize results, discussion and conclusion sections
        priority_sections = ["results", "discussion", "conclusion"]
        section_priority = 1.0
        if section_type and section_type.lower() in priority_sections:
            section_priority = 1.2  # Boost confidence for claims in these sections
        
        try:
            # Parse the text
            doc = self.nlp(text)
            
            # Store found claims
            claims = []
            
            # Process each sentence
            for sent_idx, sent in enumerate(doc.sents):
                sent_text = sent.text.strip()
                
                # Skip very short sentences
                if len(sent_text.split()) < 4:
                    continue
                
                # Flag for whether this sentence contains a claim
                is_claim = False
                claim_verb = None
                claim_evidence = []
                
                # Look for claim signal verbs
                for token in sent:
                    if token.lemma_ in self.claim_verbs:
                        is_claim = True
                        claim_verb = token.text
                        break
                
                # Look for phrases indicating findings/results
                contains_finding = False
                for word in self.finding_words:
                    if re.search(r'\b' + re.escape(word) + r'\b', sent_text.lower()):
                        contains_finding = True
                        break
                
                # Look for citations or other evidence
                has_citation = self._has_citation_evidence(sent_text)
                if has_citation:
                    claim_evidence.append("citation")
                
                # Check for hedging/uncertainty language
                has_hedging, hedge_words = self._contains_hedge_words(sent_text)
                
                # Check for confidence boosters
                has_boosters, booster_words = self._contains_confidence_boosters(sent_text)
                
                # Calculate confidence score if it looks like a claim
                if is_claim or (contains_finding and (has_citation or has_boosters)):
                    # Calculate confidence score
                    verb_strength = self._get_claim_verb_strength(claim_verb) if claim_verb else 0.5
                    raw_score = self._get_confidence_score(
                        has_hedging, 
                        has_citation, 
                        has_boosters,
                        verb_strength
                    )
                    confidence = max(0.0, min(1.0, raw_score * section_priority))  # Clamp after multiplying
                    logger.debug(f"Sentence: {sent_text}")
                    logger.debug(f"Verb strength: {verb_strength}, Has hedging: {has_hedging}, Has citation: {has_citation}, Has boosters: {has_boosters}")
                    logger.debug(f"Calculated confidence: {confidence}")
                    
                    # Create claim object
                    claim = {
                        "text": sent_text,
                        "confidence_score": confidence,  # Ensure non-None confidence_score
                        "sent_idx": sent_idx,
                        "has_citation": has_citation,
                        "has_hedging": has_hedging,
                        "hedge_words": hedge_words if has_hedging else [],
                        "has_boosters": has_boosters,
                        "booster_words": booster_words if has_boosters else [],
                        "claim_verb": claim_verb,
                        "evidence": claim_evidence
                    }
                    
                    claims.append(claim)
            
            # Sort claims by confidence (highest first)
            claims.sort(key=lambda x: x["confidence_score"], reverse=True)
            
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            return []
    
    def process_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document to extract claims from its text.
        
        Args:
            doc_data: Document data containing text
            
        Returns:
            Document with added claim information
        """
        if not isinstance(doc_data, dict):
            logger.warning(f"Invalid document data format: {type(doc_data)}")
            return {}
        
        # Create a copy to avoid modifying the original
        result = doc_data.copy()
        
        # Extract text field and section type
        text = doc_data.get("text", "")
        section_type = doc_data.get("section_type", "")
        
        # Extract claims
        claims = self.extract_claims(text, section_type)
        
        # Add to result
        result["claims"] = claims
        result["claim_count"] = len(claims)
        
        return result
    
    def process_dataset(self, 
                        input_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None,
                        max_docs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a dataset of documents to extract claims.
        
        Args:
            input_path: Path to the input JSON file containing documents
            output_path: Optional path to save the processed documents
            max_docs: Maximum number of documents to process (for testing)
            
        Returns:
            List of processed documents with extracted claims
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
                
            logger.info(f"Processing {len(documents)} documents for claim detection")
            
            # Process each document
            processed_docs = []
            for doc in tqdm(documents, desc="Extracting claims"):
                processed_doc = self.process_document(doc)
                processed_docs.append(processed_doc)
            
            # Save to output file if specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_docs, f, indent=2)
                
                logger.info(f"Saved processed documents with claims to {output_path}")
            
            return processed_docs
        
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    detector = ScientificClaimDetector()
    
    # Define paths
    input_file = "data/transfer_learning/prepared/section_classification/train/data.json"
    output_file = "data/derived/phase2.2_output/claims_output.json"
    
    # Process a small subset for testing
    processed_docs = detector.process_dataset(
        input_path=input_file,
        output_path=output_file,
        max_docs=10  # Process only 10 docs for quick testing
    )
    
    # Print statistics
    claim_counts = [doc.get("claim_count", 0) for doc in processed_docs]
    total_claims = sum(claim_counts)
    
    print(f"Processed {len(processed_docs)} documents")
    print(f"Extracted {total_claims} claims in total")
    print(f"Average {total_claims / len(processed_docs):.1f} claims per document") 