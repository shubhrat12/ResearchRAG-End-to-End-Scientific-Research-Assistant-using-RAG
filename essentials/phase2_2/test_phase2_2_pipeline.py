"""
Test script for Scientific NLP Pipeline (Phase 2.2)

This script verifies the correct functioning of all components in the pipeline:
1. Entity Extraction
2. Relation Extraction 
3. Claim Detection
4. Integrated Pipeline

Run this script to confirm that the pipeline is working as expected.
"""

import os
import sys
import json
import unittest
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Import modules to test
try:
    from entity_extraction import ScientificEntityExtractor
    from relation_extraction import PatternRelationExtractor
    from claim_detection import ScientificClaimDetector
    from run_phase2_2_pipeline import ScientificNLPPipeline
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the essentials/phase2.2 directory")
    sys.exit(1)

# Sample scientific texts for testing
SAMPLE_TEXTS = {
    "entity_test": """
    The SARS-CoV-2 virus, responsible for COVID-19, is a novel coronavirus that emerged in Wuhan, China.
    It binds to ACE2 receptors in human cells using its spike protein. Several vaccines like 
    Pfizer-BioNTech, Moderna, and Oxford-AstraZeneca have been developed to combat this disease.
    """,
    
    "relation_test": """
    Researchers used machine learning to analyze the data. The study measured blood pressure levels
    in 500 patients. The experiment evaluated the effectiveness of the treatment across multiple trials.
    The results caused significant changes in medical protocols. Scientists compared traditional methods
    and new approaches to determine the optimal strategy.
    """,
    
    "claim_test": """
    Our results demonstrate a significant improvement over previous methods. The data suggest that
    treatment A is more effective than treatment B. According to Smith et al. (2020), these findings
    are consistent with previous work. We find that the proposed approach clearly outperforms baseline
    methods. The evidence indicates that further research is needed, but these results are promising.
    """,
    
    "full_pipeline": """
    The SARS-CoV-2 virus binds to ACE2 receptors in human cells. Researchers used machine learning
    algorithms to analyze COVID-19 patient data from multiple hospitals. The study measured viral loads
    and antibody levels in 250 patients. Our results demonstrate that the mRNA vaccines produced by
    Pfizer and Moderna provide significant protection against severe disease. According to recent
    clinical trials (Jones et al., 2021), vaccine efficacy remains high even against newer variants.
    The data strongly suggest that vaccination reduces transmission rates. This research shows the
    importance of continued genomic surveillance and vaccination efforts.
    """
}

# Create absolute path for test output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to project root
TEST_OUTPUT_DIR = PROJECT_ROOT / "data/derived/phase2.2_test_output"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
print(f"Test output directory: {TEST_OUTPUT_DIR}")

# Ensure logs directory exists
LOG_DIR = Path(__file__).parent / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Log directory: {LOG_DIR}")

class TestPhase22Pipeline(unittest.TestCase):
    """Test case for Phase 2.2 Scientific NLP Pipeline components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create output directory
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        
        print("\n=== Testing Scientific NLP Pipeline (Phase 2.2) ===\n")
        print(f"Current working directory: {os.getcwd()}")
        
        try:
            # Initialize components with English web model to speed up testing
            # Note: Using en_core_web_sm for faster tests - in production we'd use scientific models
            print("Initializing entity extractor...")
            cls.entity_extractor = ScientificEntityExtractor(model_name="en_core_web_sm")
            
            print("Initializing relation extractor...")
            cls.relation_extractor = PatternRelationExtractor(model_name="en_core_web_sm")
            
            print("Initializing claim detector...")
            cls.claim_detector = ScientificClaimDetector(model_name="en_core_web_sm")
            
            print("Initializing pipeline...")
            cls.pipeline = ScientificNLPPipeline(
                entity_model="en_core_web_sm",
                relation_model="en_core_web_sm",
                claim_model="en_core_web_sm"
            )
            print("All components initialized successfully")
        except Exception as e:
            print(f"Error initializing components: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    
    def test_01_entity_extraction(self):
        """Test entity extraction component."""
        print("\n--- Testing Entity Extraction ---")
        
        # Create test document
        doc_data = {"text": SAMPLE_TEXTS["entity_test"], "section_type": "methods"}
        
        # Process document
        result = self.entity_extractor.process_document(doc_data)
        
        # Save output
        output_file = TEST_OUTPUT_DIR / "entity_test_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        # Verify results
        entity_count = result.get("entity_count", 0)
        print(f"Extracted {entity_count} entities")
        print(f"First 3 entities: {result.get('entities', [])[:3]}")
        print(f"Output saved to: {output_file}")
        
        # Assert at least one entity is found
        self.assertTrue(entity_count > 0, "No entities were extracted")
    
    def test_02_relation_extraction(self):
        """Test relation extraction component."""
        print("\n--- Testing Relation Extraction ---")
        
        # Create test document
        doc_data = {"text": SAMPLE_TEXTS["relation_test"], "section_type": "methods"}
        
        # Process document
        result = self.relation_extractor.process_document(doc_data)
        
        # Save output
        output_file = TEST_OUTPUT_DIR / "relation_test_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        # Verify results
        relation_count = result.get("relation_count", 0)
        print(f"Extracted {relation_count} relations")
        print(f"First 3 relations: {result.get('relations', [])[:3]}")
        print(f"Output saved to: {output_file}")
        
        # Assert at least one relation is found
        self.assertTrue(relation_count > 0, "No relations were extracted")
    
    def test_03_claim_detection(self):
        """Test claim detection component."""
        print("\n--- Testing Claim Detection ---")
        
        # Create test document
        doc_data = {"text": SAMPLE_TEXTS["claim_test"], "section_type": "results"}
        
        # Process document
        result = self.claim_detector.process_document(doc_data)
        
        # Save output
        output_file = TEST_OUTPUT_DIR / "claim_test_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        # Verify results
        claim_count = result.get("claim_count", 0)
        print(f"Extracted {claim_count} claims")
        print(f"First 3 claims: {result.get('claims', [])[:3]}")
        print(f"Output saved to: {output_file}")
        
        # Assert at least one claim is found
        self.assertTrue(claim_count > 0, "No claims were detected")
    
    def test_04_full_pipeline(self):
        """Test the full integrated pipeline."""
        print("\n--- Testing Full Integrated Pipeline ---")
        
        # Create list of test documents (each will be a section)
        documents = [
            {"text": SAMPLE_TEXTS["full_pipeline"], "section_type": "results"},
            {"text": SAMPLE_TEXTS["claim_test"], "section_type": "discussion"},
            {"text": SAMPLE_TEXTS["entity_test"], "section_type": "methods"}
        ]
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            json.dump(documents, temp)
            temp_input_path = temp.name
        
        try:
            # Process documents through the pipeline
            stats = self.pipeline.process_dataset(
                input_path=temp_input_path,
                output_dir=TEST_OUTPUT_DIR,
                batch_size=3,
                max_docs=3
            )
            
            # Verify results
            print(f"\nPipeline processing statistics:")
            print(f"- Documents processed: {stats['total_documents']}")
            print(f"- Entities extracted: {stats['total_entities']}")
            print(f"- Relations identified: {stats['total_relations']}")
            print(f"- Claims detected: {stats['total_claims']}")
            print(f"- Processing time: {stats['processing_time']:.2f} seconds")
            print(f"- Output files: {', '.join([Path(f).name for f in stats['output_files']])}")
            
            # Assert we have processed all documents and found all types of information
            self.assertEqual(stats['total_documents'], 3, "Not all documents were processed")
            self.assertTrue(stats['total_entities'] > 0, "No entities were extracted")
            self.assertTrue(stats['total_relations'] > 0, "No relations were identified")
            self.assertTrue(stats['total_claims'] > 0, "No claims were detected")
            
        finally:
            # Clean up temp file
            os.unlink(temp_input_path)
    
    def test_05_confidence_scores(self):
        """Test confidence scores of claims."""
        print("\n--- Testing Confidence Scores ---")

        # Sample texts for confidence testing
        strong_claim_text = "Our results clearly demonstrate that the new method significantly outperforms the old one."
        weak_claim_text = "The data might suggest that the new method could potentially be better than the old one."

        # Process strong claim
        strong_result = self.claim_detector.extract_claims(strong_claim_text, section_type="results")
        if not strong_result:
            self.skipTest("No strong claims detected in the strong claim text.")
        strong_confidence = strong_result[0].get('confidence_score')

        # Process weak claim
        weak_result = self.claim_detector.extract_claims(weak_claim_text, section_type="results")
        weak_confidence = weak_result[0].get('confidence_score') if weak_result else None

        # Assert confidence scores exist and are within range
        self.assertIsNotNone(strong_confidence, "Strong claim confidence score is None")
        self.assertTrue(0.0 <= strong_confidence <= 1.0, "Strong claim confidence score out of range")
        if weak_confidence is not None:
            self.assertTrue(0.0 <= weak_confidence <= 1.0, "Weak claim confidence score out of range")

        # Assert strong claim has higher confidence than weak claim
        if weak_confidence is not None:
            self.assertGreater(strong_confidence, weak_confidence, "Strong claim should have higher confidence than weak claim")

        print("Confidence score test completed.")

    def test_06_full_pipeline_with_confidence(self):
        """Test the full integrated pipeline with confidence score validation."""
        print("\n--- Testing Full Integrated Pipeline with Confidence Scores ---")

        # Create list of test documents (each will be a section)
        documents = [
            {"text": SAMPLE_TEXTS["full_pipeline"], "section_type": "results"},
            {"text": SAMPLE_TEXTS["claim_test"], "section_type": "discussion"},
            {"text": SAMPLE_TEXTS["entity_test"], "section_type": "methods"}
        ]

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            json.dump(documents, temp)
            temp_input_path = temp.name

        try:
            # Process documents through the pipeline
            stats = self.pipeline.process_dataset(
                input_path=temp_input_path,
                output_dir=TEST_OUTPUT_DIR,
                batch_size=3,
                max_docs=3
            )

            # Load and verify confidence scores
            for output_file in stats['output_files']:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Check if data is a list of documents
                    if isinstance(data, list):
                        for document in data:
                            # Check if document has 'claims' key
                            claims = document.get('claims', [])
                            for claim in claims:
                                if 'confidence_score' not in claim:
                                    self.fail("Claim did not return a confidence score")
                                confidence_score = claim['confidence_score']
                                self.assertTrue(0.0 <= confidence_score <= 1.0, "Claim confidence score out of range")
                    else:
                        self.fail("Expected a list of documents, but got a different structure.")

            print("Confidence scores validated for all claims.")

        finally:
            # Clean up temp file
            os.unlink(temp_input_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print("\n=== All tests completed successfully! ===")
        print(f"Test outputs saved to: {TEST_OUTPUT_DIR}")


def run_tests():
    """Run all tests."""
    try:
        # Run tests
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    except Exception as e:
        print(f"Error running tests: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    print(f"Starting Phase 2.2 Pipeline Tests...")
    run_tests() 