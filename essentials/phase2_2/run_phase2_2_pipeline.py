"""
Scientific NLP Pipeline (Phase 2.2)

This script combines entity extraction, relation extraction, and claim detection
into a unified pipeline for processing scientific documents.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import our components
from entity_extraction import ScientificEntityExtractor
from relation_extraction import PatternRelationExtractor
from claim_detection import ScientificClaimDetector

# Create logs directory
logs_dir = Path(__file__).parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "phase2_2_pipeline.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("phase2_2_pipeline")

# Define PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class ScientificNLPPipeline:
    """Main pipeline to process scientific documents with all three components."""
    
    def __init__(self, 
                entity_model: str = "en_core_sci_sm",
                relation_model: str = "en_core_web_sm",
                claim_model: str = "en_core_web_sm"):
        """
        Initialize the pipeline with models for each component.
        
        Args:
            entity_model: Model name for entity extraction
            relation_model: Model name for relation extraction
            claim_model: Model name for claim detection
        """
        logger.info("Initializing Scientific NLP Pipeline")
        
        # Initialize components
        logger.info("Loading entity extractor...")
        self.entity_extractor = ScientificEntityExtractor(model_name=entity_model)
        
        logger.info("Loading relation extractor...")
        self.relation_extractor = PatternRelationExtractor(model_name=relation_model)
        
        logger.info("Loading claim detector...")
        self.claim_detector = ScientificClaimDetector(model_name=claim_model)
        
        logger.info("Pipeline initialized successfully")
    
    def process_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document through all pipeline components.
        
        Args:
            doc_data: Document data dictionary
            
        Returns:
            Processed document with all extracted information
        """
        if not isinstance(doc_data, dict):
            logger.warning(f"Invalid document data format: {type(doc_data)}")
            return {}
        
        # Process in sequence: entities -> relations -> claims
        try:
            # Step 1: Entity extraction
            doc_with_entities = self.entity_extractor.process_document(doc_data)
            
            # Step 2: Relation extraction
            doc_with_relations = self.relation_extractor.process_document(doc_with_entities)
            
            # Step 3: Claim detection
            result = self.claim_detector.process_document(doc_with_relations)
            
            # Add metadata about processing
            result["phase2_2_processed"] = True
            result["phase2_2_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return doc_data  # Return original on error
    
    def process_dataset(self, 
                       input_path: Path, 
                       output_dir: Path,
                       batch_size: int = 50,
                       max_docs: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a dataset of documents through the pipeline.
        
        Args:
            input_path: Path to input JSON file
            output_dir: Directory to save output files
            batch_size: Number of documents to process in each batch
            max_docs: Maximum number of documents to process (for testing)
            
        Returns:
            Statistics about the processing
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return {"error": "Input file not found"}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load documents
            logger.info(f"Loading documents from {input_path}")
            with open(input_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            
            # Limit documents if specified
            if max_docs and max_docs > 0:
                documents = documents[:max_docs]
                logger.info(f"Limited to processing {max_docs} documents")
            
            total_docs = len(documents)
            logger.info(f"Processing {total_docs} documents with batch size {batch_size}")
            
            # Process in batches
            batches = [documents[i:i+batch_size] for i in range(0, total_docs, batch_size)]
            
            # Stats tracking
            stats = {
                "total_documents": total_docs,
                "total_entities": 0,
                "total_relations": 0,
                "total_claims": 0,
                "processing_time": 0,
                "output_files": []
            }
            
            start_time = time.time()
            
            # Process each batch
            for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
                batch_results = []
                
                # Process each document in the batch
                for doc in tqdm(batch, desc=f"Batch {batch_idx+1}/{len(batches)}", leave=False):
                    processed_doc = self.process_document(doc)
                    batch_results.append(processed_doc)
                    
                    # Update stats
                    stats["total_entities"] += processed_doc.get("entity_count", 0)
                    stats["total_relations"] += processed_doc.get("relation_count", 0)
                    stats["total_claims"] += processed_doc.get("claim_count", 0)
                
                # Save batch results
                batch_output_file = output_dir / f"batch_{batch_idx+1}.json"
                with open(batch_output_file, "w", encoding="utf-8") as f:
                    json.dump(batch_results, f, indent=2)
                
                stats["output_files"].append(str(batch_output_file))
                
                logger.info(f"Saved batch {batch_idx+1} to {batch_output_file}")
            
            # Save all results in one combined file
            all_output_file = output_dir / "all_processed_docs.json"
            
            # Instead of loading all documents back into memory, process the batches
            # one by one and write to a single file
            with open(all_output_file, "w", encoding="utf-8") as f:
                f.write("[\n")  # Start JSON array
                
                for i, batch_file in enumerate(stats["output_files"]):
                    with open(batch_file, "r", encoding="utf-8") as batch_f:
                        batch_content = batch_f.read()
                        # Remove the opening and closing brackets for all but the first and last batches
                        if i == 0:
                            batch_content = batch_content[1:]  # Remove opening [
                        if i == len(stats["output_files"]) - 1:
                            batch_content = batch_content[:-1]  # Remove closing ]
                        else:
                            batch_content = batch_content[1:-1] + ","  # Remove [] and add comma
                        
                        f.write(batch_content)
                
                f.write("\n]")  # End JSON array
            
            stats["output_files"].append(str(all_output_file))
            logger.info(f"Saved combined results to {all_output_file}")
            
            # Complete stats
            stats["processing_time"] = time.time() - start_time
            
            # Save stats
            stats_file = output_dir / "processing_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Processing completed in {stats['processing_time']:.2f} seconds")
            logger.info(f"Extracted {stats['total_entities']} entities, "
                      f"{stats['total_relations']} relations, "
                      f"and {stats['total_claims']} claims")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Scientific NLP Pipeline (Phase 2.2)")
    
    parser.add_argument("--input", type=str, 
                       default="data/transfer_learning/prepared/section_classification/train/data.json",
                       help="Path to input JSON file")
    
    parser.add_argument("--output-dir", type=str,
                       default="data/derived/phase2.2_output",
                       help="Directory to save output files")
    
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Number of documents to process in each batch")
    
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Maximum number of documents to process (for testing)")
    
    parser.add_argument("--entity-model", type=str, default="en_core_sci_sm",
                       help="Model to use for entity extraction")
    
    parser.add_argument("--relation-model", type=str, default="en_core_web_sm",
                       help="Model to use for relation extraction")
    
    parser.add_argument("--claim-model", type=str, default="en_core_web_sm",
                       help="Model to use for claim detection")
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Parse arguments
    args = parse_args()
    
    # Initialize pipeline
    pipeline = ScientificNLPPipeline(
        entity_model=args.entity_model,
        relation_model=args.relation_model,
        claim_model=args.claim_model
    )
    
    # Process dataset
    stats = pipeline.process_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_docs=args.max_docs
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Scientific NLP Pipeline (Phase 2.2) Results")
    print("="*50)
    print(f"Processed {stats['total_documents']} documents")
    print(f"Extracted {stats['total_entities']} entities")
    print(f"Extracted {stats['total_relations']} relations")
    print(f"Extracted {stats['total_claims']} claims")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main() 