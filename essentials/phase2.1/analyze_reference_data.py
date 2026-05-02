"""
Analyze reference parsing data to identify class distribution issues.

This script will analyze training data for reference parsing model, showing:
1. How many examples have at least one entity labeled
2. Percentage distribution of each label after conversion
3. Success rate of entity matching for each entity type
"""

import json
import os
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/reference_data_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reference_data_analysis")

# Entity types
ENTITY_TYPES = ["AUTHOR", "TITLE", "YEAR", "VENUE"]

def normalize_text(text):
    """Normalize text by removing punctuation, extra spaces, and converting to lowercase."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_similarity(s1, s2):
    """Get similarity score between two strings."""
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    if not s1 or not s2:
        return 0
    return SequenceMatcher(None, s1, s2).ratio()

def analyze_entity_matching(example):
    """Analyze how well entities in structured data match the text."""
    text = example.get('text', '')
    structured = example.get('structured', {})
    
    # Simple tokenization by whitespace
    tokens = text.split()
    
    # Default to 'O' labels for all tokens
    labels = ['O'] * len(tokens)
    
    # Extract structured information
    authors = structured.get('authors', [])
    title = structured.get('title', '')
    year = str(structured.get('year', ''))
    venue = structured.get('venue', '')
    
    # Clean tokens
    tokens_lower = [token.lower() for token in tokens]
    translator = str.maketrans('', '', string.punctuation)
    tokens_clean = [token.lower().translate(translator) for token in tokens]
    
    # Track which entity types could be found in text
    entity_matches = {
        "AUTHOR": False,
        "TITLE": False,
        "YEAR": False,
        "VENUE": False
    }
    
    # Check if authors can be found in text
    if authors:
        for author in authors:
            author_clean = normalize_text(author)
            if author_clean and len(author_clean) > 1:
                for token in tokens_clean:
                    if get_similarity(author_clean, token) >= 0.8:
                        entity_matches["AUTHOR"] = True
                        break
                if entity_matches["AUTHOR"]:
                    break
                
                # Also try checking in the whole text
                if get_similarity(author_clean, normalize_text(text)) >= 0.6:
                    entity_matches["AUTHOR"] = True
    
    # Check if title can be found in text
    if title:
        title_clean = normalize_text(title)
        if title_clean and len(title_clean) > 3:
            if get_similarity(title_clean, normalize_text(text)) >= 0.6:
                entity_matches["TITLE"] = True
    
    # Check if year can be found in text
    if year and year.isdigit() and len(year) == 4:
        if year in text:
            entity_matches["YEAR"] = True
    
    # Check if venue can be found in text
    if venue:
        venue_clean = normalize_text(venue)
        if venue_clean and len(venue_clean) > 2:
            if get_similarity(venue_clean, normalize_text(text)) >= 0.6:
                entity_matches["VENUE"] = True
    
    return entity_matches

def analyze_single_file(data_path):
    """Analyze a single reference parsing data file."""
    try:
        # Load the data file
        with open(data_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except UnicodeDecodeError:
                # Fallback to Latin-1 encoding
                with open(data_path, 'r', encoding='latin-1') as f2:
                    data = json.load(f2)
        
        logger.info(f"Analyzing {len(data)} examples from {data_path}")
        
        # Count stats
        total_examples = len(data)
        has_structured_data = 0
        has_any_entity = defaultdict(int)
        entity_counts = Counter()
        token_counts = Counter({"total": 0})
        matchable_entities = Counter()
        
        # Analyze each example
        for i, example in enumerate(data):
            if 'structured' in example:
                has_structured_data += 1
                structured = example['structured']
                
                # Count which entity types are present in structured data
                if structured.get('authors'):
                    has_any_entity["authors"] += 1
                    matchable_entities["authors"] += len(structured['authors'])
                if structured.get('title'):
                    has_any_entity["title"] += 1
                    matchable_entities["title"] += 1
                if structured.get('year'):
                    has_any_entity["year"] += 1
                    matchable_entities["year"] += 1
                if structured.get('venue'):
                    has_any_entity["venue"] += 1
                    matchable_entities["venue"] += 1
                
                # Analyze if entities can be matched in text
                if 'text' in example:
                    entity_matches = analyze_entity_matching(example)
                    
                    # Count successful entity matches
                    for entity_type, matched in entity_matches.items():
                        if matched:
                            entity_counts[entity_type] += 1
                            
                    # If example already has tokens and labels, analyze token distribution
                    if 'tokens' in example and 'labels' in example:
                        tokens = example['tokens']
                        labels = example['labels']
                        
                        # Count token labels
                        for label in labels:
                            token_counts[label] += 1
                            token_counts["total"] += 1
        
        # Calculate percentages
        structured_pct = has_structured_data / total_examples * 100 if total_examples > 0 else 0
        entity_match_pct = {
            entity_type: count / matchable_entities.get(entity_type.lower() + "s", 1) * 100
            for entity_type, count in entity_counts.items()
        }
        
        # Print results
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Examples with structured data: {has_structured_data} ({structured_pct:.2f}%)")
        logger.info("Entity presence in structured data:")
        for entity, count in has_any_entity.items():
            logger.info(f"  {entity}: {count} ({count/total_examples*100:.2f}%)")
        
        logger.info("\nEntity match success rate:")
        for entity_type, pct in entity_match_pct.items():
            logger.info(f"  {entity_type}: {entity_counts[entity_type]} matches ({pct:.2f}%)")
        
        if token_counts["total"] > 0:
            logger.info("\nToken label distribution:")
            for label, count in token_counts.items():
                if label != "total":
                    logger.info(f"  {label}: {count} ({count/token_counts['total']*100:.6f}%)")
        
        # Create visualization
        create_distribution_plots(entity_counts, token_counts, data_path)
        
        return {
            "total_examples": total_examples,
            "structured_count": has_structured_data,
            "entity_presence": dict(has_any_entity),
            "entity_matches": dict(entity_counts),
            "token_distribution": {k: v for k, v in token_counts.items() if k != "total"}
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {data_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_distribution_plots(entity_counts, token_counts, data_path):
    """Create visualizations of label distributions."""
    try:
        # Create output directory
        output_dir = Path("logs/figures")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # File name base from input path
        file_base = Path(data_path).stem
        
        # 1. Entity match success plot
        plt.figure(figsize=(10, 6))
        entity_labels = list(entity_counts.keys())
        entity_values = [entity_counts[k] for k in entity_labels]
        bars = plt.bar(entity_labels, entity_values)
        plt.title('Entity Match Success Count')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"entity_match_{file_base}.png")
        plt.close()
        
        # 2. Token label distribution plot (exclude 'total')
        if token_counts.get("total", 0) > 0:
            token_labels = [k for k in token_counts.keys() if k != "total"]
            token_values = [token_counts[k] for k in token_labels]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(token_labels, token_values)
            plt.title('Token Label Distribution')
            plt.xlabel('Label')
            plt.ylabel('Count')
            
            # Add count and percentage labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                pct = height / token_counts["total"] * 100
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}\n({pct:.2f}%)', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"token_dist_{file_base}.png")
            
            # 3. Create log-scale version to better see small values
            plt.figure(figsize=(10, 6))
            plt.bar(token_labels, token_values)
            plt.title('Token Label Distribution (Log Scale)')
            plt.xlabel('Label')
            plt.ylabel('Count (log scale)')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(output_dir / f"token_dist_log_{file_base}.png")
            plt.close()
        
        logger.info(f"Distribution plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")

def main():
    """Main function to analyze reference parsing data."""
    # Set paths
    data_dir = Path("data/transfer_learning/prepared/reference_parsing")
    
    # Check if directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Analyze train, val, and test splits
    splits = ["train", "val", "test"]
    results = {}
    
    for split in splits:
        split_path = data_dir / split / "data.json"
        if split_path.exists():
            logger.info(f"Analyzing {split} data...")
            results[split] = analyze_single_file(split_path)
        else:
            logger.warning(f"{split} data file not found at {split_path}")
    
    # Print summary comparison
    if len(results) > 0:
        logger.info("\n" + "="*50)
        logger.info("SUMMARY COMPARISON ACROSS SPLITS")
        logger.info("="*50)
        
        logger.info("\nExample counts:")
        for split, data in results.items():
            if data:
                total = data["total_examples"]
                structured = data["structured_count"]
                logger.info(f"  {split}: {total} examples, {structured} with structured data ({structured/total*100:.2f}%)")
        
        logger.info("\nEntity match success counts by split:")
        for entity_type in ENTITY_TYPES:
            logger.info(f"  {entity_type}:")
            for split, data in results.items():
                if data and entity_type in data["entity_matches"]:
                    matches = data["entity_matches"][entity_type]
                    logger.info(f"    {split}: {matches} matches")
        
        # Compare token distributions if available
        if any("token_distribution" in data and data["token_distribution"] for data in results.values()):
            logger.info("\nToken label distribution comparison:")
            all_labels = set()
            for data in results.values():
                if data and "token_distribution" in data:
                    all_labels.update(data["token_distribution"].keys())
            
            for label in sorted(all_labels):
                logger.info(f"  {label}:")
                for split, data in results.items():
                    if data and "token_distribution" in data:
                        count = data["token_distribution"].get(label, 0)
                        total = sum(data["token_distribution"].values())
                        logger.info(f"    {split}: {count} ({count/total*100:.6f}% of tokens)")

if __name__ == "__main__":
    main() 