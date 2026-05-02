#!/usr/bin/env python
"""
LangChainGPT Dataset Preparation for Transfer Learning

This script prepares data samples from CORD-19 and PubLayNet datasets for transfer learning,
creating structured training sets for section classification, figure detection, and reference parsing.
"""

import argparse
import csv
import glob
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/data_preparation.log", mode="a")
    ]
)
logger = logging.getLogger("data_preparation")

# Constants
CORD19_SOURCE_DIRS = ["data/training_datasets/cord19/document_parses/pdf_json/", 
                      "data/training_datasets/cord19/document_parses/pmc_json/"]
CORD19_METADATA = "data/training_datasets/cord19/metadata.csv"
PUBLAYNET_SOURCE_DIR = "data/training_datasets/publaynet/"
SAMPLE_OUTPUT_DIR = "data/transfer_learning/samples/"
PREPARED_DATA_DIR = "data/transfer_learning/prepared/"
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]

# Ensure directories exist
def ensure_dirs(dirs: List[str]):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

class DatasetPreparer:
    """Main class for preparing datasets for transfer learning."""
    
    def __init__(self, args):
        """Initialize with command line arguments."""
        self.args = args
        self.cord19_sample_dir = os.path.join(SAMPLE_OUTPUT_DIR, "cord19_samples")
        self.publaynet_sample_dir = os.path.join(SAMPLE_OUTPUT_DIR, "publaynet_samples")
        
        # Output directories for each component
        self.section_dir = os.path.join(PREPARED_DATA_DIR, "section_classification")
        self.figure_dir = os.path.join(PREPARED_DATA_DIR, "figure_detection")
        self.reference_dir = os.path.join(PREPARED_DATA_DIR, "reference_parsing")
        
        # Create all necessary directories
        ensure_dirs([
            self.cord19_sample_dir, 
            self.publaynet_sample_dir,
            self.section_dir,
            self.figure_dir, 
            self.reference_dir,
            os.path.join(self.section_dir, "train"),
            os.path.join(self.section_dir, "val"),
            os.path.join(self.section_dir, "test"),
            os.path.join(self.figure_dir, "train"),
            os.path.join(self.figure_dir, "val"),
            os.path.join(self.figure_dir, "test"),
            os.path.join(self.reference_dir, "train"),
            os.path.join(self.reference_dir, "val"),
            os.path.join(self.reference_dir, "test"),
        ])
        
        # Stats tracking
        self.stats = {
            "cord19": {
                "total_papers": 0,
                "selected_papers": 0,
                "topics": Counter(),
                "years": Counter(),
                "sections": Counter(),
                "figures": 0,
                "references": 0,
            },
            "publaynet": {
                "total_papers": 0,
                "selected_papers": 0,
                "layout_types": Counter(),
            },
            "processing": {
                "start_time": time.time(),
                "end_time": None,
                "disk_usage_before": self._get_dir_size(PREPARED_DATA_DIR),
                "disk_usage_after": None,
            }
        }
    
    def _get_dir_size(self, path: str) -> int:
        """Calculate directory size in bytes."""
        if not os.path.exists(path):
            return 0
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
    
    def extract_cord19_samples(self):
        """Extract sample papers from CORD-19 dataset."""
        logger.info("Starting CORD-19 sample extraction")
        
        # Read metadata to get topic distribution
        metadata_df = None
        if os.path.exists(CORD19_METADATA):
            try:
                metadata_df = pd.read_csv(CORD19_METADATA)
                logger.info(f"Loaded metadata with {len(metadata_df)} records")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        # Gather all potential paper files
        all_papers = []
        for source_dir in CORD19_SOURCE_DIRS:
            if os.path.exists(source_dir):
                paper_files = glob.glob(f"{source_dir}/*.json")
                all_papers.extend(paper_files)
                logger.info(f"Found {len(paper_files)} papers in {source_dir}")
        
        if not all_papers:
            logger.error("No CORD-19 papers found!")
            return
        
        self.stats["cord19"]["total_papers"] = len(all_papers)
        
        # Evaluate papers in batches
        selected_papers = []
        batch_size = min(1000, len(all_papers))  # Process in batches of 1000
        paper_scores = {}
        
        for i in range(0, len(all_papers), batch_size):
            batch = all_papers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_papers) + batch_size - 1)//batch_size}")
            
            for paper_path in tqdm.tqdm(batch):
                try:
                    with open(paper_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Calculate paper score based on completeness
                    score = 0
                    paper_id = os.path.basename(paper_path).replace('.json', '')
                    
                    # Check for sections
                    if 'body_text' in data and len(data['body_text']) > 5:
                        score += 3
                        self.stats["cord19"]["sections"].update([len(data['body_text'])])
                    
                    # Check for figures
                    if 'ref_entries' in data:
                        figure_count = sum(1 for ref_type, _ in data['ref_entries'].items() 
                                           if 'fig' in ref_type.lower())
                        if figure_count > 0:
                            score += 2
                            self.stats["cord19"]["figures"] += figure_count
                    
                    # Check for references
                    if 'bib_entries' in data and len(data['bib_entries']) > 5:
                        score += 2
                        self.stats["cord19"]["references"] += len(data['bib_entries'])
                    
                    # Add metadata score if available
                    if metadata_df is not None:
                        try:
                            paper_meta = metadata_df[metadata_df['sha'] == paper_id]
                            if not paper_meta.empty:
                                # Add score for complete metadata
                                if not paper_meta['title'].iloc[0] in [None, '', 'Unknown']:
                                    score += 1
                                
                                # Track topic and year
                                if 'publish_time' in paper_meta.columns:
                                    year = str(paper_meta['publish_time'].iloc[0]).split('-')[0]
                                    if year.isdigit():
                                        self.stats["cord19"]["years"][year] += 1
                                
                                if 'journal' in paper_meta.columns and not pd.isna(paper_meta['journal'].iloc[0]):
                                    self.stats["cord19"]["topics"][paper_meta['journal'].iloc[0]] += 1
                        except Exception as e:
                            logger.debug(f"Error processing metadata for {paper_id}: {e}")
                    
                    paper_scores[paper_path] = score
                    
                except Exception as e:
                    logger.warning(f"Error processing {paper_path}: {e}")
        
        # Select top scoring papers
        top_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        selected_papers = [path for path, _ in top_papers[:min(self.args.cord19_samples, len(top_papers))]]
        
        # Copy selected papers
        logger.info(f"Copying {len(selected_papers)} selected CORD-19 papers to sample directory")
        for paper_path in tqdm.tqdm(selected_papers):
            try:
                paper_id = os.path.basename(paper_path)
                output_path = os.path.join(self.cord19_sample_dir, paper_id)
                shutil.copy(paper_path, output_path)
                self.stats["cord19"]["selected_papers"] += 1
            except Exception as e:
                logger.warning(f"Failed to copy {paper_path}: {e}")
        
        logger.info(f"CORD-19 extraction complete: {self.stats['cord19']['selected_papers']} papers")
    
    def extract_publaynet_samples(self):
        """Extract sample documents from PubLayNet dataset."""
        logger.info("Starting PubLayNet sample extraction")
        
        # Check if PubLayNet directory exists
        if not os.path.exists(PUBLAYNET_SOURCE_DIR):
            logger.error(f"PubLayNet source directory {PUBLAYNET_SOURCE_DIR} not found!")
            return
        
        # Look for the annotations and images
        train_json = os.path.join(PUBLAYNET_SOURCE_DIR, "train.json")
        val_json = os.path.join(PUBLAYNET_SOURCE_DIR, "val.json")
        images_dir = os.path.join(PUBLAYNET_SOURCE_DIR, "images")
        
        # Load annotations
        annotations = []
        if os.path.exists(train_json):
            try:
                with open(train_json, 'r') as f:
                    train_data = json.load(f)
                    annotations.extend(train_data.get("annotations", []))
                    self.stats["publaynet"]["total_papers"] += len(train_data.get("images", []))
                    logger.info(f"Loaded {len(train_data.get('images', []))} training samples from PubLayNet")
            except Exception as e:
                logger.warning(f"Error loading PubLayNet training data: {e}")
        
        if os.path.exists(val_json):
            try:
                with open(val_json, 'r') as f:
                    val_data = json.load(f)
                    annotations.extend(val_data.get("annotations", []))
                    self.stats["publaynet"]["total_papers"] += len(val_data.get("images", []))
                    logger.info(f"Loaded {len(val_data.get('images', []))} validation samples from PubLayNet")
            except Exception as e:
                logger.warning(f"Error loading PubLayNet validation data: {e}")
        
        if not annotations:
            logger.error("No PubLayNet annotations found!")
            return
        
        # Group annotations by image_id to determine layout richness
        images_by_id = defaultdict(list)
        for ann in annotations:
            images_by_id[ann["image_id"]].append(ann)
        
        # Score images by layout diversity
        image_scores = {}
        for image_id, anns in images_by_id.items():
            # Count different layout element types
            category_counts = Counter(ann["category_id"] for ann in anns)
            # More diverse layouts get higher scores
            layout_diversity = len(category_counts)
            # More elements get higher scores (up to a cap)
            element_count = min(len(anns), 20) / 20
            # Combined score
            image_scores[image_id] = layout_diversity + element_count
            
            # Track layout types
            self.stats["publaynet"]["layout_types"].update(category_counts.keys())
        
        # Select top scoring images
        top_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        selected_images = [img_id for img_id, _ in top_images[:min(self.args.publaynet_samples, len(top_images))]]
        
        # Copy selected annotations and images
        logger.info(f"Preparing {len(selected_images)} selected PubLayNet samples")
        
        # Create a new annotation file for selected samples
        selected_annotations = {
            "images": [],
            "annotations": [],
            "categories": [] if not annotations else next(
                (data.get("categories", []) for data in [train_data, val_data] if "categories" in data), 
                []
            )
        }
        
        # Find and collect annotations and images
        for img_id in selected_images:
            # Find image info
            img_info = None
            for data in [train_data, val_data]:
                if "images" in data:
                    img_info = next((img for img in data["images"] if img["id"] == img_id), None)
                    if img_info:
                        break
            
            if not img_info:
                continue
                
            # Add image to selection
            selected_annotations["images"].append(img_info)
            
            # Add annotations for this image
            img_anns = [ann for ann in annotations if ann["image_id"] == img_id]
            selected_annotations["annotations"].extend(img_anns)
            
            # Copy image file if it exists
            img_file = img_info.get("file_name")
            if img_file and os.path.exists(os.path.join(images_dir, img_file)):
                try:
                    shutil.copy(
                        os.path.join(images_dir, img_file),
                        os.path.join(self.publaynet_sample_dir, img_file)
                    )
                    self.stats["publaynet"]["selected_papers"] += 1
                except Exception as e:
                    logger.warning(f"Failed to copy image {img_file}: {e}")
        
        # Save the selected annotations
        with open(os.path.join(self.publaynet_sample_dir, "selected_annotations.json"), 'w') as f:
            json.dump(selected_annotations, f)
            
        logger.info(f"PubLayNet extraction complete: {self.stats['publaynet']['selected_papers']} samples")
    
    def prepare_section_classification_data(self):
        """Prepare data for section classification model."""
        logger.info("Preparing section classification data")
        
        section_data = []
        
        # Process CORD-19 papers
        cord19_papers = glob.glob(f"{self.cord19_sample_dir}/*.json")
        logger.info(f"Processing {len(cord19_papers)} CORD-19 papers for section classification")
        
        for paper_path in tqdm.tqdm(cord19_papers):
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'body_text' not in data or not data['body_text']:
                    continue
                
                for section in data['body_text']:
                    # Extract section info
                    section_title = section.get('section', '').strip()
                    section_text = section.get('text', '').strip()
                    
                    if not section_text:
                        continue
                    
                    # Normalize section title for classification
                    normalized_title = self._normalize_section_title(section_title)
                    
                    if normalized_title:  # Only add if we have a valid section type
                        section_data.append({
                            "text": section_text[:1000],  # Limit text length
                            "section_type": normalized_title,
                            "original_title": section_title,
                            "source": "cord19",
                            "paper_id": os.path.basename(paper_path)
                        })
            except Exception as e:
                logger.warning(f"Error processing {paper_path} for section classification: {e}")
        
        # Split the data
        logger.info(f"Splitting {len(section_data)} section samples into train/val/test")
        train_data, test_data = train_test_split(section_data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
        
        # Save the splits
        self._save_json(os.path.join(self.section_dir, "train", "data.json"), train_data)
        self._save_json(os.path.join(self.section_dir, "val", "data.json"), val_data)
        self._save_json(os.path.join(self.section_dir, "test", "data.json"), test_data)
        
        logger.info(f"Section classification data prepared: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    def prepare_figure_detection_data(self):
        """Prepare data for figure detection model."""
        logger.info("Preparing figure detection data")
        
        figure_data = []
        
        # Process CORD-19 papers for text mentions of figures
        cord19_papers = glob.glob(f"{self.cord19_sample_dir}/*.json")
        logger.info(f"Processing {len(cord19_papers)} CORD-19 papers for figure detection")
        
        for paper_path in tqdm.tqdm(cord19_papers):
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'body_text' not in data or not data['body_text']:
                    continue
                
                # Find paragraphs mentioning figures
                for section in data['body_text']:
                    text = section.get('text', '').strip()
                    section_name = section.get('section', '').strip()
                    
                    if not text:
                        continue
                    
                    # Check for figure mentions
                    has_figure = False
                    figure_markers = ['Figure', 'Fig.', 'Fig ', 'figure', 'fig.', 'fig ']
                    
                    for marker in figure_markers:
                        if marker in text:
                            has_figure = True
                            break
                    
                    # Add to dataset
                    figure_data.append({
                        "text": text[:1000],  # Limit text length
                        "has_figure": has_figure,
                        "section": section_name,
                        "source": "cord19",
                        "paper_id": os.path.basename(paper_path)
                    })
            except Exception as e:
                logger.warning(f"Error processing {paper_path} for figure detection: {e}")
        
        # Process PubLayNet data
        publaynet_annotations = os.path.join(self.publaynet_sample_dir, "selected_annotations.json")
        if os.path.exists(publaynet_annotations):
            try:
                with open(publaynet_annotations, 'r') as f:
                    pub_data = json.load(f)
                
                # Create mapping from image_id to annotations
                img_to_anns = defaultdict(list)
                for ann in pub_data.get('annotations', []):
                    img_to_anns[ann['image_id']].append(ann)
                
                # Create mapping from category_id to name
                cat_id_to_name = {cat['id']: cat['name'] for cat in pub_data.get('categories', [])}
                
                # Add sample for each image
                for img in pub_data.get('images', []):
                    img_id = img['id']
                    anns = img_to_anns[img_id]
                    
                    # Check if this image has figure annotations
                    figure_types = ['figure', 'table', 'chart', 'diagram']
                    has_figure = any(
                        cat_id_to_name.get(ann['category_id'], '').lower() in figure_types 
                        for ann in anns
                    )
                    
                    # Add example
                    figure_data.append({
                        "image_id": img_id,
                        "file_name": img.get('file_name', ''),
                        "has_figure": has_figure,
                        "source": "publaynet",
                        "annotation_count": len(anns)
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing PubLayNet annotations: {e}")
        
        # Balance dataset if too imbalanced
        has_figure_samples = [sample for sample in figure_data if sample['has_figure']]
        no_figure_samples = [sample for sample in figure_data if not sample['has_figure']]
        
        logger.info(f"Figure detection dataset: {len(has_figure_samples)} positive, {len(no_figure_samples)} negative")
        
        # Rebalance if needed (downsample majority class)
        if len(has_figure_samples) > 2 * len(no_figure_samples):
            has_figure_samples = random.sample(has_figure_samples, 2 * len(no_figure_samples))
            figure_data = has_figure_samples + no_figure_samples
            logger.info(f"Rebalanced to {len(has_figure_samples)} positive, {len(no_figure_samples)} negative")
        elif len(no_figure_samples) > 2 * len(has_figure_samples):
            no_figure_samples = random.sample(no_figure_samples, 2 * len(has_figure_samples))
            figure_data = has_figure_samples + no_figure_samples
            logger.info(f"Rebalanced to {len(has_figure_samples)} positive, {len(no_figure_samples)} negative")
        
        # Split the data
        random.shuffle(figure_data)
        train_size = int(len(figure_data) * TRAIN_VAL_TEST_SPLIT[0])
        val_size = int(len(figure_data) * TRAIN_VAL_TEST_SPLIT[1])
        
        train_data = figure_data[:train_size]
        val_data = figure_data[train_size:train_size+val_size]
        test_data = figure_data[train_size+val_size:]
        
        # Save the splits
        self._save_json(os.path.join(self.figure_dir, "train", "data.json"), train_data)
        self._save_json(os.path.join(self.figure_dir, "val", "data.json"), val_data)
        self._save_json(os.path.join(self.figure_dir, "test", "data.json"), test_data)
        
        logger.info(f"Figure detection data prepared: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    def prepare_reference_parsing_data(self):
        """Prepare data for reference parsing model."""
        logger.info("Preparing reference parsing data")
        
        reference_data = []
        
        # Process CORD-19 papers
        cord19_papers = glob.glob(f"{self.cord19_sample_dir}/*.json")
        logger.info(f"Processing {len(cord19_papers)} CORD-19 papers for reference parsing")
        
        for paper_path in tqdm.tqdm(cord19_papers):
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'bib_entries' not in data or not data['bib_entries']:
                    continue
                
                for ref_id, ref_data in data['bib_entries'].items():
                    if not ref_data or not isinstance(ref_data, dict):
                        continue
                    
                    # Extract structured reference data
                    ref_text = ""
                    
                    # Authors
                    if 'authors' in ref_data and ref_data['authors']:
                        authors = []
                        for author in ref_data['authors']:
                            if 'last' in author:
                                author_name = author.get('last', '')
                                if 'first' in author:
                                    author_name = f"{author.get('first', '')} {author_name}"
                                authors.append(author_name)
                        
                        if authors:
                            ref_text += ", ".join(authors) + ". "
                    
                    # Title
                    if 'title' in ref_data and ref_data['title']:
                        ref_text += ref_data['title'] + ". "
                    
                    # Year
                    if 'year' in ref_data and ref_data['year']:
                        ref_text += f"({ref_data['year']}). "
                    
                    # Venue
                    if 'venue' in ref_data and ref_data['venue']:
                        ref_text += ref_data['venue'] + ". "
                    
                    if not ref_text:
                        continue
                    
                    # Add to dataset with structured fields
                    reference_data.append({
                        "text": ref_text,
                        "structured": {
                            "authors": [a.get('last', '') for a in ref_data.get('authors', []) if 'last' in a],
                            "title": ref_data.get('title', ''),
                            "year": ref_data.get('year', ''),
                            "venue": ref_data.get('venue', '')
                        },
                        "ref_id": ref_id,
                        "source": "cord19",
                        "paper_id": os.path.basename(paper_path)
                    })
            except Exception as e:
                logger.warning(f"Error processing {paper_path} for reference parsing: {e}")
        
        # Split the data
        logger.info(f"Splitting {len(reference_data)} reference samples into train/val/test")
        random.shuffle(reference_data)
        train_size = int(len(reference_data) * TRAIN_VAL_TEST_SPLIT[0])
        val_size = int(len(reference_data) * TRAIN_VAL_TEST_SPLIT[1])
        
        train_data = reference_data[:train_size]
        val_data = reference_data[train_size:train_size+val_size]
        test_data = reference_data[train_size+val_size:]
        
        # Save the splits
        self._save_json(os.path.join(self.reference_dir, "train", "data.json"), train_data)
        self._save_json(os.path.join(self.reference_dir, "val", "data.json"), val_data)
        self._save_json(os.path.join(self.reference_dir, "test", "data.json"), test_data)
        
        logger.info(f"Reference parsing data prepared: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    def _normalize_section_title(self, title: str) -> str:
        """Normalize section titles to standard categories."""
        title = title.lower().strip()
        
        if not title:
            return "body"
        
        # Map common section titles to standard categories
        if any(t in title for t in ["abstract", "summary"]):
            return "abstract"
        elif any(t in title for t in ["introduc", "background", "overview"]):
            return "introduction"
        elif any(t in title for t in ["method", "materials", "procedure", "experiment"]):
            return "methods"
        elif any(t in title for t in ["result", "finding", "outcome"]):
            return "results"
        elif any(t in title for t in ["discuss", "conclusion", "summary"]):
            return "discussion"
        elif any(t in title for t in ["reference", "bibliography", "literature", "citation"]):
            return "references"
        elif any(t in title for t in ["acknowledge", "funding", "support", "contribut"]):
            return "acknowledgments"
        elif any(t in title for t in ["appendix", "supplement"]):
            return "appendix"
        elif any(t in title for t in ["table", "figure", "fig.", "chart", "graph"]):
            return "figure"
        else:
            return "body"  # Default category
    
    def _save_json(self, filepath: str, data: Any):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved data to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
            return False
    
    def prepare_all_datasets(self):
        """Run the full dataset preparation pipeline."""
        start_time = time.time()
        logger.info("Starting dataset preparation for transfer learning")
        
        # Extract samples
        self.extract_cord19_samples()
        self.extract_publaynet_samples()
        
        # Prepare structured datasets
        self.prepare_section_classification_data()
        self.prepare_figure_detection_data()
        self.prepare_reference_parsing_data()
        
        # Update stats
        self.stats["processing"]["end_time"] = time.time()
        self.stats["processing"]["elapsed_time"] = self.stats["processing"]["end_time"] - self.stats["processing"]["start_time"]
        self.stats["processing"]["disk_usage_after"] = self._get_dir_size(PREPARED_DATA_DIR)
        self.stats["processing"]["disk_usage_increase"] = (
            self.stats["processing"]["disk_usage_after"] - self.stats["processing"]["disk_usage_before"]
        )
        
        # Save stats
        self._save_json(os.path.join(PREPARED_DATA_DIR, "preparation_stats.json"), self.stats)
        
        # Print summary
        self._print_summary()
        
        logger.info(f"Dataset preparation complete in {self.stats['processing']['elapsed_time']:.2f} seconds")
    
    def _print_summary(self):
        """Print a summary of the dataset preparation."""
        summary = {
            "CORD-19": {
                "Selected papers": self.stats["cord19"]["selected_papers"],
                "Total sections": sum(self.stats["cord19"]["sections"].values()),
                "Total figures": self.stats["cord19"]["figures"],
                "Total references": self.stats["cord19"]["references"],
                "Top topics": dict(self.stats["cord19"]["topics"].most_common(5)),
                "Publication years": dict(self.stats["cord19"]["years"].most_common(5))
            },
            "PubLayNet": {
                "Selected papers": self.stats["publaynet"]["selected_papers"],
                "Layout types": dict(self.stats["publaynet"]["layout_types"])
            },
            "Processing": {
                "Elapsed time": f"{self.stats['processing']['elapsed_time']:.2f} seconds",
                "Disk usage": f"{self.stats['processing']['disk_usage_increase'] / (1024*1024):.2f} MB"
            }
        }
        
        print("\n" + "="*80)
        print(" Dataset Preparation Summary ")
        print("="*80)
        
        for section, data in summary.items():
            print(f"\n{section}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print(f" Data saved to: {PREPARED_DATA_DIR}")
        print("="*80)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare datasets for LangChainGPT transfer learning')
    
    parser.add_argument('--cord19-samples', type=int, default=5000,
                        help='Number of CORD-19 papers to extract (default: 5000)')
    parser.add_argument('--publaynet-samples', type=int, default=1000,
                        help='Number of PubLayNet samples to extract (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for processing (default: 1000)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip extraction step and use existing samples')
    parser.add_argument('--skip-section', action='store_true',
                        help='Skip section classification data preparation')
    parser.add_argument('--skip-figure', action='store_true',
                        help='Skip figure detection data preparation')
    parser.add_argument('--skip-reference', action='store_true', 
                        help='Skip reference parsing data preparation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize preparer
    preparer = DatasetPreparer(args)
    
    # Run preparation
    try:
        preparer.prepare_all_datasets()
    except KeyboardInterrupt:
        logger.info("Dataset preparation interrupted by user")
    except Exception as e:
        logger.error(f"Error during dataset preparation: {e}", exc_info=True)
    
    logger.info("Dataset preparation script finished")

if __name__ == "__main__":
    main() 