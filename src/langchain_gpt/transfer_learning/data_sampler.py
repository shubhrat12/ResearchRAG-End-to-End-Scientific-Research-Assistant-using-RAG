"""
Data sampling and preparation module for transfer learning.

This module handles sampling papers from datasets and preparing them for transfer learning.
"""

import os
import json
import random
import logging
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

class DataSampler:
    """Class for sampling and preparing data for transfer learning."""
    
    def __init__(
        self,
        cord19_dir: str = "data/training_datasets/cord19",
        publaynet_dir: str = "data/training_datasets/publaynet",
        output_dir: str = "data/transfer_learning",
        random_seed: int = 42
    ):
        """
        Initialize the data sampler.
        
        Args:
            cord19_dir: Path to CORD-19 dataset
            publaynet_dir: Path to PubLayNet dataset
            output_dir: Path to output directory for prepared data
            random_seed: Random seed for reproducibility
        """
        self.cord19_dir = Path(cord19_dir)
        self.publaynet_dir = Path(publaynet_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories if they don't exist
        self.cord19_samples_dir = self.output_dir / "samples" / "cord19_samples"
        self.publaynet_samples_dir = self.output_dir / "samples" / "publaynet_samples"
        
        self.prepared_dir = self.output_dir / "prepared"
        self.section_classification_dir = self.prepared_dir / "section_classification"
        self.figure_detection_dir = self.prepared_dir / "figure_detection"
        self.reference_parsing_dir = self.prepared_dir / "reference_parsing"
        
        # Create all directories
        for dir_path in [
            self.cord19_samples_dir,
            self.publaynet_samples_dir,
            self.section_classification_dir,
            self.figure_detection_dir,
            self.reference_parsing_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info(f"Initialized DataSampler with CORD-19 dir: {cord19_dir}, PubLayNet dir: {publaynet_dir}")
    
    def sample_cord19_papers(self, sample_size: int = 5000) -> List[str]:
        """
        Sample papers from CORD-19 dataset.
        
        Args:
            sample_size: Number of papers to sample
            
        Returns:
            List of sampled paper IDs
        """
        logger.info(f"Sampling {sample_size} papers from CORD-19 dataset")
        
        # Load metadata
        metadata_path = self.cord19_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"CORD-19 metadata not found at {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata for {len(metadata)} papers")
        
        # Filter papers that have both PDF JSON and full text
        has_pdf_json = metadata['pdf_json_files'].notna()
        has_fulltext = metadata['has_full_text'] == True
        valid_papers = metadata[has_pdf_json & has_fulltext]
        
        logger.info(f"Found {len(valid_papers)} papers with PDF JSON and full text")
        
        # If we have fewer valid papers than requested sample size, adjust sample size
        if len(valid_papers) < sample_size:
            logger.warning(f"Only {len(valid_papers)} valid papers available, adjusting sample size")
            sample_size = len(valid_papers)
        
        # Sample papers
        sampled_papers = valid_papers.sample(sample_size, random_state=42)
        logger.info(f"Sampled {len(sampled_papers)} papers")
        
        # Copy sampled papers to output directory
        pdf_json_dir = self.cord19_dir / "document_parses" / "pdf_json"
        
        # Get list of paper IDs and their JSON files
        sampled_ids = []
        
        for _, row in tqdm(sampled_papers.iterrows(), total=len(sampled_papers), desc="Copying sampled papers"):
            if pd.isna(row['pdf_json_files']):
                continue
                
            # Each paper might have multiple JSON files (one per version)
            json_files = row['pdf_json_files'].split('; ')
            
            for json_file in json_files:
                if not json_file:
                    continue
                    
                # Extract the paper ID from the filename
                paper_id = Path(json_file).stem
                
                # Source and destination paths
                src_path = pdf_json_dir / f"{paper_id}.json"
                dst_path = self.cord19_samples_dir / f"{paper_id}.json"
                
                if src_path.exists():
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
                    sampled_ids.append(paper_id)
                    # Break after finding the first valid JSON file for this paper
                    break
        
        logger.info(f"Copied {len(sampled_ids)} JSON files to {self.cord19_samples_dir}")
        
        # Write list of sampled paper IDs to file
        with open(self.cord19_samples_dir / "sampled_papers.json", 'w') as f:
            json.dump(sampled_ids, f, indent=2)
        
        return sampled_ids
    
    def prepare_section_classification_data(self, cord19_sample_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Prepare data for section classification.
        
        Args:
            cord19_sample_ids: List of sampled CORD-19 paper IDs
            
        Returns:
            Dictionary with train, val, and test data
        """
        logger.info("Preparing data for section classification")
        
        # Define section types
        section_types = [
            "abstract", "introduction", "background", "related_work", 
            "methods", "experiments", "results", "discussion", 
            "conclusion", "references", "appendix", "acknowledgements"
        ]
        
        section_data = []
        
        # Process sampled CORD-19 papers
        for paper_id in tqdm(cord19_sample_ids, desc="Extracting sections"):
            json_path = self.cord19_samples_dir / f"{paper_id}.json"
            
            if not json_path.exists():
                continue
                
            try:
                with open(json_path, 'r') as f:
                    paper = json.load(f)
                
                # Extract body text and attempt to identify sections
                for idx, section in enumerate(paper.get('body_text', [])):
                    section_name = section.get('section', '').lower()
                    text = section.get('text', '').strip()
                    
                    if not text or len(text) < 50:
                        continue
                    
                    # Map section name to one of our standard types
                    section_type = self._map_section_name(section_name)
                    
                    # Only include mapped sections
                    if section_type:
                        section_data.append({
                            'paper_id': paper_id,
                            'section_idx': idx,
                            'section_name': section_name,
                            'section_type': section_type,
                            'text': text[:1024],  # Truncate to avoid very long sequences
                        })
            except Exception as e:
                logger.warning(f"Error processing paper {paper_id}: {str(e)}")
        
        logger.info(f"Extracted {len(section_data)} sections from {len(cord19_sample_ids)} papers")
        
        # Split data into train, validation, and test sets (80/10/10)
        random.shuffle(section_data)
        n = len(section_data)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        train_data = section_data[:train_size]
        val_data = section_data[train_size:train_size + val_size]
        test_data = section_data[train_size + val_size:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
        
        # Save datasets
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split, data in datasets.items():
            output_path = self.section_classification_dir / f"{split}.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {split} data to {output_path}")
        
        return datasets
    
    def prepare_figure_detection_data(self) -> Dict[str, List[Dict]]:
        """
        Prepare data for figure detection using PubLayNet.
        
        Returns:
            Dictionary with train, val, and test data
        """
        logger.info("Preparing data for figure detection")
        
        # PubLayNet has pre-defined train/val/test splits
        train_dir = self.publaynet_dir / "train"
        val_dir = self.publaynet_dir / "val"
        test_dir = self.publaynet_dir / "test"
        annotations_dir = self.publaynet_dir / "annotations"
        
        # Check for annotations files
        train_ann_path = next(annotations_dir.glob("*train*.json"), None)
        val_ann_path = next(annotations_dir.glob("*val*.json"), None)
        test_ann_path = next(annotations_dir.glob("*test*.json"), None)
        
        if not train_ann_path or not val_ann_path:
            logger.warning("PubLayNet annotations not found, using sample data")
            # Create sample data for demonstration
            figure_data = self._create_sample_figure_data()
        else:
            # Load annotations
            figure_data = {}
            
            for split, path in [('train', train_ann_path), ('val', val_ann_path), ('test', test_ann_path)]:
                if path and path.exists():
                    with open(path, 'r') as f:
                        annotations = json.load(f)
                    
                    # Process and convert to LayoutLM format
                    figure_data[split] = self._convert_publaynet_to_layoutlm(annotations)
                    logger.info(f"Loaded {len(figure_data[split])} {split} examples for figure detection")
                else:
                    figure_data[split] = []
        
        # Save datasets
        for split, data in figure_data.items():
            # Sample a smaller subset for efficiency if dataset is large
            if split == 'train' and len(data) > 5000:
                data = random.sample(data, 5000)
                logger.info(f"Sampled 5000 examples from {split} data for efficiency")
            
            output_path = self.figure_detection_dir / f"{split}.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {split} data to {output_path}")
        
        return figure_data
    
    def prepare_reference_parsing_data(self, cord19_sample_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Prepare data for reference parsing.
        
        Args:
            cord19_sample_ids: List of sampled CORD-19 paper IDs
            
        Returns:
            Dictionary with train, val, and test data
        """
        logger.info("Preparing data for reference parsing")
        
        reference_data = []
        
        # Process sampled CORD-19 papers
        for paper_id in tqdm(cord19_sample_ids, desc="Extracting references"):
            json_path = self.cord19_samples_dir / f"{paper_id}.json"
            
            if not json_path.exists():
                continue
                
            try:
                with open(json_path, 'r') as f:
                    paper = json.load(f)
                
                # Extract bib entries
                bib_entries = paper.get('bib_entries', {})
                
                for ref_id, ref_data in bib_entries.items():
                    title = ref_data.get('title', '')
                    authors = [a.get('first', '') + ' ' + a.get('last', '') for a in ref_data.get('authors', [])]
                    year = ref_data.get('year', '')
                    venue = ref_data.get('venue', '')
                    
                    if not title:
                        continue
                    
                    # Create formatted reference string
                    ref_string = self._format_reference(title, authors, year, venue)
                    
                    # Create labeled data for sequence labeling
                    labeled_data = self._label_reference_components(ref_string, {
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'venue': venue
                    })
                    
                    reference_data.append({
                        'paper_id': paper_id,
                        'ref_id': ref_id,
                        'reference_string': ref_string,
                        'labeled_data': labeled_data,
                        'components': {
                            'title': title,
                            'authors': authors,
                            'year': year,
                            'venue': venue
                        }
                    })
            except Exception as e:
                logger.warning(f"Error processing references for paper {paper_id}: {str(e)}")
        
        logger.info(f"Extracted {len(reference_data)} references from {len(cord19_sample_ids)} papers")
        
        # Split data into train, validation, and test sets (80/10/10)
        random.shuffle(reference_data)
        n = len(reference_data)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        train_data = reference_data[:train_size]
        val_data = reference_data[train_size:train_size + val_size]
        test_data = reference_data[train_size + val_size:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
        
        # Save datasets
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split, data in datasets.items():
            output_path = self.reference_parsing_dir / f"{split}.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {split} data to {output_path}")
        
        return datasets
    
    def _map_section_name(self, section_name: str) -> Optional[str]:
        """Map section name to standard section type."""
        section_name = section_name.lower()
        
        mapping = {
            'abstract': 'abstract',
            'introduction': 'introduction',
            'background': 'background',
            'related work': 'related_work',
            'literature review': 'related_work',
            'method': 'methods',
            'methods': 'methods',
            'methodology': 'methods',
            'experimental': 'experiments',
            'experiments': 'experiments',
            'result': 'results',
            'results': 'results',
            'discussion': 'discussion',
            'conclusion': 'conclusion',
            'conclusions': 'conclusion',
            'acknowledgement': 'acknowledgements',
            'acknowledgements': 'acknowledgements',
            'references': 'references',
            'appendix': 'appendix'
        }
        
        for key, value in mapping.items():
            if key in section_name:
                return value
        
        return None
    
    def _convert_publaynet_to_layoutlm(self, annotations: Dict) -> List[Dict]:
        """Convert PubLayNet annotations to LayoutLM format."""
        # This is a simplified conversion - actual implementation would be more complex
        converted_data = []
        
        # Map category ID to label name
        category_map = {
            1: "text",
            2: "title",
            3: "list",
            4: "table",
            5: "figure"
        }
        
        # Process each image annotation
        for image in annotations.get('images', []):
            image_id = image.get('id')
            width = image.get('width')
            height = image.get('height')
            
            # Get annotations for this image
            image_anns = [a for a in annotations.get('annotations', []) if a.get('image_id') == image_id]
            
            # Create bounding boxes and labels
            boxes = []
            labels = []
            
            for ann in image_anns:
                category_id = ann.get('category_id')
                bbox = ann.get('bbox')  # [x, y, width, height]
                
                if bbox and category_id:
                    # Convert to normalized coordinates
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x/width, y/height, (x+w)/width, (y+h)/height
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(category_map.get(category_id, "unknown"))
            
            if boxes and labels:
                converted_data.append({
                    'image_id': image_id,
                    'file_name': image.get('file_name', ''),
                    'width': width,
                    'height': height,
                    'boxes': boxes,
                    'labels': labels
                })
        
        return converted_data
    
    def _create_sample_figure_data(self) -> Dict[str, List[Dict]]:
        """Create sample figure detection data if PubLayNet is not available."""
        # Create synthetic data for demonstration
        sample_data = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Create simple synthetic examples
        for split in ['train', 'val', 'test']:
            n_samples = 1000 if split == 'train' else 100
            
            for i in range(n_samples):
                # Random document dimensions
                width, height = 800, 1200
                
                # Create random boxes and labels
                n_boxes = random.randint(3, 10)
                boxes = []
                labels = []
                
                for _ in range(n_boxes):
                    # Random box dimensions
                    x1 = random.random() * 0.8
                    y1 = random.random() * 0.8
                    x2 = x1 + random.random() * 0.2
                    y2 = y1 + random.random() * 0.2
                    
                    # Random label (bias towards text)
                    label = random.choices(
                        ["text", "title", "list", "table", "figure"],
                        weights=[0.6, 0.1, 0.1, 0.1, 0.1]
                    )[0]
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)
                
                sample_data[split].append({
                    'image_id': f"{split}_{i}",
                    'file_name': f"sample_{split}_{i}.png",
                    'width': width,
                    'height': height,
                    'boxes': boxes,
                    'labels': labels
                })
        
        return sample_data
    
    def _format_reference(self, title: str, authors: List[str], year: str, venue: str) -> str:
        """Format reference components into a reference string."""
        if not title:
            return ""
            
        ref_parts = []
        
        # Authors
        if authors:
            if len(authors) == 1:
                ref_parts.append(authors[0])
            elif len(authors) == 2:
                ref_parts.append(f"{authors[0]} and {authors[1]}")
            else:
                ref_parts.append(f"{authors[0]} et al.")
        
        # Year
        if year:
            ref_parts.append(f"({year})")
        
        # Title
        ref_parts.append(title)
        
        # Venue
        if venue:
            ref_parts.append(f"In: {venue}")
        
        return " ".join(ref_parts)
    
    def _label_reference_components(self, ref_string: str, components: Dict) -> List[Dict]:
        """Create labeled data for reference parsing."""
        # This is a simplified approach - a real implementation would use
        # more sophisticated NLP to match components in the reference string
        
        labeled_data = []
        
        # Split reference string into tokens
        tokens = ref_string.split()
        
        # For each token, determine its label
        for token in tokens:
            token_lower = token.lower()
            
            # Try to match token to components
            if components.get('title', '') and token_lower in components['title'].lower():
                label = 'TITLE'
            elif any(token_lower in author.lower() for author in components.get('authors', [])):
                label = 'AUTHOR'
            elif components.get('year', '') and token_lower in components['year'].lower():
                label = 'YEAR'
            elif components.get('venue', '') and token_lower in components['venue'].lower():
                label = 'VENUE'
            else:
                label = 'O'  # Other
            
            labeled_data.append({
                'token': token,
                'label': label
            })
        
        return labeled_data
    
    def prepare_all_datasets(self, cord19_sample_size: int = 5000) -> None:
        """
        Prepare all datasets for transfer learning.
        
        Args:
            cord19_sample_size: Number of CORD-19 papers to sample
        """
        # Sample CORD-19 papers
        cord19_sample_ids = self.sample_cord19_papers(cord19_sample_size)
        
        # Prepare datasets for each task
        self.prepare_section_classification_data(cord19_sample_ids)
        self.prepare_figure_detection_data()
        self.prepare_reference_parsing_data(cord19_sample_ids)
        
        logger.info("All datasets prepared successfully") 