"""
Comprehensive transfer learning implementation for LangChainGPT project.

This script implements transfer learning for section classification
and reference parsing using pretrained models like SciBERT and LayoutLM.
"""

# ======== DEVICE CONFIGURATION ========
import os
import sys
import torch
import traceback

# Check for CUDA availability
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    # Set CPU-specific optimizations if no GPU available
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    DEVICE = torch.device("cpu")
    print("CUDA not available. Using CPU mode.")

import json
import logging
import argparse
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import torch modules
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/transfer_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transfer_learning")

# ======== MODEL CONFIGURATIONS ========
# Define section types for classification
SECTION_TYPES = [
    "abstract", "introduction", "background", "related_work", 
    "methods", "experiments", "results", "discussion", 
    "conclusion", "references", "appendix", "acknowledgements"
]

# Define label types for reference parsing
REF_LABEL_TYPES = ["O", "AUTHOR", "TITLE", "YEAR", "VENUE"]

# Model configurations
MODEL_CONFIGS = {
    "section_classifier": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "task_type": "classification",
        "num_labels": len(SECTION_TYPES),
        "id2label": {i: label for i, label in enumerate(SECTION_TYPES)},
        "label2id": {label: i for i, label in enumerate(SECTION_TYPES)}
    },
    "reference_parser": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "task_type": "token-classification",
        "num_labels": len(REF_LABEL_TYPES),
        "id2label": {i: label for i, label in enumerate(REF_LABEL_TYPES)},
        "label2id": {label: i for i, label in enumerate(REF_LABEL_TYPES)}
    }
}

# ======== DATASETS ========
class SectionClassificationDataset(Dataset):
    """Dataset for section classification."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """Initialize the dataset."""
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except UnicodeDecodeError:
            # Fallback to Latin-1 encoding if UTF-8 fails
            logger.warning(f"UTF-8 decoding failed for {self.data_path}, trying with ISO-8859-1 encoding")
            with open(self.data_path, 'r', encoding='ISO-8859-1') as f:
                self.data = json.load(f)
        
        # Validate schema on first item
        if len(self.data) > 0:
            first_item = self.data[0]
            required_keys = ['text', 'section_type']
            missing_keys = [key for key in required_keys if key not in first_item]
            if missing_keys:
                logger.warning(f"Missing required keys in section classification data: {missing_keys}")
                # Fix data if possible
                self._fix_data_schema()
        
        # Create label map
        self.label_map = MODEL_CONFIGS["section_classifier"]["label2id"]
        
        logger.info(f"Loaded {len(self.data)} examples for section classification from {self.data_path}")
    
    def _fix_data_schema(self):
        """Fix data schema issues if possible."""
        fixed_items = 0
        for i, item in enumerate(self.data):
            # Check for required keys
            if 'text' not in item:
                # Try to find text under different keys
                for possible_key in ['content', 'body', 'paragraph']:
                    if possible_key in item:
                        item['text'] = item[possible_key]
                        fixed_items += 1
                        break
                # If still no text, create empty text
                if 'text' not in item:
                    item['text'] = ""
                    logger.warning(f"Item {i} has no text content, using empty string")
            
            # Check for section_type
            if 'section_type' not in item:
                # Try to find section type under different keys
                for possible_key in ['type', 'section', 'category']:
                    if possible_key in item:
                        item['section_type'] = item[possible_key]
                        fixed_items += 1
                        break
                # If still no section_type, use default
                if 'section_type' not in item:
                    item['section_type'] = 'body'  # Default to body section
                    logger.warning(f"Item {i} has no section_type, using 'body'")
        
        if fixed_items > 0:
            logger.info(f"Fixed {fixed_items} items with schema issues in section classification data")
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an example from the dataset."""
        try:
            example = self.data[idx]
            text = example.get('text', '')
            section_type = example.get('section_type', 'body')  # Default to body if missing
            
            # Convert section type to label ID
            label_id = self.label_map.get(section_type, 0)
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Add label
            encoding['labels'] = torch.tensor(label_id, dtype=torch.long)
            
            return encoding
        
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(traceback.format_exc())
            
            # Return a default empty example
            default_encoding = self.tokenizer(
                "",
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Remove batch dimension
            default_encoding = {k: v.squeeze(0) for k, v in default_encoding.items()}
            
            # Add default label (first class)
            default_encoding['labels'] = torch.tensor(0, dtype=torch.long)
            
            return default_encoding

class ReferenceParsingDataset(Dataset):
    """Dataset for reference parsing."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """Initialize the dataset."""
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except UnicodeDecodeError:
            # Fallback to Latin-1 encoding if UTF-8 fails
            logger.warning(f"UTF-8 decoding failed for {self.data_path}, trying with ISO-8859-1 encoding")
            with open(self.data_path, 'r', encoding='ISO-8859-1') as f:
                self.data = json.load(f)
        
        # Convert list-based data to token format if needed
        if len(self.data) > 0 and 'tokens' not in self.data[0]:
            self._convert_to_token_format()
        
        # Create label map
        self.label_map = MODEL_CONFIGS["reference_parser"]["label2id"]
        
        logger.info(f"Loaded {len(self.data)} examples for reference parsing from {self.data_path}")
        
    def _convert_to_token_format(self):
        """Convert data to token format if it's in a different format."""
        token_format_data = []
        converted_count = 0
        skipped_count = 0
        labeled_count = 0
        
        for i, example in enumerate(self.data):
            # Check if example contains reference text
            if 'text' in example and 'structured' in example:
                text = example['text']
                structured = example.get('structured', {})
                
                # Skip examples with no structured fields
                if not structured.get('authors') and not structured.get('title') and not structured.get('year') and not structured.get('venue'):
                    skipped_count += 1
                    continue
                
                # Simple tokenization by whitespace
                tokens = text.split()
                
                # Default to 'O' labels for all tokens
                labels = ['O'] * len(tokens)
                
                # Extract structured information
                authors = structured.get('authors', [])
                title = structured.get('title', '')
                year = str(structured.get('year', ''))
                venue = structured.get('venue', '')
                
                # Map entities to tokens
                # This process aligns entities from the structured data to positions in the text
                entities_labeled = self._map_entities_to_tokens(tokens, labels, authors, title, year, venue)
                
                # Skip examples where no entities were successfully labeled
                if not entities_labeled or all(label == 'O' for label in labels):
                    skipped_count += 1
                    continue
                
                # Create token format example
                token_example = {
                    'id': i,
                    'tokens': tokens,
                    'labels': labels
                }
                
                # Copy metadata if present
                if 'metadata' in example:
                    token_example['metadata'] = example['metadata']
                
                token_format_data.append(token_example)
                converted_count += 1
                labeled_count += 1
            else:
                logger.warning(f"Example {i} does not contain required fields ('text' and 'structured'), skipping")
                skipped_count += 1
        
        # Log statistics
        if token_format_data:
            logger.info(f"Converted {converted_count} examples to token format")
            logger.info(f"Skipped {skipped_count} examples with no structured data or no entity matches")
            logger.info(f"Retained {labeled_count} examples with at least one labeled entity")
            self.data = token_format_data
            
            # Log label distribution
            self._log_label_distribution()
            
        else:
            logger.error("No valid examples could be converted to token format")
    
    def _map_entities_to_tokens(self, tokens, labels, authors, title, year, venue):
        """Map structured entities to token-level labels with improved fuzzy matching."""
        import re
        from difflib import SequenceMatcher
        import string
        
        # Convert tokens to lowercase for case-insensitive matching
        tokens_lower = [token.lower() for token in tokens]
        
        # Remove punctuation from tokens for more flexible matching
        translator = str.maketrans('', '', string.punctuation)
        tokens_clean = [token.lower().translate(translator) for token in tokens]
        
        # Track spans to avoid overlapping entity labels
        labeled_spans = []
        
        # Track if any entities were successfully labeled
        any_entity_labeled = False
        
        def normalize_text(text):
            """Normalize text by removing punctuation, extra spaces, and converting to lowercase."""
            if not text or not isinstance(text, str):
                return ""
            text = text.lower()
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
        
        def find_best_window_match(entity_words, min_similarity=0.75):
            """Find the best window match for a multi-word entity."""
            best_score = 0
            best_position = -1
            best_length = 0
            
            # Original entity for better matching
            original_entity = ' '.join(entity_words)
            
            # Try different window sizes (allow some words to be missing)
            min_words = max(1, len(entity_words) - 1)  # Allow missing 1 word
            
            for window_size in range(min_words, min(len(tokens), len(entity_words) + 3) + 1):
                for start in range(len(tokens) - window_size + 1):
                    # Skip if any part of this span is already labeled
                    if any(overlap(start, start + window_size - 1, s_start, s_end) for s_start, s_end, _ in labeled_spans):
                        continue
                    
                    # Get window text
                    window_tokens = tokens[start:start + window_size]
                    window_text = ' '.join(window_tokens).lower()
                    
                    # Try matching against original entity
                    similarity = get_similarity(original_entity, window_text)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_position = start
                        best_length = window_size
            
            if best_score >= min_similarity:
                return best_position, best_length, best_score
            return -1, 0, 0
        
        def label_entity(entity, entity_type, min_similarity=0.75):
            """Label tokens matching an entity with fuzzy matching.
            Returns True if successfully labeled any tokens."""
            if not entity:
                return False
                    
            # For handling entities that can be either strings or lists
            entity_items = [entity] if isinstance(entity, str) else entity
            
            labeled_any = False
            for item in entity_items:
                # Skip empty items
                if not item or not isinstance(item, str) or not item.strip():
                    continue
                
                # Clean and tokenize entity
                clean_item = normalize_text(item)
                entity_words = clean_item.split()
                
                if not entity_words:
                    continue
                
                # Special case for years - exact/near-exact matching only
                if entity_type == "YEAR":
                    for i, token in enumerate(tokens):
                        if token == str(item) or clean_item == tokens_clean[i]:
                            if not any(i >= start and i <= end for start, end, _ in labeled_spans):
                                labels[i] = entity_type
                                labeled_spans.append((i, i, entity_type))
                                labeled_any = True
                                nonlocal any_entity_labeled
                                any_entity_labeled = True
                    continue
                        
                # For single-word entities, try direct matching with similarity
                if len(entity_words) == 1:
                    for i, token in enumerate(tokens_clean):
                        similarity = get_similarity(entity_words[0], token)
                        # Use a higher threshold for single words to reduce false positives
                        if similarity >= 0.85 and not any(i >= start and i <= end for start, end, _ in labeled_spans):
                            labels[i] = entity_type
                            labeled_spans.append((i, i, entity_type))
                            labeled_any = True
                            any_entity_labeled = True
                    continue
                
                # For multi-word entities, use window matching
                best_pos, best_len, score = find_best_window_match(entity_words, min_similarity)
                
                if best_pos >= 0:
                    # Label all tokens in the matched span
                    for j in range(best_len):
                        if best_pos + j < len(labels):
                            labels[best_pos + j] = entity_type
                    
                    # Remember this span is labeled
                    labeled_spans.append((best_pos, best_pos + best_len - 1, entity_type))
                    labeled_any = True
                    any_entity_labeled = True
                else:
                    # If no good window match, try token-by-token matching for important words
                    # Filter out short/common words
                    important_words = [w for w in entity_words if len(w) > 3]
                    if important_words:
                        for word in important_words:
                            for i, token in enumerate(tokens_clean):
                                similarity = get_similarity(word, token)
                                if similarity >= 0.9 and not any(i >= start and i <= end for start, end, _ in labeled_spans):
                                    labels[i] = entity_type
                                    labeled_spans.append((i, i, entity_type))
                                    labeled_any = True
                                    any_entity_labeled = True
            
            return labeled_any
        
        def overlap(start1, end1, start2, end2):
            """Check if two spans overlap."""
            return not (end1 < start2 or start1 > end2)
        
        # Order of matching priority: YEAR (precise), AUTHORS, VENUE, TITLE (more context needed)
        # Year (should be a number)
        year_labeled = label_entity(year, "YEAR", min_similarity=0.9)  # Higher threshold for years
        
        # Map authors (may be multiple)
        authors_labeled = label_entity(authors, "AUTHOR", min_similarity=0.8)
        
        # Venue (journal/conference name)
        venue_labeled = label_entity(venue, "VENUE", min_similarity=0.75)
        
        # Then title (usually one string, needs more context)
        title_labeled = label_entity(title, "TITLE", min_similarity=0.75)
        
        return any_entity_labeled
    
    def _log_label_distribution(self):
        """Log distribution of labels in the dataset."""
        label_counts = {}
        total_labels = 0
        
        for example in self.data:
            for label in example.get('labels', []):
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
                total_labels += 1
        
        # Log counts
        logger.info(f"Label distribution in dataset (total {total_labels} labels):")
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_labels) * 100
            logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    def _fix_data_schema(self):
        """Fix data schema issues if possible."""
        fixed_items = 0
        for i, item in enumerate(self.data):
            # Check for required keys
            required_keys = ['tokens', 'labels']
            missing_keys = [key for key in required_keys if key not in item]
            
            if missing_keys:
                logger.warning(f"Example {i} missing keys: {missing_keys}")
                
                # If missing tokens but has text, tokenize it
                if 'tokens' not in item and 'text' in item:
                    item['tokens'] = item['text'].split()
                    fixed_items += 1
                
                # If missing labels but has tokens, create default labels
                if 'labels' not in item and 'tokens' in item:
                    item['labels'] = ['O'] * len(item['tokens'])
                    fixed_items += 1
                
                # If still missing required keys, set to empty lists
                for key in missing_keys:
                    if key not in item:
                        item[key] = []
                        logger.warning(f"Item {i} still missing {key}, using empty list")
        
        if fixed_items > 0:
            logger.info(f"Fixed {fixed_items} items with schema issues in reference parsing data")
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an example from the dataset."""
        try:
            example = self.data[idx]
            tokens = example.get('tokens', [])
            labels = example.get('labels', ['O'] * len(tokens))
            
            # Ensure tokens and labels have the same length
            min_len = min(len(tokens), len(labels))
            tokens = tokens[:min_len]
            labels = labels[:min_len]
            
            # Explicitly define label map to ensure consistent IDs
            self.label_map = {
                "O": 0,
                "AUTHOR": 1,
                "TITLE": 2,
                "YEAR": 3,
                "VENUE": 4
            }
            
            # Convert labels to IDs
            label_ids = [self.label_map.get(label, 0) for label in labels]
            
            # Validate label IDs to prevent CUDA errors
            num_classes = len(self.label_map)
            for idx, lid in enumerate(label_ids):
                if lid < 0 or lid >= num_classes:
                    logger.warning(f"[Fixed Invalid Label] Example {idx}: Invalid label ID {lid}, replacing with 0")
                    label_ids[idx] = 0  # Replace with 'O' (safe default)
            
            # Handle empty examples
            if not tokens:
                return self._create_empty_encoding()
            
            # Tokenize each token separately to maintain alignment
            encoded_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
            word_ids_mapping = []  # to track which subword belongs to which word
            
            # Start with [CLS] token
            encoded_inputs['input_ids'].append(self.tokenizer.cls_token_id)
            encoded_inputs['attention_mask'].append(1)
            encoded_inputs['token_type_ids'].append(0)
            word_ids_mapping.append(None)  # [CLS] isn't tied to any word
            
            # Process each token
            for word_idx, word in enumerate(tokens):
                # Skip empty tokens
                if not word.strip():
                    continue
                    
                # Tokenize this single word
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    # Skip if tokenizer returned empty
                    continue
                
                # Get token IDs
                token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                
                # Add to our encoded inputs
                encoded_inputs['input_ids'].extend(token_ids)
                encoded_inputs['attention_mask'].extend([1] * len(token_ids))
                encoded_inputs['token_type_ids'].extend([0] * len(token_ids))
                
                # Map each subword to its original word
                for _ in range(len(token_ids)):
                    word_ids_mapping.append(word_idx)
            
            # Add [SEP] token
            encoded_inputs['input_ids'].append(self.tokenizer.sep_token_id)
            encoded_inputs['attention_mask'].append(1)
            encoded_inputs['token_type_ids'].append(0)
            word_ids_mapping.append(None)  # [SEP] isn't tied to any word
            
            # Truncate or pad sequences to max_length
            if len(encoded_inputs['input_ids']) > self.max_length:
                # Truncate
                for key in encoded_inputs:
                    encoded_inputs[key] = encoded_inputs[key][:self.max_length]
                word_ids_mapping = word_ids_mapping[:self.max_length]
                
                # Ensure [SEP] token is at the end after truncation
                encoded_inputs['input_ids'][-1] = self.tokenizer.sep_token_id
            else:
                # Pad
                padding_length = self.max_length - len(encoded_inputs['input_ids'])
                
                encoded_inputs['input_ids'].extend([self.tokenizer.pad_token_id] * padding_length)
                encoded_inputs['attention_mask'].extend([0] * padding_length)
                encoded_inputs['token_type_ids'].extend([0] * padding_length)
                word_ids_mapping.extend([None] * padding_length)
            
            # Convert to tensors
            encoded_inputs = {k: torch.tensor(v, dtype=torch.long) for k, v in encoded_inputs.items()}
            
            # Create labels tensor - default all to -100 (ignored in loss calculation)
            labels_tensor = torch.ones(self.max_length, dtype=torch.long) * -100
            
            # Assign labels based on word_ids_mapping
            for i, word_idx in enumerate(word_ids_mapping):
                if word_idx is not None and word_idx < len(label_ids):
                    # Assign valid label ID, ensure it's within bounds
                    labels_tensor[i] = label_ids[word_idx]
            
            # Add labels to encoded inputs
            encoded_inputs['labels'] = labels_tensor
            
            return encoded_inputs
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_encoding()
    
    def _create_empty_encoding(self):
        """Create an empty encoding for fallback."""
        return {
            'input_ids': torch.zeros((1, self.max_length), dtype=torch.long),
            'attention_mask': torch.zeros((1, self.max_length), dtype=torch.long),
            'token_type_ids': torch.zeros((1, self.max_length), dtype=torch.long),
            'labels': torch.ones((1, self.max_length), dtype=torch.long) * -100
        }
    
    def get_predictions(self, outputs, batch):
        """Get predictions from model outputs."""
        try:
            # Check that outputs has logits
            if not hasattr(outputs, 'logits'):
                logger.warning("Model outputs don't contain logits")
                return None, None
                
            # Get predicted token classes (ignore padding)
            logits = outputs.logits
            
            # Validate logits shape
            if len(logits.shape) != 3:
                logger.warning(f"Unexpected logits shape: {logits.shape}, expected 3 dimensions")
                return None, None
                
            # Get predictions safely
            with torch.no_grad():
                # Move to CPU for safer operations
                logits_cpu = logits.detach().cpu()
                
                # Get argmax along last dimension
                try:
                    preds = torch.argmax(logits_cpu, dim=2)
                except Exception as e:
                    logger.error(f"Error calculating argmax: {e}")
                    return None, None
            
            # Check for labels in batch
            if 'labels' not in batch:
                logger.warning("No labels in batch")
                return preds, None
                
            # Return predictions and labels
            return preds, batch['labels']
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            logger.error(traceback.format_exc())
            # Return None values as fallback
            return None, None
    
    def calculate_metrics(self, preds, labels):
        """Calculate metrics for token classification."""
        try:
            # Ensure inputs are tensors but handle the case where they might already be tensors
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, device=self.device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
                
            # Ensure no invalid values before flattening
            num_classes = len(MODEL_CONFIGS["reference_parser"]["id2label"])
            # Fix any invalid prediction values  
            preds = torch.clamp(preds, min=0, max=num_classes-1)
            
            # Flatten the arrays and convert to CPU numpy arrays BEFORE any indexing
            preds_flat = preds.flatten().cpu().numpy()
            labels_flat = labels.flatten().cpu().numpy()
            
            # Filter out ignored index (-100)
            mask = labels_flat != -100
            preds_filtered = preds_flat[mask]
            labels_filtered = labels_flat[mask]
            
            # Ensure all labels are valid and non-negative
            mask_valid_labels = (labels_filtered >= 0) & (labels_filtered < num_classes)
            if not mask_valid_labels.all():
                logger.warning(f"Found {(~mask_valid_labels).sum()} invalid label indices. Filtering out.")
                preds_filtered = preds_filtered[mask_valid_labels]
                labels_filtered = labels_filtered[mask_valid_labels]
            
            if len(labels_filtered) == 0:
                logger.warning("No valid labels for metric calculation")
                return {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'entity_f1': 0.0
                }
            
            # Calculate standard metrics
            accuracy = accuracy_score(labels_filtered, preds_filtered)
            f1 = f1_score(labels_filtered, preds_filtered, average='weighted')
            precision = precision_score(labels_filtered, preds_filtered, average='weighted', zero_division=0)
            recall = recall_score(labels_filtered, preds_filtered, average='weighted', zero_division=0)
            
            # Calculate per-class metrics
            class_f1s = f1_score(labels_filtered, preds_filtered, average=None, zero_division=0)
            
            # Calculate entity-level F1 score
            entity_f1 = self._calculate_entity_f1(preds, labels)
            
            metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'entity_f1': entity_f1
            }
            
            # Add per-class metrics for each reference element type
            for i, label_name in enumerate(REF_LABEL_TYPES):
                if i < len(class_f1s):
                    metrics[f'f1_{label_name}'] = class_f1s[i]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logger.error(traceback.format_exc())
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'entity_f1': 0.0
            }
    
    def _calculate_entity_f1(self, preds, labels):
        """Calculate entity-level F1 score."""
        try:
            # Ensure inputs are tensors
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, device=self.device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
                
            # Ensure no invalid values
            num_classes = len(MODEL_CONFIGS["reference_parser"]["id2label"])
            preds = torch.clamp(preds, min=0, max=num_classes-1)
            
            # Convert tensor predictions to CPU numpy arrays
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Lists to store all entities
            true_entities = []
            pred_entities = []
            
            # Process each sequence in the batch
            for i in range(preds_np.shape[0]):
                # Filter out invalid label indices
                valid_indices = labels_np[i] != -100
                seq_true_labels = labels_np[i][valid_indices]
                seq_pred_labels = preds_np[i][valid_indices]
                
                if len(seq_true_labels) == 0:
                    continue
                    
                true_entities_seq = self._extract_entities(seq_true_labels)
                pred_entities_seq = self._extract_entities(seq_pred_labels)
                
                true_entities.extend(true_entities_seq)
                pred_entities.extend(pred_entities_seq)
            
            # Calculate entity-level metrics
            if not true_entities and not pred_entities:
                return 1.0  # Perfect score if no entities in ground truth and predictions
            
            if not true_entities or not pred_entities:
                return 0.0  # Zero score if either ground truth or predictions are empty
            
            # Count true positives, false positives, and false negatives
            tp = len([entity for entity in pred_entities if entity in true_entities])
            fp = len([entity for entity in pred_entities if entity not in true_entities])
            fn = len([entity for entity in true_entities if entity not in pred_entities])
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            # Log entity-level metrics for debugging
            if tp + fp + fn > 0:
                logger.info(f"Entity-level metrics - TP: {tp}, FP: {fp}, FN: {fn}")
                logger.info(f"Entity-level precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Log unique entities found
                unique_pred_entity_types = set(entity[0] for entity in pred_entities)
                unique_true_entity_types = set(entity[0] for entity in true_entities)
                logger.info(f"Predicted entity types: {unique_pred_entity_types}")
                logger.info(f"True entity types: {unique_true_entity_types}")
            
            return f1
            
        except Exception as e:
            logger.error(f"Error calculating entity F1: {e}")
            logger.error(traceback.format_exc())
            return 0.0
    
    def _extract_entities(self, sequence):
        """Extract entities from a sequence of token labels."""
        entities = []
        current_entity = None
        
        # Get a map of ID to label name
        id2label = MODEL_CONFIGS["reference_parser"]["id2label"]
        num_labels = len(id2label)
        
        # Process tokens, ignoring padding (-100)
        for i, label_id in enumerate(sequence):
            # Skip padding or invalid labels
            if label_id == -100 or label_id < 0 or label_id >= num_labels:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Convert ID to label safely
            label = id2label.get(int(label_id), "O")
            
            # Skip 'O' (outside) tokens
            if label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Start a new entity or continue the current one
            if current_entity is None or current_entity[0] != label:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = (label, i, i)
            else:
                # Extend the current entity
                current_entity = (label, current_entity[1], i)
        
        # Add the last entity if there's one
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities


# ======== TRAINER CLASSES ========
class BaseTrainer:
    """Base trainer class for all models."""
    
    def __init__(
        self,
        model_type,
        model_name=None,
        model_dir=None,
        freeze_base_model=True,
        learning_rate=2e-5,
        batch_size=16,
        max_length=512,
        device=None
    ):
        """Initialize trainer."""
        self.model_type = model_type
        self.freeze_base_model = freeze_base_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device if device is not None else DEVICE
        
        # Set model paths
        if model_dir is None:
            model_dir = f"models/{model_type}"
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model-specific configs
        self._init_config()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._init_model()
        
        # Initialize training stats
        self.training_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def _init_config(self):
        """Initialize model-specific configurations."""
        self.model_config = MODEL_CONFIGS[self.model_type]
        self.model_name = self.model_config["model_name"]
    
    def _init_model(self):
        """Initialize the model. Implemented by subclasses."""
        raise NotImplementedError()
    
    def train(
        self,
        train_data_path,
        val_data_path,
        epochs=5,
        warmup_steps=0,
        weight_decay=0.01,
        early_stopping_patience=3,
        save_best_model=True,
        use_lr_scheduler=False,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        gradient_clip=1.0,
        label_smoothing=0.0,
        show_examples=False  # New parameter to control example display
    ):
        """Train the model."""
        try:
            # Get data loaders
            train_loader = self.get_dataloader(train_data_path, shuffle=True)
            val_loader = self.get_dataloader(val_data_path, shuffle=False)
            
            if train_loader is None or val_loader is None:
                logger.error("Failed to create data loaders. Check data paths and formats.")
                return self.training_stats
            
            # Verify model and loaders are compatible
            sample_batch = next(iter(train_loader))
            if not isinstance(sample_batch, dict) or 'input_ids' not in sample_batch:
                logger.error("Invalid batch format from data loader. Check data preparation.")
                return self.training_stats
            
            # Confirm tensor sizes and content
            input_tensor = sample_batch.get('input_ids')
            logger.info(f"Sample batch shape: {input_tensor.shape}")
            
            # Check label validity if present
            if 'labels' in sample_batch:
                labels = sample_batch['labels']
                # Check for negative labels (except -100 which is the ignore index)
                if ((labels < 0) & (labels != -100)).any():
                    logger.warning("Found invalid negative labels in training data. Will replace with ignore_index.")
                
                # Count label distribution in sample
                unique_labels = torch.unique(labels[labels != -100])
                logger.info(f"Unique label values in sample: {unique_labels.tolist()}")
            
            # Prepare optimizer and scheduler
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
            
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            # Add ReduceLROnPlateau scheduler if requested
            lr_scheduler = None
            if use_lr_scheduler:
                lr_scheduler = ReduceLROnPlateau(
                    optimizer, 
                    mode='max', 
                    factor=lr_scheduler_factor,
                    patience=lr_scheduler_patience,
                    verbose=True
                )
                logger.info(f"Using ReduceLROnPlateau learning rate scheduler with factor {lr_scheduler_factor} and patience {lr_scheduler_patience}")
            
            # Training loop
            best_val_metric = 0.0
            best_epoch = 0
            patience_counter = 0
            
            logger.info(f"Starting training for {epochs} epochs with {len(train_loader)} batches per epoch")
            logger.info(f"Early stopping patience: {early_stopping_patience}")
            
            for epoch in range(epochs):
                try:
                    start_time = time.time()
                    
                    # Train
                    self.model.train()
                    train_loss = 0.0
                    
                    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
                    for batch_idx, batch in enumerate(train_progress):
                        try:
                            # Handle negative labels before moving to device
                            if 'labels' in batch:
                                labels = batch['labels']
                                if ((labels < 0) & (labels != -100)).any():
                                    # Replace invalid negative labels with -100 (ignore_index)
                                    batch['labels'] = torch.where(
                                        (labels < 0) & (labels != -100),
                                        torch.tensor(-100, device=labels.device),
                                        labels
                                    )
                            
                            # Move batch to device
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            
                            # Forward pass
                            outputs = self.model(**batch)
                            
                            # For reference parser with weighted loss
                            if hasattr(self, 'use_weighted_loss') and self.use_weighted_loss and hasattr(self, '_compute_weighted_loss'):
                                loss = self._compute_weighted_loss(outputs, batch['labels'])
                            else:
                                loss = outputs.loss
                            
                            if label_smoothing > 0:
                                # Apply label smoothing to reduce confidence and prevent overfitting
                                if hasattr(outputs, "logits"):
                                    if hasattr(self, "model_type") and self.model_type == "reference_parser":
                                        # For token classification, apply label smoothing manually
                                        logits = outputs.logits.view(-1, outputs.logits.size(-1))
                                        labels = batch['labels'].view(-1)
                                        
                                        # Filter out ignore index
                                        valid_indices = labels != -100
                                        if valid_indices.any():
                                            valid_logits = logits[valid_indices]
                                            valid_labels = labels[valid_indices]
                                            
                                            # Create one-hot encodings for valid labels
                                            num_classes = valid_logits.size(-1)
                                            one_hot = torch.zeros_like(valid_logits).scatter_(
                                                1, valid_labels.unsqueeze(1), 1.0
                                            )
                                            
                                            # Apply smoothing
                                            smoothed = one_hot * (1 - label_smoothing) + \
                                                      label_smoothing / num_classes
                                            
                                            # Cross entropy loss with smoothed targets
                                            log_probs = F.log_softmax(valid_logits, dim=-1)
                                            loss = -torch.sum(smoothed * log_probs) / valid_indices.sum()
                            
                            # Backward pass
                            optimizer.zero_grad()
                            loss.backward()
                            
                            # Apply gradient clipping
                            if gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                            
                            optimizer.step()
                            scheduler.step()
                            
                            # Update metrics
                            train_loss += loss.item()
                            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
                        except RuntimeError as e:
                            # Handle CUDA out of memory or other runtime errors
                            if "CUDA out of memory" in str(e):
                                logger.error(f"CUDA out of memory in batch {batch_idx}. Consider reducing batch size.")
                                # Skip this batch and continue
                                continue
                            elif "cuDNN error" in str(e):
                                logger.error(f"cuDNN error in batch {batch_idx}. Skipping batch.")
                                continue
                            elif "must be non-negative" in str(e):
                                logger.error(f"Class values must be non-negative error in batch {batch_idx}. Skipping batch.")
                                if 'labels' in batch:
                                    labels = batch['labels']
                                    unique_labels = torch.unique(labels).tolist()
                                    logger.error(f"Unique label values: {unique_labels}")
                                continue
                            else:
                                # For other runtime errors, just warn and continue
                                logger.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                                continue
                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                            continue
                    
                    # Calculate average training loss
                    avg_train_loss = train_loss / len(train_loader)
                    self.training_stats['train_loss'].append(avg_train_loss)
                    
                    # Evaluate
                    val_loss, val_metrics = self.evaluate(val_loader)
                    self.training_stats['val_loss'].append(val_loss)
                    self.training_stats['val_metrics'].append(val_metrics)
                    
                    # Track learning rate
                    self.training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
                    
                    # Calculate primary metric for early stopping
                    primary_metric = val_metrics.get('f1', val_metrics.get('accuracy', 0.0))
                    
                    # Update learning rate if using ReduceLROnPlateau
                    if use_lr_scheduler and lr_scheduler is not None:
                        lr_scheduler.step(primary_metric)
                    
                    # Print epoch summary
                    epoch_time = time.time() - start_time
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                                f"Train Loss: {avg_train_loss:.4f}, "
                                f"Val Loss: {val_loss:.4f}, "
                                f"Val Metrics: {val_metrics}, "
                                f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
                                f"Time: {epoch_time:.2f}s")
                    
                    # Check for improvement
                    if primary_metric > best_val_metric:
                        logger.info(f"Validation metric improved from {best_val_metric:.4f} to {primary_metric:.4f}")
                        best_val_metric = primary_metric
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best model
                        if save_best_model:
                            self.save_model()
                    else:
                        patience_counter += 1
                        logger.info(f"Validation metric did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                        
                        # Early stopping
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
                except Exception as e:
                    logger.error(f"Error in epoch {epoch+1}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Save checkpoint if possible
                    try:
                        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch+1}"
                        self.save_model(path=checkpoint_path)
                        logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
                    except Exception as save_error:
                        logger.error(f"Failed to save checkpoint: {str(save_error)}")
                    
                    # Continue to next epoch
                    continue
            
            # Final summary
            logger.info(f"Training complete. Best validation metric: {best_val_metric:.4f} at epoch {best_epoch+1}")
            
            # Show example predictions on validation set if requested
            if show_examples:
                try:
                    self.show_example_predictions(val_loader)
                except Exception as e:
                    logger.error(f"Error showing examples: {str(e)}")
            
            return self.training_stats
        except Exception as e:
            logger.error(f"Fatal error in training process: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return partial training stats if available
            return self.training_stats
    
    def evaluate(self, val_loader):
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        # Keep track of successful batches
        successful_batches = 0
        failed_batches = 0
        
        # Important: Don't try to move the model between devices during evaluation
        # This causes CUDA errors with existing tensors
        model_device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
                try:
                    # Copy batch to CPU first to ensure clean tensors
                    cpu_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            cpu_batch[k] = v.detach().clone().cpu()
                        else:
                            cpu_batch[k] = v
                    
                    # First try on CPU to validate the tensors
                    # Separate labels from input to avoid loss computation errors
                    labels = None
                    model_inputs = {}
                    for k, v in cpu_batch.items():
                        if k == 'labels':
                            labels = v
                        else:
                            model_inputs[k] = v
                    
                    try:
                        # Move batch to same device as model
                        device_inputs = {}
                        for k, v in model_inputs.items():
                            if isinstance(v, torch.Tensor):
                                device_inputs[k] = v.to(model_device)
                            else:
                                device_inputs[k] = v
                        
                        device_labels = labels.to(model_device) if isinstance(labels, torch.Tensor) else labels
                        
                        # Forward pass using the model's current device
                        outputs = self.model(**device_inputs)
                        
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning(f"CUDA error in batch {batch_idx}. Skipping this batch.")
                            failed_batches += 1
                            continue
                        else:
                            raise
                    
                    # Calculate loss separately if we have labels
                    if device_labels is not None and hasattr(outputs, 'logits'):
                        try:
                            # Use weighted loss if enabled
                            if self.use_weighted_loss and self.class_weights is not None:
                                weights = self.class_weights.to(model_device)
                                loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                            else:
                                # Otherwise use standard cross entropy
                                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                            
                            # Reshape tensors for loss calculation, ensuring we don't exceed valid indices
                            logits = outputs.logits.view(-1, outputs.logits.size(-1))
                            target = device_labels.view(-1)
                            
                            # Check for invalid indices
                            max_idx = logits.size(1) - 1
                            valid_indices = (target >= -100) & (target <= max_idx)
                            if not valid_indices.all():
                                # Filter out invalid indices (replace with -100 which is ignored)
                                target = torch.where(valid_indices, target, torch.tensor(-100, device=target.device))
                            
                            loss = loss_fct(logits, target)
                            
                            if loss is not None:  # Ensure loss is not None before calling item()
                                total_loss += loss.item()
                                num_batches += 1
                        except RuntimeError as e:
                            logger.warning(f"Error calculating loss in batch {batch_idx}: {str(e)}")
                            # Continue without adding loss for this batch
                    
                    # Get predictions
                    batch_for_preds = device_inputs.copy()
                    if device_labels is not None:
                        batch_for_preds['labels'] = device_labels
                    
                    try:
                        preds, actual_labels = self.get_predictions(outputs, batch_for_preds)
                        
                        # Only process valid predictions and labels
                        if preds is not None and actual_labels is not None:
                            # Move to CPU for safe storage
                            preds_cpu = preds.detach().cpu() if isinstance(preds, torch.Tensor) else torch.tensor([])
                            labels_cpu = actual_labels.detach().cpu() if isinstance(actual_labels, torch.Tensor) else torch.tensor([])
                            
                            # Only add valid tensors
                            if preds_cpu.numel() > 0 and labels_cpu.numel() > 0:
                                all_preds.append(preds_cpu)
                                all_labels.append(labels_cpu)
                                successful_batches += 1
                    except Exception as e:
                        logger.warning(f"Error getting predictions in batch {batch_idx}: {str(e)}")
                        failed_batches += 1
                
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    logger.error(traceback.format_exc())
                    failed_batches += 1
                    continue
        
        # Compute average loss
        avg_loss = total_loss / max(1, num_batches)
        
        # Log success rate
        logger.info(f"Evaluation completed with {successful_batches} successful batches, {failed_batches} failed batches")
        
        # Compute metrics if we have predictions and labels
        metrics = {}
        if all_preds and all_labels:
            try:
                # Safely concatenate batched predictions and labels
                preds_concat = torch.cat(all_preds, dim=0) 
                labels_concat = torch.cat(all_labels, dim=0)
                
                # Log prediction distribution
                self._log_prediction_distribution(preds_concat, labels_concat)
                
                metrics = self.calculate_metrics(preds_concat, labels_concat)
            except Exception as e:
                logger.error(f"Error calculating final metrics: {e}")
                logger.error(traceback.format_exc())
                metrics = {
                    'accuracy': 0,
                    'f1': 0, 
                    'precision': 0,
                    'recall': 0,
                    'entity_f1': 0
                }
        
        logger.info(f"Validation complete - Loss: {avg_loss:.4f}, Metrics: {metrics}")
        return avg_loss, metrics
    
    def get_predictions(self, outputs, batch):
        """Get predictions from model outputs. Implemented by subclasses."""
        raise NotImplementedError()
    
    def calculate_metrics(self, preds, labels):
        """Calculate metrics from predictions and labels. Implemented by subclasses."""
        raise NotImplementedError()
    
    def get_dataloader(self, data_path, shuffle=True):
        """Get data loader for the dataset. Implemented by subclasses."""
        raise NotImplementedError()
    
    def save_model(self, path=None):
        """Save the model."""
        path = path or self.model_dir / "best_model"
        # Save model
        self.model.save_pretrained(path)
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        # Save training stats
        with open(path / "training_stats.json", "w") as f:
            json.dump(self.training_stats, f)
        
        logger.info(f"Model saved to {path}")
    
    def show_example_predictions(self, data_loader):
        """Show example predictions from the model."""
        try:
            # Get a batch of data
            iterator = iter(data_loader)
            example_batch = next(iterator)
            
            # Additional error checking to ensure the batch is valid
            if not example_batch or not isinstance(example_batch, dict):
                logger.warning("Invalid batch format, cannot show example predictions")
                return
                
            # Check that required tensors exist
            required_keys = ['input_ids', 'attention_mask', 'labels']
            for key in required_keys:
                if key not in example_batch:
                    logger.warning(f"Batch missing required key: {key}, cannot show example predictions")
                    return
            
            try:
                # Get model's current device
                model_device = next(self.model.parameters()).device
                
                # Ensure model is in eval mode
                self.model.eval()
                
                # Create a small subset for example predictions to avoid memory issues
                small_batch = {}
                for k, v in example_batch.items():
                    if isinstance(v, torch.Tensor):
                        # Just take the first example
                        small_batch[k] = v[:1].to(model_device)
                    else:
                        small_batch[k] = v
                
                # Forward pass with error handling
                with torch.no_grad():
                    try:
                        # Run model without labels to avoid loss computation errors
                        model_inputs = {k: v for k, v in small_batch.items() if k != 'labels'}
                        outputs = self.model(**model_inputs)
                        
                        # Get predictions safely
                        preds, labels = self.get_predictions(outputs, small_batch)
                        
                        # Handle case where predictions are empty
                        if preds is None or labels is None:
                            logger.warning("Empty predictions or labels, cannot show example predictions")
                            return
                        
                        # Convert to numpy safely and ensure arrays are not empty
                        if isinstance(preds, torch.Tensor) and preds.numel() > 0:
                            preds_np = preds.detach().cpu().numpy()
                        else:
                            logger.warning("Empty predictions tensor")
                            return
                            
                        if 'labels' in small_batch and isinstance(small_batch['labels'], torch.Tensor) and small_batch['labels'].numel() > 0:
                            labels_np = small_batch['labels'].detach().cpu().numpy()
                        else:
                            logger.warning("Empty labels tensor")
                            return
                        
                        # Show the single example
                        logger.info(f"Example prediction:")
                        
                        # Get tokens from input IDs
                        tokens = self.tokenizer.convert_ids_to_tokens(small_batch['input_ids'][0].cpu().numpy())
                        
                        # Handle different output shapes for different model types
                        if len(preds_np.shape) == 1:  # Classification
                            pred_label = MODEL_CONFIGS[self.model_type]["id2label"].get(preds_np[0], "Unknown")
                            true_label = MODEL_CONFIGS[self.model_type]["id2label"].get(labels_np[0], "Unknown") if labels_np[0] != -100 else "PAD"
                            logger.info(f"Text: {''.join(tokens)}")
                            logger.info(f"Predicted: {pred_label}, True: {true_label}")
                        
                        elif len(preds_np.shape) == 2:  # Token classification
                            # Convert IDs to label names
                            pred_labels = [MODEL_CONFIGS[self.model_type]["id2label"].get(p, "O") for p in preds_np[0]]
                            true_labels = [MODEL_CONFIGS[self.model_type]["id2label"].get(l, "O") if l != -100 else "PAD" for l in labels_np[0]]
                            
                            # Zip together and display, limiting to 50 tokens for readability
                            logger.info(f"{'Token':<15} | {'Prediction':<10} | {'True Label':<10}")
                            logger.info("-" * 50)
                            
                            for j, (token, pred, true) in enumerate(list(zip(tokens, pred_labels, true_labels))[:50]):
                                if true != "PAD" and token not in (self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token):
                                    logger.info(f"{token:<15} | {pred:<10} | {true:<10}")
                        
                        logger.info("-" * 50)
                    
                    except RuntimeError as e:
                        logger.error(f"RuntimeError during example predictions: {e}")
                        return
                    except Exception as e:
                        logger.error(f"Error during example prediction: {e}")
                        return
                        
            except Exception as e:
                logger.error(f"Error processing example batch: {str(e)}")
                logger.error(traceback.format_exc())
                return
                
        except StopIteration:
            logger.warning("No examples available in the dataloader")
        except Exception as e:
            logger.error(f"Error showing example predictions: {e}")
            logger.error(traceback.format_exc())


class SectionClassifierTrainer(BaseTrainer):
    """Trainer for section classification model."""
    
    def _init_model(self):
        """Initialize the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load config and model
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.model_config["num_labels"],
            id2label=self.model_config["id2label"],
            label2id=self.model_config["label2id"]
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=self.config
        )
        
        # Freeze base model layers if required
        if self.freeze_base_model:
            logger.info("Freezing base model layers")
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        
        # Move model to device
        self.model.to(self.device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Initialized {self.model_name} with {trainable_params:,} trainable parameters out of {total_params:,} total")
    
    def get_dataloader(self, data_path, shuffle=True):
        """Get data loader for the dataset."""
        try:
            dataset = SectionClassificationDataset(
                data_path=data_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            
            # Check a sample to catch potential data format issues early
            sample = dataset[0]
            # Make sure 'labels' tensor exists and has sensible dimensions
            assert 'labels' in sample and isinstance(sample['labels'], torch.Tensor)
            assert len(sample['labels'].shape) == 0, f"Expected scalar tensor for labels, got shape {sample['labels'].shape}"
            
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0  # No multiprocessing for CPU
            )
        except Exception as e:
            logger.error(f"Error creating section classification dataloader: {str(e)}")
            raise
    
    def get_predictions(self, outputs, batch):
        """Get predictions from model outputs."""
        try:
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            return preds.tolist(), labels.tolist()
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            # Return empty lists as fallback
            return [], []
    
    def calculate_metrics(self, preds, labels):
        """Calculate metrics from predictions and labels for section classification."""
        if not preds or not labels:
            logger.warning("Empty predictions or labels array when calculating section metrics")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        try:
            # Convert numeric labels to class names for more informative logging
            label_map = {i: label for i, label in enumerate(SECTION_TYPES)}
            
            # Log some examples of predictions
            num_examples = min(3, len(preds))
            for i in range(num_examples):
                logger.debug(f"  Example {i}: Predicted {label_map[preds[i]]} (Expected {label_map[labels[i]]})")
            
            # Calculate overall metrics
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            precision = precision_score(labels, preds, average='weighted', zero_division=0)
            recall = recall_score(labels, preds, average='weighted', zero_division=0)
            
            # Calculate per-class metrics for the most important sections
            class_f1s = f1_score(labels, preds, average=None, zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
            # Add per-class metrics for key section types
            for i, section_type in enumerate(SECTION_TYPES):
                if i < len(class_f1s):
                    metrics[f'f1_{section_type}'] = class_f1s[i]
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating section metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }


class ReferenceParserTrainer(BaseTrainer):
    """Trainer class for reference parsing."""
    
    def __init__(
        self,
        model_type,
        model_name=None,
        model_dir=None,
        freeze_base_model=False,  # Changed to False to allow full fine-tuning by default
        learning_rate=2e-5,
        batch_size=16,
        max_length=512,
        device=None,
        use_weighted_loss=False,  # New parameter for weighted loss
        data_dir="data/transfer_learning/prepared"  # Added data_dir parameter for finding training data
    ):
        """Initialize trainer with additional parameters."""
        self.use_weighted_loss = use_weighted_loss
        self.class_weights = None  # Will be computed when training data is loaded
        self.data_dir = Path(data_dir)
        super().__init__(
            model_type=model_type,
            model_name=model_name,
            model_dir=model_dir,
            freeze_base_model=freeze_base_model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length,
            device=device
        )
        
        # If using weighted loss, precompute the weights from training data
        if self.use_weighted_loss:
            self._precompute_class_weights()
    
    def _precompute_class_weights(self):
        """Precompute class weights from training data."""
        try:
            # Locate training data
            train_data_path = self.data_dir / "reference_parsing" / "train" / "data.json"
            if not train_data_path.exists():
                logger.warning(f"Training data not found at {train_data_path}. Cannot compute class weights.")
                return
            
            logger.info(f"Computing class weights from training data at {train_data_path}")
            
            # Load training data
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["reference_parser"]["model_name"])
            dataset = ReferenceParsingDataset(train_data_path, tokenizer, self.max_length)
            
            # Check if data is properly labeled
            has_labels = False
            entity_examples = 0
            total_examples = len(dataset.data)
            
            for i in range(min(total_examples, len(dataset))):
                example = dataset.data[i]
                if 'labels' in example:
                    labels = example['labels']
                    if any(label != 'O' for label in labels):
                        has_labels = True
                        entity_examples += 1
            
            if not has_labels:
                logger.warning("Training data does not have proper entity labels, all labels are 'O'. Ensure structured data is mapped to tokens.")
            else:
                logger.info(f"Found {entity_examples} out of {total_examples} examples with entity labels")
            
            # Extract all labels from training data
            all_labels = []
            for i in range(len(dataset)):
                example = dataset.data[i]
                if 'labels' in example:
                    all_labels.extend(example['labels'])
            
            # Count frequency of each label
            label_counts = {}
            for label in all_labels:
                # Skip invalid labels
                if not label or label not in dataset.label_map:
                    continue
                    
                label_id = dataset.label_map.get(label, 0)  # Convert label to ID
                if label_id not in label_counts:
                    label_counts[label_id] = 0
                label_counts[label_id] += 1
            
            # Check which labels are missing from the data and log warnings
            missing_labels = []
            for label_name, label_id in MODEL_CONFIGS["reference_parser"]["label2id"].items():
                if label_id not in label_counts:
                    missing_labels.append(label_name)
                    logger.warning(f"Label '{label_name}' (ID: {label_id}) not found in training data.")
            
            if missing_labels:
                logger.warning(f"Missing labels in training data: {', '.join(missing_labels)}")
                
                # Use modified label map excluding missing labels
                modified_label_map = {}
                i = 0
                for label, _ in MODEL_CONFIGS["reference_parser"]["label2id"].items():
                    if label not in missing_labels:
                        modified_label_map[label] = i
                        i += 1
                
                if modified_label_map:
                    logger.info(f"Using modified label map: {modified_label_map}")
                    
                    # Ensure 'O' label is included at minimum
                    if 'O' not in modified_label_map:
                        modified_label_map['O'] = 0
                    
                    # Update label2id in model_config
                    MODEL_CONFIGS["reference_parser"]["label2id"] = modified_label_map
                    MODEL_CONFIGS["reference_parser"]["id2label"] = {v: k for k, v in modified_label_map.items()}
                    MODEL_CONFIGS["reference_parser"]["num_labels"] = len(modified_label_map)
                    
                    logger.info(f"Updated model config with {len(modified_label_map)} labels")
            
            # Compute class weights based on label counts
            self.class_weights = self._compute_class_weights(label_counts)
            
            # Log results
            self._log_class_distribution(label_counts, self.class_weights)
        except Exception as e:
            logger.error(f"Error precomputing class weights: {e}")
            logger.error(traceback.format_exc())
            # Set default weights
            logger.info("Using default class weights (equal weights)")
            num_labels = len(MODEL_CONFIGS["reference_parser"]["id2label"])
            self.class_weights = torch.ones(num_labels)
    
    def _log_class_distribution(self, label_counts, class_weights=None):
        """Log label distribution and weights for verification."""
        try:
            # Calculate total labels
            total_labels = sum(label_counts.values())
            
            # Create a nicely formatted table for display
            logger.info("=" * 60)
            logger.info("LABEL DISTRIBUTION AND CLASS WEIGHTS FROM TRAINING DATA")
            logger.info("=" * 60)
            logger.info(f"{'Label':<10} {'ID':<5} {'Count':<8} {'Percentage':<12} {'Weight':<8}")
            logger.info("-" * 60)
            
            # Sort by label ID for consistent output
            for label_id, count in sorted(label_counts.items()):
                label_name = MODEL_CONFIGS["reference_parser"]["id2label"].get(label_id, "Unknown")
                percentage = count / total_labels * 100
                
                # Get weight if available
                weight = "N/A"
                if class_weights is not None and 0 <= label_id < len(class_weights):
                    weight = f"{class_weights[label_id]:.4f}"
                
                logger.info(f"{label_name:<10} {label_id:<5} {count:<8} {percentage:.2f}%{' ':<8} {weight:<8}")
            
            logger.info("=" * 60)
            
            # Visualize the distribution if matplotlib is available
            try:
                # Prepare data for visualization
                labels = [MODEL_CONFIGS["reference_parser"]["id2label"].get(label_id, f"ID:{label_id}") 
                          for label_id in sorted(label_counts.keys())]
                counts = [label_counts[label_id] for label_id in sorted(label_counts.keys())]
                
                # Create bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(labels, counts)
                plt.title('Label Distribution in Training Data')
                plt.xlabel('Label')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.0f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save the figure
                os.makedirs('logs/figures', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plt.savefig(f'logs/figures/label_distribution_{timestamp}.png')
                logger.info(f"Label distribution visualization saved to logs/figures/label_distribution_{timestamp}.png")
                
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create label distribution visualization: {e}")
            
        except Exception as e:
            logger.error(f"Error logging class distribution: {e}")
    
    def _init_model(self):
        """Initialize the model for reference parsing."""
        model_config = MODEL_CONFIGS["reference_parser"]
        
        config = AutoConfig.from_pretrained(
            model_config["model_name"],
            num_labels=model_config["num_labels"],
            id2label=model_config["id2label"],
            label2id=model_config["label2id"]
        )
        
        # Add dropout for regularization to prevent overfitting
        config.hidden_dropout_prob = 0.3
        config.attention_probs_dropout_prob = 0.3
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_config["model_name"],
            config=config
        )
        
        # Freeze base model layers if required
        if self.freeze_base_model:
            logger.info("Freezing base model layers")
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        else:
            logger.info("Full fine-tuning enabled (all layers unfrozen)")
        
        # Move model to device
        self.model.to(self.device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Initialized {self.model_name} with {trainable_params:,} trainable parameters out of {total_params:,} total")

    def evaluate(self, val_loader):
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        # Keep track of successful batches
        successful_batches = 0
        failed_batches = 0
        
        # Important: Don't try to move the model between devices during evaluation
        # This causes CUDA errors with existing tensors
        model_device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
                try:
                    # Copy batch to CPU first to ensure clean tensors
                    cpu_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            cpu_batch[k] = v.detach().clone().cpu()
                        else:
                            cpu_batch[k] = v
                    
                    # First try on CPU to validate the tensors
                    # Separate labels from input to avoid loss computation errors
                    labels = None
                    model_inputs = {}
                    for k, v in cpu_batch.items():
                        if k == 'labels':
                            labels = v
                        else:
                            model_inputs[k] = v
                    
                    try:
                        # Move batch to same device as model
                        device_inputs = {}
                        for k, v in model_inputs.items():
                            if isinstance(v, torch.Tensor):
                                device_inputs[k] = v.to(model_device)
                            else:
                                device_inputs[k] = v
                        
                        device_labels = labels.to(model_device) if isinstance(labels, torch.Tensor) else labels
                        
                        # Forward pass using the model's current device
                        outputs = self.model(**device_inputs)
                        
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning(f"CUDA error in batch {batch_idx}. Skipping this batch.")
                            failed_batches += 1
                            continue
                        else:
                            raise
                    
                    # Calculate loss separately if we have labels
                    if device_labels is not None and hasattr(outputs, 'logits'):
                        try:
                            # Handle negative labels to avoid class values error
                            if isinstance(device_labels, torch.Tensor):
                                labels_view = device_labels.view(-1)
                                if (labels_view < 0).any() and (labels_view != -100).any():
                                    # Replace negative values (except -100) with -100 (ignore_index)
                                    fixed_labels = torch.where(
                                        (labels_view < 0) & (labels_view != -100),
                                        torch.tensor(-100, device=labels_view.device),
                                        labels_view
                                    )
                                    device_labels = fixed_labels.view(device_labels.shape)
                            
                            # Use weighted loss if enabled
                            if hasattr(self, 'use_weighted_loss') and self.use_weighted_loss and hasattr(self, 'class_weights') and self.class_weights is not None:
                                weights = self.class_weights.to(model_device)
                                loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                            else:
                                # Otherwise use standard cross entropy
                                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                            
                            # Reshape tensors for loss calculation, ensuring we don't exceed valid indices
                            logits = outputs.logits.view(-1, outputs.logits.size(-1))
                            target = device_labels.view(-1)
                            
                            # Check for invalid indices
                            max_idx = logits.size(1) - 1
                            valid_indices = (target >= -100) & (target <= max_idx)
                            if not valid_indices.all():
                                # Filter out invalid indices (replace with -100 which is ignored)
                                target = torch.where(valid_indices, target, torch.tensor(-100, device=target.device))
                            
                            loss = loss_fct(logits, target)
                            
                            if loss is not None:  # Ensure loss is not None before calling item()
                                total_loss += loss.item()
                                num_batches += 1
                        except RuntimeError as e:
                            logger.warning(f"Error calculating loss in batch {batch_idx}: {str(e)}")
                            # Continue without adding loss for this batch
                    
                    # Get predictions
                    batch_for_preds = device_inputs.copy()
                    if device_labels is not None:
                        batch_for_preds['labels'] = device_labels
                    
                    try:
                        preds, actual_labels = self.get_predictions(outputs, batch_for_preds)
                        
                        # Only process valid predictions and labels
                        if preds is not None and actual_labels is not None:
                            # Move to CPU for safe storage
                            preds_cpu = preds.detach().cpu() if isinstance(preds, torch.Tensor) else torch.tensor([])
                            labels_cpu = actual_labels.detach().cpu() if isinstance(actual_labels, torch.Tensor) else torch.tensor([])
                            
                            # Only add valid tensors
                            if preds_cpu.numel() > 0 and labels_cpu.numel() > 0:
                                all_preds.append(preds_cpu)
                                all_labels.append(labels_cpu)
                                successful_batches += 1
                    except Exception as e:
                        logger.warning(f"Error getting predictions in batch {batch_idx}: {str(e)}")
                        failed_batches += 1
                
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    logger.error(traceback.format_exc())
                    failed_batches += 1
                    continue
        
        # Compute average loss
        avg_loss = total_loss / max(1, num_batches)
        
        # Log success rate
        logger.info(f"Evaluation completed with {successful_batches} successful batches, {failed_batches} failed batches")
        
        # Compute metrics if we have predictions and labels
        metrics = {}
        if all_preds and all_labels:
            try:
                # Safely concatenate batched predictions and labels
                preds_concat = torch.cat(all_preds, dim=0) 
                labels_concat = torch.cat(all_labels, dim=0)
                
                # Log prediction distribution
                self._log_prediction_distribution(preds_concat, labels_concat)
                
                metrics = self.calculate_metrics(preds_concat, labels_concat)
            except Exception as e:
                logger.error(f"Error calculating final metrics: {e}")
                logger.error(traceback.format_exc())
                metrics = {
                    'accuracy': 0,
                    'f1': 0, 
                    'precision': 0,
                    'recall': 0,
                    'entity_f1': 0
                }
        
        logger.info(f"Validation complete - Loss: {avg_loss:.4f}, Metrics: {metrics}")
        return avg_loss, metrics
    
    def _log_prediction_distribution(self, preds, labels):
        """Log distribution of predicted labels to check for class imbalance issues."""
        try:
            # Convert to numpy for easier analysis
            preds_np = preds.flatten().cpu().numpy()
            labels_np = labels.flatten().cpu().numpy()
            
            # Filter out padding (-100)
            mask = labels_np != -100
            preds_filtered = preds_np[mask]
            labels_filtered = labels_np[mask]
            
            # Get unique predicted labels and their counts
            pred_labels, pred_counts = np.unique(preds_filtered, return_counts=True)
            true_labels, true_counts = np.unique(labels_filtered, return_counts=True)
            
            # Calculate totals
            total_preds = len(preds_filtered)
            
            # Log distributions
            logger.info(f"Predicted label distribution (total: {total_preds}):")
            for i, (label_id, count) in enumerate(zip(pred_labels, pred_counts)):
                label_name = MODEL_CONFIGS["reference_parser"]["id2label"].get(label_id, "Unknown")
                percentage = (count / total_preds) * 100
                logger.info(f"  {label_name}: {count} ({percentage:.2f}%)")
            
            # Compare with true label distribution
            logger.info(f"True label distribution (total: {len(labels_filtered)}):")
            for i, (label_id, count) in enumerate(zip(true_labels, true_counts)):
                label_name = MODEL_CONFIGS["reference_parser"]["id2label"].get(label_id, "Unknown")
                percentage = (count / len(labels_filtered)) * 100
                logger.info(f"  {label_name}: {count} ({percentage:.2f}%)")
                
                # Check for potential overfit to the "O" class
                o_label_id = MODEL_CONFIGS["reference_parser"]["label2id"].get("O", 0)
                o_pred_count = pred_counts[pred_labels == o_label_id][0] if o_label_id in pred_labels else 0
                o_true_count = true_counts[true_labels == o_label_id][0] if o_label_id in true_labels else 0
                
                o_pred_percentage = (o_pred_count / total_preds) * 100
                o_true_percentage = (o_true_count / len(labels_filtered)) * 100
                
                if o_pred_percentage > o_true_percentage + 5:
                    logger.warning(f"Model is overpredicting 'O' class: {o_pred_percentage:.2f}% vs {o_true_percentage:.2f}% in true data")
            
        except Exception as e:
            logger.error(f"Error logging prediction distribution: {e}")
            logger.error(traceback.format_exc())
    
    def get_dataloader(self, data_path, shuffle=True):
        """Create a DataLoader for the reference parsing dataset."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["reference_parser"]["model_name"])
            dataset = ReferenceParsingDataset(data_path, tokenizer, self.max_length)
            
            # Create the DataLoader
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self._safe_collate
            )
            
            return loader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader for {data_path}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _compute_class_weights(self, label_counts):
        """Compute class weights based on inverse frequency."""
        # Get label IDs and their counts
        label_ids = list(label_counts.keys())
        counts = list(label_counts.values())
        
        if not counts:
            logger.warning("No valid label counts found for computing weights")
            return None
            
        # Calculate inverse frequency weights
        total_count = sum(counts)
        inv_freqs = [total_count / max(1, count) for count in counts]
        
        # Normalize weights
        total_inv = sum(inv_freqs)
        weights = [max(0.1, freq / total_inv * len(counts)) for freq in inv_freqs]
        
        # Create tensor with weights ordered by label ID
        weight_tensor = torch.ones(len(MODEL_CONFIGS["reference_parser"]["id2label"]))
        
        # Assign weights to correct positions
        for label_id, weight in zip(label_ids, weights):
            # Ensure label_id is a valid index and weight is positive
            if 0 <= label_id < len(weight_tensor):
                weight_tensor[label_id] = max(0.1, weight)  # Ensure no zero or negative weights
        
        logger.info(f"Class weights by label name:")
        for i, w in enumerate(weight_tensor):
            label_name = MODEL_CONFIGS["reference_parser"]["id2label"].get(i, "Unknown")
            logger.info(f"  {label_name}: {w:.4f}")
            
        return weight_tensor
    
    def _compute_weighted_loss(self, outputs, labels):
        """Compute weighted loss for token classification."""
        logits = outputs.logits
        # Reshape logits and labels for computing loss
        logits_view = logits.view(-1, logits.size(-1))
        labels_view = labels.view(-1)
        
        # Handle negative labels - replace with ignore_index
        # CrossEntropyLoss will ignore these in the loss calculation
        # This is critical to avoid "class values must be non-negative" errors
        if (labels_view < 0).any() and (labels_view != -100).any():
            labels_view = torch.where(
                (labels_view < 0) & (labels_view != -100),
                torch.tensor(-100, device=labels_view.device),
                labels_view
            )
        
        # Apply class weights to cross entropy loss
        if self.class_weights is not None:
            weights = self.class_weights.to(self.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
        return loss_fct(logits_view, labels_view)
    
    def _safe_collate(self, batch):
        """Safely collate examples for token classification."""
        if not batch:
            return None
        
        # Check if all tensors in the batch are valid
        valid_batch = []
        for example in batch:
            if all(t is not None and isinstance(t, torch.Tensor) for t in example.values()):
                valid_batch.append(example)
        
        if not valid_batch:
            logger.warning("No valid examples in batch")
            return self._create_empty_batch()
        
        # Get all keys from the first example
        keys = valid_batch[0].keys()
        
        # Prepare tensors for each key
        batch_tensors = {}
        for key in keys:
            try:
                # For input_ids, attention_mask, etc.
                if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    # Check dimensions and max sequence length
                    tensors = [example[key] for example in valid_batch]
                    max_len = max(t.size(0) for t in tensors)
                    max_len = min(max_len, self.max_length)  # Limit to max_length
                    
                    # Pad or truncate each tensor
                    padded_tensors = []
                    for tensor in tensors:
                        if tensor.size(0) < max_len:
                            # Pad
                            padding = torch.zeros(max_len - tensor.size(0), dtype=tensor.dtype)
                            padded = torch.cat([tensor[:max_len], padding])
                        else:
                            # Truncate
                            padded = tensor[:max_len]
                        padded_tensors.append(padded)
                    
                    batch_tensors[key] = torch.stack(padded_tensors)
                
                # For labels
                elif key == 'labels':
                    tensors = [example[key] for example in valid_batch]
                    max_len = max(t.size(0) for t in tensors)
                    max_len = min(max_len, self.max_length)  # Limit to max_length
                    
                    # Pad or truncate each tensor, using -100 for padding (ignored in loss)
                    padded_tensors = []
                    for tensor in tensors:
                        if tensor.size(0) < max_len:
                            # Pad with -100 (ignored in loss calculation)
                            padding = torch.ones(max_len - tensor.size(0), dtype=tensor.dtype) * -100
                            padded = torch.cat([tensor[:max_len], padding])
                        else:
                            # Truncate
                            padded = tensor[:max_len]
                        padded_tensors.append(padded)
                    
                    batch_tensors[key] = torch.stack(padded_tensors)
                
            except Exception as e:
                logger.error(f"Error collating {key}: {e}")
                logger.error(traceback.format_exc())
                # Create a default tensor of appropriate shape
                if key == 'labels':
                    batch_tensors[key] = torch.ones(len(valid_batch), self.max_length, dtype=torch.long) * -100
                else:
                    batch_tensors[key] = torch.zeros(len(valid_batch), self.max_length, dtype=torch.long)
        
        return batch_tensors
    
    def _create_empty_batch(self):
        """Create an empty batch for fallback."""
        return {
            'input_ids': torch.zeros((1, self.max_length), dtype=torch.long),
            'attention_mask': torch.zeros((1, self.max_length), dtype=torch.long),
            'token_type_ids': torch.zeros((1, self.max_length), dtype=torch.long),
            'labels': torch.ones((1, self.max_length), dtype=torch.long) * -100
        }
    
    def get_predictions(self, outputs, batch):
        """Get predictions from model outputs."""
        try:
            # Check that outputs has logits
            if not hasattr(outputs, 'logits'):
                logger.warning("Model outputs don't contain logits")
                return None, None
                
            # Get predicted token classes (ignore padding)
            logits = outputs.logits
            
            # Validate logits shape
            if len(logits.shape) != 3:
                logger.warning(f"Unexpected logits shape: {logits.shape}, expected 3 dimensions")
                return None, None
                
            # Get predictions safely
            with torch.no_grad():
                # Move to CPU for safer operations
                logits_cpu = logits.detach().cpu()
                
                # Get argmax along last dimension
                try:
                    preds = torch.argmax(logits_cpu, dim=2)
                except Exception as e:
                    logger.error(f"Error calculating argmax: {e}")
                    return None, None
            
            # Check for labels in batch
            if 'labels' not in batch:
                logger.warning("No labels in batch")
                return preds, None
                
            # Return predictions and labels
            return preds, batch['labels']
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            logger.error(traceback.format_exc())
            # Return None values as fallback
            return None, None
    
    def calculate_metrics(self, preds, labels):
        """Calculate metrics for token classification."""
        try:
            # Ensure inputs are tensors but handle the case where they might already be tensors
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, device=self.device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
                
            # Ensure no invalid values before flattening
            num_classes = len(MODEL_CONFIGS["reference_parser"]["id2label"])
            # Fix any invalid prediction values  
            preds = torch.clamp(preds, min=0, max=num_classes-1)
            
            # Flatten the arrays and convert to CPU numpy arrays BEFORE any indexing
            preds_flat = preds.flatten().cpu().numpy()
            labels_flat = labels.flatten().cpu().numpy()
            
            # Filter out ignored index (-100)
            mask = labels_flat != -100
            preds_filtered = preds_flat[mask]
            labels_filtered = labels_flat[mask]
            
            # Ensure all labels are valid and non-negative
            mask_valid_labels = (labels_filtered >= 0) & (labels_filtered < num_classes)
            if not mask_valid_labels.all():
                logger.warning(f"Found {(~mask_valid_labels).sum()} invalid label indices. Filtering out.")
                preds_filtered = preds_filtered[mask_valid_labels]
                labels_filtered = labels_filtered[mask_valid_labels]
            
            if len(labels_filtered) == 0:
                logger.warning("No valid labels for metric calculation")
                return {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'entity_f1': 0.0
                }
            
            # Calculate standard metrics
            accuracy = accuracy_score(labels_filtered, preds_filtered)
            f1 = f1_score(labels_filtered, preds_filtered, average='weighted')
            precision = precision_score(labels_filtered, preds_filtered, average='weighted', zero_division=0)
            recall = recall_score(labels_filtered, preds_filtered, average='weighted', zero_division=0)
            
            # Calculate per-class metrics
            class_f1s = f1_score(labels_filtered, preds_filtered, average=None, zero_division=0)
            
            # Calculate entity-level F1 score
            entity_f1 = self._calculate_entity_f1(preds, labels)
            
            metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'entity_f1': entity_f1
            }
            
            # Add per-class metrics for each reference element type
            for i, label_name in enumerate(REF_LABEL_TYPES):
                if i < len(class_f1s):
                    metrics[f'f1_{label_name}'] = class_f1s[i]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logger.error(traceback.format_exc())
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'entity_f1': 0.0
            }
    
    def _calculate_entity_f1(self, preds, labels):
        """Calculate entity-level F1 score."""
        try:
            # Ensure inputs are tensors
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, device=self.device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
                
            # Ensure no invalid values
            num_classes = len(MODEL_CONFIGS["reference_parser"]["id2label"])
            preds = torch.clamp(preds, min=0, max=num_classes-1)
            
            # Convert tensor predictions to CPU numpy arrays
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Lists to store all entities
            true_entities = []
            pred_entities = []
            
            # Process each sequence in the batch
            for i in range(preds_np.shape[0]):
                # Filter out invalid label indices
                valid_indices = labels_np[i] != -100
                seq_true_labels = labels_np[i][valid_indices]
                seq_pred_labels = preds_np[i][valid_indices]
                
                if len(seq_true_labels) == 0:
                    continue
                    
                true_entities_seq = self._extract_entities(seq_true_labels)
                pred_entities_seq = self._extract_entities(seq_pred_labels)
                
                true_entities.extend(true_entities_seq)
                pred_entities.extend(pred_entities_seq)
            
            # Calculate entity-level metrics
            if not true_entities and not pred_entities:
                return 1.0  # Perfect score if no entities in ground truth and predictions
            
            if not true_entities or not pred_entities:
                return 0.0  # Zero score if either ground truth or predictions are empty
            
            # Count true positives, false positives, and false negatives
            tp = len([entity for entity in pred_entities if entity in true_entities])
            fp = len([entity for entity in pred_entities if entity not in true_entities])
            fn = len([entity for entity in true_entities if entity not in pred_entities])
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            # Log entity-level metrics for debugging
            if tp + fp + fn > 0:
                logger.info(f"Entity-level metrics - TP: {tp}, FP: {fp}, FN: {fn}")
                logger.info(f"Entity-level precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Log unique entities found
                unique_pred_entity_types = set(entity[0] for entity in pred_entities)
                unique_true_entity_types = set(entity[0] for entity in true_entities)
                logger.info(f"Predicted entity types: {unique_pred_entity_types}")
                logger.info(f"True entity types: {unique_true_entity_types}")
            
            return f1
            
        except Exception as e:
            logger.error(f"Error calculating entity F1: {e}")
            logger.error(traceback.format_exc())
            return 0.0
    
    def _extract_entities(self, sequence):
        """Extract entities from a sequence of token labels."""
        entities = []
        current_entity = None
        
        # Get a map of ID to label name
        id2label = MODEL_CONFIGS["reference_parser"]["id2label"]
        num_labels = len(id2label)
        
        # Process tokens, ignoring padding (-100)
        for i, label_id in enumerate(sequence):
            # Skip padding or invalid labels
            if label_id == -100 or label_id < 0 or label_id >= num_labels:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Convert ID to label safely
            label = id2label.get(int(label_id), "O")
            
            # Skip 'O' (outside) tokens
            if label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Start a new entity or continue the current one
            if current_entity is None or current_entity[0] != label:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = (label, i, i)
            else:
                # Extend the current entity
                current_entity = (label, current_entity[1], i)
        
        # Add the last entity if there's one
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NLP models for scientific document processing')
    
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['section_classifier', 'reference_parser'],
                        help='Type of model to train')
    
    parser.add_argument('--data-dir', type=str, default="data/transfer_learning/prepared",
                      help='Base directory containing prepared training data')
    
    parser.add_argument('--model-dir', type=str, default="models",
                      help='Directory to save trained models')
    
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    
    parser.add_argument('--max-length', type=int, default=512,
                      help='Maximum sequence length')
    
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate')
    
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    
    parser.add_argument('--unfreeze', action='store_true',
                      help='Unfreeze base model layers during training')

    parser.add_argument('--freeze-base-model', action='store_true',
                      help='Freeze base model layers during training (overrides --unfreeze)')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--train-all', action='store_true',
                      help='Train all models in sequence')
    
    # Add new arguments for addressing overfitting
    parser.add_argument('--weight-decay', type=float, default=0.01,
                      help='Weight decay for regularization')
    
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate for regularization')
    
    parser.add_argument('--early-stopping', type=int, default=2,
                      help='Early stopping patience')
    
    # Add dynamic learning rate arguments
    parser.add_argument('--use-scheduler', action='store_true',
                      help='Use learning rate scheduler')
    
    parser.add_argument('--warmup-steps', type=int, default=0,
                      help='Number of warmup steps for learning rate')
    
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.5,
                      help='Factor by which to reduce learning rate on plateau')
    
    parser.add_argument('--lr-scheduler-patience', type=int, default=1,
                      help='Patience for learning rate scheduler')
    
    # Add gradient clipping and label smoothing
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                      help='Max gradient norm for gradient clipping')
    
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                      help='Label smoothing factor (0.0-0.2 recommended)')
    
    # Add weighted loss argument
    parser.add_argument('--weighted-loss', action='store_true',
                      help='Use class-weighted loss function to address class imbalance')
    
    # Add show examples argument
    parser.add_argument('--show-examples', action='store_true',
                      help='Show example predictions after training (may cause CUDA errors with large models)')
    
    # Add CPU-only mode for avoiding CUDA errors
    parser.add_argument('--cpu-only', action='store_true',
                     help='Force CPU-only training to avoid CUDA errors')
    
    return parser.parse_args()

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_models():
    """Train all models in sequence."""
    args = parse_args()
    
    set_seed(args.seed)
    
    # Override device if CPU-only mode is specified
    global DEVICE
    if args.cpu_only:
        logger.info("CPU-only mode enabled, disabling CUDA")
        DEVICE = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Function to find data paths
    def find_data_path(base_path):
        """Find train and validation data paths for a model type."""
        train_path = Path(base_path) / "train" / "data.json"
        val_path = Path(base_path) / "val" / "data.json"
        
        # Check if files exist
        if not train_path.exists() or not val_path.exists():
            logger.error(f"Data files not found at {train_path} or {val_path}")
            return None, None
            
        return train_path, val_path
    
    # Train section classifier if requested
    if args.train_all or args.model_type == "section_classifier":
        logger.info("Training section classifier model")
        
        # Setup paths
        data_base = Path(args.data_dir) / "section_classification"
        train_path, val_path = find_data_path(data_base)
        
        if train_path and val_path:
            try:
                # Initialize trainer
                trainer = SectionClassifierTrainer(
                    model_type="section_classifier",
                    model_dir=Path(args.model_dir) / "section_classifier",
                    freeze_base_model=args.freeze_base_model if args.freeze_base_model else not args.unfreeze,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=DEVICE
                )
                
                # Train and save model
                trainer.train(
                    train_data_path=train_path,
                    val_data_path=val_path,
                    epochs=args.epochs,
                    warmup_steps=args.warmup_steps,
                    weight_decay=args.weight_decay,
                    early_stopping_patience=args.early_stopping,
                    save_best_model=True,
                    use_lr_scheduler=args.use_scheduler,
                    lr_scheduler_factor=args.lr_scheduler_factor,
                    lr_scheduler_patience=args.lr_scheduler_patience,
                    gradient_clip=args.gradient_clip,
                    label_smoothing=args.label_smoothing,
                    show_examples=args.show_examples
                )
                
                logger.info("Section classifier training complete!")
                
            except Exception as e:
                logger.error(f"Error training section classifier: {e}")
                logger.error(traceback.format_exc())
    
    # Train reference parser if requested
    if args.train_all or args.model_type == "reference_parser":
        logger.info("Training reference parser model")
        
        # Setup paths
        data_base = Path(args.data_dir) / "reference_parsing"
        train_path, val_path = find_data_path(data_base)
        
        if train_path and val_path:
            try:
                # Initialize trainer with weighted loss option
                trainer = ReferenceParserTrainer(
                    model_type="reference_parser",
                    model_dir=Path(args.model_dir) / "reference_parser",
                    freeze_base_model=args.freeze_base_model if args.freeze_base_model else not args.unfreeze,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=DEVICE,
                    use_weighted_loss=args.weighted_loss  # Pass weighted loss flag
                )
                
                # Log whether using weighted loss
                if args.weighted_loss:
                    logger.info("Using class-weighted loss to address class imbalance")
                
                # Train and save model
                trainer.train(
                    train_data_path=train_path,
                    val_data_path=val_path,
                    epochs=args.epochs,
                    warmup_steps=args.warmup_steps,
                    weight_decay=args.weight_decay * 2,  # Increase weight decay for regularization
                    early_stopping_patience=args.early_stopping,
                    save_best_model=True,
                    use_lr_scheduler=args.use_scheduler,
                    lr_scheduler_factor=args.lr_scheduler_factor,
                    lr_scheduler_patience=args.lr_scheduler_patience,
                    gradient_clip=args.gradient_clip,
                    label_smoothing=args.label_smoothing,
                    show_examples=args.show_examples
                )
                
                logger.info("Reference parser training complete!")
                
            except Exception as e:
                logger.error(f"Error training reference parser: {e}")
                logger.error(traceback.format_exc())
    
    logger.info("All training tasks completed!")

if __name__ == "__main__":
    train_models() 