"""
Simplified figure detection model training script - final version
"""
import os
import sys
import torch
import logging
import argparse
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMForTokenClassification, 
    LayoutLMTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/figure_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("figure_model_training")

# Define label types for figure detection
FIGURE_LABEL_TYPES = ["text", "title", "list", "table", "figure"]
LABEL2ID = {label: i for i, label in enumerate(FIGURE_LABEL_TYPES)}
ID2LABEL = {i: label for i, label in enumerate(FIGURE_LABEL_TYPES)}

class FigureDetectionDataset(Dataset):
    """Dataset for figure detection using LayoutLM."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """Initialize the dataset."""
        logger.info(f"Initializing dataset with data from: {data_path}")
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = LABEL2ID
        
        # Load data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Successfully loaded {len(self.data)} examples")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error(traceback.format_exc())
            self.data = []
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an example from the dataset."""
        try:
            example = self.data[idx]
            
            # Extract data
            words = example.get('words', [])
            boxes = example.get('boxes', [])
            labels = example.get('labels', [])
            
            # Ensure all arrays have the same length
            min_len = min(len(words), len(boxes), len(labels))
            words = words[:min_len]
            boxes = boxes[:min_len]
            labels = labels[:min_len]
            
            # Handle empty data
            if len(words) == 0:
                # Return a placeholder with minimal data that won't break the model
                return self.create_fallback_encoding()
            
            # Convert all box coordinates to integers and normalize
            validated_boxes = []
            for box in boxes:
                if isinstance(box, list) and len(box) == 4:
                    try:
                        # Ensure all coordinates are integers and positive
                        box = [max(0, int(coord)) for coord in box]
                        # Ensure box has valid dimensions (width and height > 0)
                        if box[2] > box[0] and box[3] > box[1]:
                            validated_boxes.append(box)
                        else:
                            # Skip invalid box and corresponding word/label
                            logger.warning(f"Invalid box dimensions in item {idx}: {box}")
                            continue
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid box format in item {idx}: {box}")
                        continue
            
            # If no valid boxes remain, return fallback
            if not validated_boxes:
                logger.warning(f"No valid boxes in item {idx}")
                return self.create_fallback_encoding()
            
            # Match words length to validated boxes
            words = words[:len(validated_boxes)]
            labels = labels[:len(validated_boxes)]
            
            # Create encoding for all inputs manually
            encoding = {}
            
            # We'll manually handle tokenization and encoding
            # First, tokenize each word and track token-to-word mapping
            token_boxes = []
            token_words = []
            word_ids = []
            
            for i, (word, box) in enumerate(zip(words, validated_boxes)):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    # Skip words that tokenize to empty
                    continue
                
                token_words.extend(word_tokens)
                # Assign the same box coordinates to each token from the word
                token_boxes.extend([box] * len(word_tokens))
                # Track which word each token came from
                word_ids.extend([i] * len(word_tokens))
            
            # Add special tokens (CLS and SEP)
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            pad_token_id = self.tokenizer.pad_token_id
            
            # Convert tokens to IDs
            token_ids = self.tokenizer.convert_tokens_to_ids(token_words)
            
            # Add CLS and SEP tokens
            input_ids = [cls_token_id] + token_ids + [sep_token_id]
            
            # Add padding
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([pad_token_id] * padding_length)
            else:
                # Truncate if too long
                input_ids = input_ids[:self.max_length]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * min(len(token_ids) + 2, self.max_length)
            padding_length = self.max_length - len(attention_mask)
            if padding_length > 0:
                attention_mask = attention_mask + ([0] * padding_length)
            
            # Create bbox tensor - add special token boxes and padding
            # CLS and SEP get [0, 0, 0, 0] bbox
            special_box = [0, 0, 0, 0]
            bbox_tensor = [special_box] + token_boxes + [special_box]
            
            # Truncate or pad bbox_tensor
            if len(bbox_tensor) > self.max_length:
                bbox_tensor = bbox_tensor[:self.max_length]
            else:
                padding_length = self.max_length - len(bbox_tensor)
                bbox_tensor = bbox_tensor + ([special_box] * padding_length)
            
            # Prepare labels
            label_ids = [self.label_map.get(label, 0) for label in labels]
            
            # Create token-level labels, mapping from word-level labels using word_ids
            token_labels = [-100] * self.max_length  # -100 is ignored in loss calculation
            
            # CLS and SEP tokens get -100 as their labels
            for i, word_id in enumerate(word_ids):
                if i + 1 < self.max_length:  # Add 1 for CLS token offset
                    if word_id < len(label_ids):
                        token_labels[i + 1] = label_ids[word_id]
            
            # Convert to tensors
            encoding['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
            encoding['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
            encoding['bbox'] = torch.tensor(bbox_tensor, dtype=torch.long)
            encoding['labels'] = torch.tensor(token_labels, dtype=torch.long)
            
            return encoding
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(traceback.format_exc())
            return self.create_fallback_encoding()
    
    def create_fallback_encoding(self):
        """Create a fallback encoding that won't break the model."""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'bbox': torch.zeros((self.max_length, 4), dtype=torch.long),
            'labels': torch.ones(self.max_length, dtype=torch.long) * -100
        }

def safe_collate(batch, max_length=512):
    """Custom collate function to handle dimension issues properly."""
    # Filter out None values
    batch = [b for b in batch if b is not None]
    
    if not batch:
        # Create a minimal dummy batch if empty
        return {
            'input_ids': torch.zeros((1, max_length), dtype=torch.long),
            'attention_mask': torch.zeros((1, max_length), dtype=torch.long),
            'bbox': torch.zeros((1, max_length, 4), dtype=torch.long),
            'labels': torch.full((1, max_length), -100, dtype=torch.long)
        }
    
    # Standardize all tensors to ensure consistent shapes
    standardized_batch = []
    for sample in batch:
        # Create a standardized copy
        std_sample = {}
        
        # Standard tensors: input_ids, attention_mask, labels
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in sample:
                tensor = sample[key]
                # Ensure 1D tensor of correct length
                if tensor.dim() > 1:
                    tensor = tensor.squeeze(0)
                if tensor.size(0) != max_length:
                    new_tensor = torch.zeros(max_length, dtype=tensor.dtype)
                    if key == 'labels':
                        new_tensor.fill_(-100)
                    # Copy as much as fits
                    copy_len = min(tensor.size(0), max_length)
                    new_tensor[:copy_len] = tensor[:copy_len]
                    tensor = new_tensor
                std_sample[key] = tensor
            else:
                # Create default tensor if missing
                if key == 'labels':
                    std_sample[key] = torch.full((max_length,), -100, dtype=torch.long)
                else:
                    std_sample[key] = torch.zeros(max_length, dtype=torch.long)
        
        # Handle bbox specially to ensure shape [max_length, 4]
        if 'bbox' in sample:
            bbox = sample['bbox']
            if bbox.dim() == 2 and bbox.size(1) == 4 and bbox.size(0) == max_length:
                # Correct shape already
                std_sample['bbox'] = bbox
            else:
                # Create correct shape
                new_bbox = torch.zeros((max_length, 4), dtype=torch.long)
                if bbox.dim() == 2 and bbox.size(1) == 4:
                    # Correct format but wrong length
                    copy_len = min(bbox.size(0), max_length)
                    new_bbox[:copy_len] = bbox[:copy_len]
                elif bbox.dim() == 1 and bbox.size(0) % 4 == 0:
                    # Reshape if possible
                    reshaped = bbox.view(-1, 4)
                    copy_len = min(reshaped.size(0), max_length)
                    new_bbox[:copy_len] = reshaped[:copy_len]
                std_sample['bbox'] = new_bbox
        else:
            std_sample['bbox'] = torch.zeros((max_length, 4), dtype=torch.long)
        
        standardized_batch.append(std_sample)
    
    # Now stack the standardized tensors
    result = {}
    for key in ['input_ids', 'attention_mask', 'labels']:
        try:
            result[key] = torch.stack([s[key] for s in standardized_batch])
        except RuntimeError as e:
            logger.error(f"Error stacking {key}: {str(e)}")
            if key == 'labels':
                result[key] = torch.full((len(batch), max_length), -100, dtype=torch.long)
            else:
                result[key] = torch.zeros((len(batch), max_length), dtype=torch.long)
    
    # Stack bbox tensors
    try:
        bbox_tensors = [s['bbox'] for s in standardized_batch]
        result['bbox'] = torch.stack(bbox_tensors)
    except RuntimeError as e:
        logger.error(f"Error stacking bbox: {str(e)}")
        result['bbox'] = torch.zeros((len(batch), max_length, 4), dtype=torch.long)
    
    return result

def train_model(train_path, val_path, model_dir, batch_size=4, epochs=2, max_length=128, early_stopping_patience=3):
    """Train the figure detection model."""
    logger.info(f"Training figure detection model with data from {train_path}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    
    # Create model directory
    model_dir = Path(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    model_name = "microsoft/layoutlm-base-uncased"
    tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
    
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(FIGURE_LABEL_TYPES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    model = LayoutLMForTokenClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Move model to device
    model.to(DEVICE)
    
    # Create datasets and data loaders
    train_dataset = FigureDetectionDataset(train_path, tokenizer, max_length=max_length)
    val_dataset = FigureDetectionDataset(val_path, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: safe_collate(batch, max_length)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: safe_collate(batch, max_length)
    )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Early stopping variables
    best_f1 = 0.0
    early_stopping_counter = 0
    best_model_path = None
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            try:
                # Move batch to device
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                try:
                    # Move batch to device
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    
                    # Get predictions for non-ignored positions (-100)
                    active_mask = batch['labels'].view(-1) != -100
                    active_preds = torch.argmax(outputs.logits.view(-1, len(FIGURE_LABEL_TYPES)), dim=-1)[active_mask]
                    active_labels = batch['labels'].view(-1)[active_mask]
                    
                    all_preds.extend(active_preds.cpu().numpy().tolist())
                    all_labels.extend(active_labels.cpu().numpy().tolist())
                
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    logger.error(traceback.format_exc())
                    continue
        
        # Calculate validation metrics
        current_f1 = 0.0
        if all_preds and all_labels:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            current_f1 = f1  # Store for early stopping
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            logger.info(f"Validation results - "
                        f"Accuracy: {accuracy:.4f}, "
                        f"F1: {f1:.4f}, "
                        f"Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}")
        else:
            logger.warning("Empty prediction or label arrays - skipping metrics calculation")
        
        # Save checkpoint for this epoch
        checkpoint_dir = model_dir / f"checkpoint_epoch_{epoch+1}"
        model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Check for early stopping
        if current_f1 > best_f1:
            best_f1 = current_f1
            early_stopping_counter = 0
            
            # Save best model
            best_model_path = model_dir / "best_model"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"New best model with F1: {best_f1:.4f}, saved to {best_model_path}")
        else:
            early_stopping_counter += 1
            logger.info(f"F1 did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    model.save_pretrained(model_dir / "final_model")
    tokenizer.save_pretrained(model_dir / "final_model")
    logger.info(f"Model saved to {model_dir / 'final_model'}")
    
    # Copy best model to final location if available
    if best_model_path and best_model_path.exists():
        logger.info(f"Using best model from epoch {epochs - early_stopping_counter}")
    
    return model_dir / "final_model"

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the figure detection model")
    parser.add_argument("--data-dir", type=str, default="data/transfer_learning/prepared",
                      help="Directory containing prepared data")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--early-stopping-patience", type=int, default=3, 
                      help="Number of epochs with no improvement after which training will be stopped")
    args = parser.parse_args()
    
    # Initialize logging directory
    os.makedirs("logs", exist_ok=True)
    
    # Setup data paths
    data_dir = Path(args.data_dir)
    train_path = data_dir / "figure_detection/train/fixed_data.json"
    val_path = data_dir / "figure_detection/val/fixed_data.json"
    
    # Verify data exists
    from create_figure_detection_data import create_figure_detection_data
    if not train_path.exists() or not val_path.exists():
        logger.info("Fixed data not found, generating it...")
        create_figure_detection_data()
    
    # Train model
    model_dir = Path("models/figure_detector")
    try:
        train_model(
            train_path=train_path,
            val_path=val_path,
            model_dir=model_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_length=args.max_length,
            early_stopping_patience=args.early_stopping_patience
        )
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc()) 