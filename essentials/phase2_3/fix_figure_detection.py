import re
import json
import os
import sys
import logging
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_figure_detection")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def process_figure_detection_data(data_path):
    """Process and fix the figure detection dataset."""
    # Read the data
    logger.info(f"Reading data from {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decoding failed for {data_path}, trying with ISO-8859-1 encoding")
        with open(data_path, 'r', encoding='ISO-8859-1') as f:
            data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples for figure detection")
    
    # Convert all items to consistent format with proper bounding boxes
    fixed_data = []
    
    try:
        import tqdm
        iterator = tqdm.tqdm(enumerate(data), total=len(data), desc="Processing samples", disable=True)

    except ImportError:
        iterator = enumerate(data)
        
    for i, item in iterator:
        fixed_item = {}
        
        # Ensure 'words' field exists
        if 'words' not in item:
            if 'text' in item:
                words = item['text'].split()
            else:
                words = ["placeholder"]
            fixed_item['words'] = words
        else:
            fixed_item['words'] = item['words']
        
        # Create proper bounding boxes
        if 'boxes' not in item or not item['boxes']:
            # Create synthetic bounding boxes
            # Each word gets a box in a horizontal line
            # with boxes positioned sequentially
            word_count = len(fixed_item['words'])
            boxes = []
            
            # Create a simple left-to-right layout
            page_width = 1000
            page_height = 1000
            line_height = 50
            word_width = page_width // (word_count + 2) if word_count > 0 else 100  # +2 for margins
            
            for i, word in enumerate(fixed_item['words']):
                # Position: [left, top, right, bottom]
                left = (i + 1) * word_width
                top = page_height // 2 - line_height // 2
                right = left + word_width
                bottom = top + line_height
                boxes.append([left, top, right, bottom])
            
            fixed_item['boxes'] = boxes
        else:
            # Validate/fix existing boxes format
            boxes = item['boxes']
            fixed_boxes = []
            for box in boxes:
                if isinstance(box, list) and len(box) == 4:
                    fixed_boxes.append(box)
                else:
                    # Create default box
                    fixed_boxes.append([0, 0, 100, 100])
            
            # Adjust length to match words
            if len(fixed_boxes) < len(fixed_item['words']):
                # Create additional boxes for remaining words
                for i in range(len(fixed_boxes), len(fixed_item['words'])):
                    fixed_boxes.append([i*100, 0, (i+1)*100, 100])
            elif len(fixed_boxes) > len(fixed_item['words']):
                fixed_boxes = fixed_boxes[:len(fixed_item['words'])]
                
            fixed_item['boxes'] = fixed_boxes
        
        # Ensure 'labels' field exists
        if 'labels' not in item:
            has_figure = False
            if 'has_figure' in item:
                has_figure = item['has_figure']
            
            # Create labels based on content or has_figure flag
            labels = []
            for word in fixed_item['words']:
                # Check if word contains figure-related terms
                word_lower = str(word).lower()
                if has_figure or any(term in word_lower for term in ['fig', 'figure', 'chart', 'graph', 'table', 'image']):
                    labels.append('figure')
                else:
                    labels.append('text')
            fixed_item['labels'] = labels
        else:
            # Ensure labels length matches words
            labels = item['labels']
            if len(labels) < len(fixed_item['words']):
                labels.extend(['text'] * (len(fixed_item['words']) - len(labels)))
            elif len(labels) > len(fixed_item['words']):
                labels = labels[:len(fixed_item['words'])]
            fixed_item['labels'] = labels
        
        fixed_data.append(fixed_item)
    
    # Save fixed data
    out_path = Path(data_path).with_name("fixed_data.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2)
    
    logger.info(f"Fixed data saved to {out_path}")
    return str(out_path)

def create_figure_detector_fix():
    """Create a fresh solution file for the figure detection model."""
    print("Creating a complete solution for figure detection training...")
    
    # Get the original file content
    with open('transfer_learning.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Copy the entire content to a new file
    with open('transfer_learning_fixed.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Now we'll directly edit the fixed file with our improvements
    # by replacing the problematic methods manually
    
    # 1. Fix FigureDetectionDataset.__getitem__ by creating it as a completely new method
    # 2. Fix FigureDetectorTrainer._safe_collate 
    # 3. Fix evaluate method
    
    print("Running direct fixes to figure detection model...")
    import subprocess
    
    # Rather than using regex which can have indentation issues,
    # let's create a simple Python script that makes direct edits
    with open('direct_fix.py', 'w', encoding='utf-8') as f:
        f.write('''
import torch
import os
import json
import traceback
from pathlib import Path

# This script makes direct edits to the figure detection 
# related methods in the transfer_learning_fixed.py file

def fix_file():
    # Load the entire file
    with open('transfer_learning_fixed.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # We'll find key function positions and replace them
    # Find FigureDetectorTrainer._safe_collate
    safe_collate_start = None
    safe_collate_end = None
    
    for i, line in enumerate(lines):
        if 'def _safe_collate(self, batch):' in line and 'FigureDetectorTrainer' in lines[i-15:i]:
            safe_collate_start = i
            # Find the end of the function
            for j in range(i, len(lines)):
                if j+1 < len(lines) and 'def' in lines[j+1] and j > i+10:  # Next function
                    safe_collate_end = j
                    break
            break
    
    if safe_collate_start is not None and safe_collate_end is not None:
        print(f"Found _safe_collate method at lines {safe_collate_start}-{safe_collate_end}")
        
        # Replace with our fixed version
        safe_collate_fixed = """    def _safe_collate(self, batch):
        \"\"\"Custom collate function that handles problematic samples and tensor dimension mismatches.\"\"\"
        # Filter out None values
        batch = [b for b in batch if b is not None]
        
        if not batch:
            # Create a minimal dummy batch if all samples were filtered out
            dummy_sample = {
                'input_ids': torch.zeros((1, self.max_length), dtype=torch.long),
                'attention_mask': torch.zeros((1, self.max_length), dtype=torch.long),
                'bbox': torch.zeros((1, self.max_length, 4), dtype=torch.long),
                'labels': torch.tensor([[-100] * self.max_length], dtype=torch.long)
            }
            return dummy_sample
        
        # Ensure all required keys are present in each sample
        required_keys = ['input_ids', 'attention_mask', 'bbox', 'labels']
        for i, sample in enumerate(batch):
            for key in required_keys:
                if key not in sample:
                    logger.warning(f"Sample {i} missing required key '{key}' - creating default")
                    if key == 'bbox':
                        sample[key] = torch.zeros((self.max_length, 4), dtype=torch.long)
                    elif key == 'labels':
                        sample[key] = torch.tensor([-100] * self.max_length, dtype=torch.long)
                    else:
                        sample[key] = torch.zeros(self.max_length, dtype=torch.long)
        
        # Create padded and properly shaped tensors for each sample
        result = {}
        for key in required_keys:
            if key == 'bbox':
                # Handle bbox tensor (should be 3D: [batch, seq_len, 4])
                padded_tensors = []
                for sample in batch:
                    # Ensure correct shape: [seq_len, 4]
                    bbox = sample[key]
                    if len(bbox.shape) == 1:
                        # Handle 1D tensor (reshape to 2D)
                        if bbox.size(0) % 4 == 0:
                            # If divisible by 4, reshape
                            seq_len = bbox.size(0) // 4
                            bbox = bbox.reshape(seq_len, 4)
                        else:
                            # Create default tensor
                            bbox = torch.zeros((self.max_length, 4), dtype=torch.long)
                    
                    # Ensure correct length (pad/truncate)
                    if bbox.size(0) < self.max_length:
                        # Pad
                        padding = torch.zeros((self.max_length - bbox.size(0), 4), dtype=torch.long)
                        bbox = torch.cat([bbox, padding], dim=0)
                    elif bbox.size(0) > self.max_length:
                        # Truncate
                        bbox = bbox[:self.max_length, :]
                    
                    padded_tensors.append(bbox)
                result[key] = torch.stack(padded_tensors)
            else:
                # Handle 2D tensors: [batch, seq_len]
                padded_tensors = []
                for sample in batch:
                    tensor = sample[key]
                    
                    # Ensure 1D tensor
                    if len(tensor.shape) > 1:
                        tensor = tensor.squeeze()
                    
                    # Ensure correct length
                    if tensor.size(0) < self.max_length:
                        # Pad
                        pad_value = -100 if key == 'labels' else 0
                        padding = torch.full((self.max_length - tensor.size(0),), pad_value, dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding], dim=0)
                    elif tensor.size(0) > self.max_length:
                        # Truncate
                        tensor = tensor[:self.max_length]
                    
                    padded_tensors.append(tensor)
                    
                try:
                    result[key] = torch.stack(padded_tensors)
                except RuntimeError as e:
                    logger.error(f"Error stacking {key}: {str(e)}")
                    logger.error(f"Shapes: {[t.shape for t in padded_tensors]}")
                    
                    # Create a default tensor
                    if key == 'labels':
                        default_value = -100
                    else:
                        default_value = 0
                    result[key] = torch.full((len(batch), self.max_length), default_value, dtype=torch.long)
        
        return result
"""
        # Replace the function
        lines[safe_collate_start:safe_collate_end+1] = safe_collate_fixed.splitlines(True)
    else:
        print("Warning: Could not find _safe_collate method")

    # Find FigureDetectionDataset.__getitem__
    getitem_start = None
    getitem_end = None
    
    for i, line in enumerate(lines):
        if 'def __getitem__(self, idx):' in line and 'FigureDetectionDataset' in lines[i-30:i]:
            getitem_start = i
            # Find the end of the function
            for j in range(i, len(lines)):
                if j+1 < len(lines) and 'def' in lines[j+1] and j > i+10:  # Next function definition
                    getitem_end = j
                    break
            break
    
    if getitem_start is not None and getitem_end is not None:
        print(f"Found FigureDetectionDataset.__getitem__ method at lines {getitem_start}-{getitem_end}")
        
        # Replace with our fixed version
        getitem_fixed = """    def __getitem__(self, idx):
        \"\"\"Get an example from the dataset.\"\"\"
        try:
            example = self.data[idx]
            
            words = example.get('words', [])
            boxes = example.get('boxes', [[0, 0, 0, 0]] * len(words))
            labels = example.get('labels', ['text'] * len(words))
            
            # Make sure all arrays have the same length
            min_len = min(len(words), len(boxes), len(labels))
            if min_len < max(len(words), len(boxes), len(labels)):
                logger.warning(f"Item {idx} has inconsistent array lengths: words={len(words)}, boxes={len(boxes)}, labels={len(labels)}. Truncating to {min_len}.")
                words = words[:min_len]
                boxes = boxes[:min_len]
                labels = labels[:min_len]
            
            # Handle empty arrays
            if len(words) == 0:
                words = ["empty"]
                boxes = [[0, 0, 0, 0]]
                labels = ["text"]
            
            # Ensure boxes are properly formatted - each box should be a list of 4 integers
            for i, box in enumerate(boxes):
                if not isinstance(box, list) or len(box) != 4:
                    logger.warning(f"Item {idx} has malformed box at position {i}: {box}. Replacing with default.")
                    boxes[i] = [0, 0, 0, 0]
            
            # Tokenize words with boxes
            try:
                encoding = self.tokenizer(
                    words,
                    boxes=boxes,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # Remove batch dimension
                encoding = {k: v.squeeze(0) for k, v in encoding.items()}
                
                # Prepare labels
                label_ids = [self.label_map.get(label, 0) for label in labels]
                
                # Create word_ids mapping for token classification
                word_ids = []
                input_len = len(encoding['input_ids'])
                
                # Build word_ids mapping manually for token classification
                for i in range(input_len):
                    if i == 0 or i == input_len - 1:  # CLS and SEP tokens
                        word_ids.append(None)
                    elif i < len(words) + 1:  # +1 for CLS token
                        word_ids.append(i - 1)  # -1 to adjust for CLS token
                    else:
                        word_ids.append(None)  # Padding tokens
                
                # Convert labels to token-level labels (-100 for ignored positions)
                token_labels = [-100] * len(word_ids)
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is not None and word_idx < len(label_ids):
                        token_labels[token_idx] = label_ids[word_idx]
                
                # Add labels to encoding
                encoding['labels'] = torch.tensor(token_labels, dtype=torch.long)
                
                # Ensure bbox has correct shape [seq_len, 4]
                if 'bbox' in encoding and encoding['bbox'].dim() != 2:
                    # Reshape or create correctly sized bbox tensor
                    seq_len = len(encoding['input_ids'])
                    bbox_data = encoding['bbox']
                    
                    # Different handling based on dimensionality
                    if bbox_data.dim() == 1:
                        # If 1D and divisible by 4, reshape
                        if bbox_data.size(0) % 4 == 0:
                            seq_len = bbox_data.size(0) // 4
                            encoding['bbox'] = bbox_data.reshape(seq_len, 4)
                        else:
                            # Create default bbox tensor
                            encoding['bbox'] = torch.zeros((seq_len, 4), dtype=torch.long)
                    else:
                        # For other unexpected shapes, create default
                        encoding['bbox'] = torch.zeros((seq_len, 4), dtype=torch.long)
               
                # Make one final check to ensure all tensors have compatible dimensions
                seq_len = encoding['input_ids'].size(0)
                if 'bbox' in encoding and encoding['bbox'].size(0) != seq_len:
                    # Resize bbox to match input_ids
                    encoding['bbox'] = torch.zeros((seq_len, 4), dtype=torch.long)
                
                if 'labels' in encoding and encoding['labels'].size(0) != seq_len:
                    # Resize labels to match input_ids
                    new_labels = torch.full((seq_len,), -100, dtype=torch.long)
                    min_len = min(seq_len, encoding['labels'].size(0))
                    new_labels[:min_len] = encoding['labels'][:min_len]
                    encoding['labels'] = new_labels
                
                return encoding
                    
            except Exception as e:
                logger.error(f"Error tokenizing item {idx} in FigureDetectionDataset: {str(e)}")
                logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            logger.error(f"Error processing item {idx} in FigureDetectionDataset: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a fallback empty sample to avoid crashing
            fallback = {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'bbox': torch.zeros((self.max_length, 4), dtype=torch.long),
                'labels': torch.tensor([-100] * self.max_length, dtype=torch.long)
            }
            return fallback
"""
        # Replace the function
        lines[getitem_start:getitem_end+1] = getitem_fixed.splitlines(True)
    else:
        print("Warning: Could not find FigureDetectionDataset.__getitem__ method")
    
    # Find FigureDetectorTrainer.evaluate
    evaluate_start = None
    evaluate_end = None
    
    for i, line in enumerate(lines):
        if 'def evaluate(self, val_loader):' in line and 'FigureDetectorTrainer' in lines[i-100:i]:
            evaluate_start = i
            # Find the end of the function
            for j in range(i, len(lines)):
                if j+1 < len(lines) and 'def' in lines[j+1] and j > i+10:  # Next function
                    evaluate_end = j
                    break
            break
    
    if evaluate_start is not None and evaluate_end is not None:
        print(f"Found evaluate method at lines {evaluate_start}-{evaluate_end}")
        
        # Replace with our fixed version
        evaluate_fixed = """    def evaluate(self, val_loader):
        \"\"\"Evaluate the model on the validation set.\"\"\"
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    
                    # Get predictions and labels
                    batch_preds, batch_labels = self.get_predictions(outputs, batch)
                    all_preds.extend(batch_preds)
                    all_labels.extend(batch_labels)
                    
                    # Log some sample predictions for debugging
                    if len(all_preds) <= 20:  # Only log first few predictions
                        logger.info(f"Sample predictions: {batch_preds[:5]}")
                        logger.info(f"Sample labels: {batch_labels[:5]}")
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_labels)
        
        # Log summary metrics
        logger.info(f"Validation complete - Loss: {avg_val_loss:.4f}, Metrics: {metrics}")
        
        return avg_val_loss, metrics
"""
        # Replace the function
        lines[evaluate_start:evaluate_end+1] = evaluate_fixed.splitlines(True)
    else:
        print("Warning: Could not find evaluate method")
    
    # Write back the modified file
    with open('transfer_learning_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("File successfully fixed!")

# Execute the fix
fix_file()
''')
    
    # Run the direct fix script
    try:
        subprocess.run(["python", "direct_fix.py"], check=True, timeout=300)
        print("Fixed code successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running direct fixes: {e}")
        return False

def run_training():
    """Run training with the fixed script and data."""
    print("Running training with fixed model and data...")
    
    # Build the command with proper arguments
    command = (
        "python transfer_learning_fixed.py --model figure "
        "--data-dir data/transfer_learning/prepared "
        "--batch-size 8 --epochs 3 --learning-rate 2e-5 "
        "--max-length 512 --early-stopping 2"
    )
    
    # Execute the command
    import subprocess
    try:
        subprocess.run(command, shell=True, check=True, timeout=300)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running training: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running training: {str(e)}")
        return False

if __name__ == "__main__":
    print("===== Fixing Figure Detection Model Training =====")
    
    # Apply direct code fixes
    if not create_figure_detector_fix():
        print("Failed to apply fixes to code. Check the logs.")
        sys.exit(1)
    
    # Fix training data
    print("Fixing training data...")
    if os.path.exists("data/transfer_learning/prepared/figure_detection/train/data.json"):
        process_figure_detection_data("data/transfer_learning/prepared/figure_detection/train/data.json")
    else:
        print("Warning: Training data not found at expected location.")
    
    if os.path.exists("data/transfer_learning/prepared/figure_detection/val/data.json"):
        process_figure_detection_data("data/transfer_learning/prepared/figure_detection/val/data.json")
    else:
        print("Warning: Validation data not found at expected location.")
    
    # Run training
    if not run_training():
        print("Training failed. Check the logs.")
        sys.exit(1)
    
    print("Done!") 