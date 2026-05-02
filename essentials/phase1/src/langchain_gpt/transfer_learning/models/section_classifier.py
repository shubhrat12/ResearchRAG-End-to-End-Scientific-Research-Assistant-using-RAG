"""
Section classification model based on SciBERT.

This module implements a scientific section classifier using SciBERT as the base model.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AdamW, 
    get_linear_schedule_with_warmup,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Set up logging
logger = logging.getLogger(__name__)

# Define section types
SECTION_TYPES = [
    "abstract", "introduction", "background", "related_work", 
    "methods", "experiments", "results", "discussion", 
    "conclusion", "references", "appendix", "acknowledgements"
]

class SectionClassificationDataset(Dataset):
    """Dataset for section classification."""
    
    def __init__(
        self, 
        data_path: Union[str, Path], 
        tokenizer, 
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        # Create label map
        self.label_map = {label: idx for idx, label in enumerate(SECTION_TYPES)}
        
        logger.info(f"Loaded {len(self.data)} examples from {self.data_path}")
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an example from the dataset.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with encoded inputs
        """
        example = self.data[idx]
        text = example['text']
        section_type = example['section_type']
        
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


class SectionClassifier:
    """
    Section classification model based on SciBERT.
    
    This model uses SciBERT as a base and adds a classification head
    for predicting section types.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = len(SECTION_TYPES),
        model_dir: str = "models/section_classifier",
        freeze_base_model: bool = True,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the section classifier.
        
        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of section types to classify
            model_dir: Directory to save the model
            freeze_base_model: Whether to freeze the base model layers
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_dir = Path(model_dir)
        self.freeze_base_model = freeze_base_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load config and model
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(SECTION_TYPES)},
            label2id={label: i for i, label in enumerate(SECTION_TYPES)}
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
        # Freeze base model layers if required
        if freeze_base_model:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Initialized {model_name} with {num_labels} labels")
        
        # Training metrics
        self.training_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def train(
        self,
        train_data_path: Union[str, Path],
        val_data_path: Union[str, Path],
        epochs: int = 5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 3,
        save_best_model: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for AdamW optimizer
            early_stopping_patience: Number of epochs with no improvement before stopping
            save_best_model: Whether to save the best model based on validation performance
            
        Returns:
            Dictionary with training statistics
        """
        # Prepare datasets
        train_dataset = SectionClassificationDataset(
            train_data_path, 
            self.tokenizer, 
            max_length=self.max_length
        )
        
        val_dataset = SectionClassificationDataset(
            val_data_path, 
            self.tokenizer, 
            max_length=self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
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
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_f1 = 0.0
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            # Progress bar for training
            train_progress_bar = range(len(train_loader))
            try:
                from tqdm import tqdm
                train_progress_bar = tqdm(
                    train_progress_bar, 
                    desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                    position=0
                )
            except ImportError:
                pass
            
            for step, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Update progress bar
                if isinstance(train_progress_bar, tqdm):
                    train_progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Update training stats
            self.training_stats['train_loss'].append(avg_train_loss)
            self.training_stats['val_loss'].append(val_metrics['loss'])
            self.training_stats['val_accuracy'].append(val_metrics['accuracy'])
            self.training_stats['val_f1'].append(val_metrics['f1'])
            
            # Save best model
            if save_best_model and val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model()
                early_stopping_counter = 0
                logger.info(f"Saved new best model with F1: {best_val_f1:.4f}")
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Plot training stats
        self._plot_training_stats()
        
        return self.training_stats
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar for validation
        val_progress_bar = range(len(val_loader))
        try:
            from tqdm import tqdm
            val_progress_bar = tqdm(
                val_progress_bar, 
                desc="Validation", 
                position=0
            )
        except ImportError:
            pass
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Update metrics
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Calculate average loss
        avg_val_loss = val_loss / len(val_loader)
        
        return {
            'loss': avg_val_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict section types for texts.
        
        Args:
            texts: List of section texts
            
        Returns:
            List of dictionaries with prediction results
        """
        self.model.eval()
        
        results = []
        
        # Process in batches
        batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Convert to numpy
            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()
            
            # Create results
            for j, pred in enumerate(preds):
                results.append({
                    'text': batch_texts[j],
                    'section_type': SECTION_TYPES[pred],
                    'confidence': float(probs[j, pred]),
                    'probabilities': {
                        SECTION_TYPES[k]: float(probs[j, k])
                        for k in range(len(SECTION_TYPES))
                    }
                })
        
        return results
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model (if None, uses self.model_dir)
        """
        save_dir = Path(path) if path else self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer, model, and config
        self.tokenizer.save_pretrained(save_dir)
        self.model.save_pretrained(save_dir)
        
        # Save training stats
        with open(save_dir / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from (if None, uses self.model_dir)
        """
        load_dir = Path(path) if path else self.model_dir
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        
        # Move model to device
        self.model.to(self.device)
        
        # Load training stats if available
        stats_path = load_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
        
        logger.info(f"Model loaded from {load_dir}")
    
    def _plot_training_stats(self) -> None:
        """Plot training statistics."""
        # Create figure directory if it doesn't exist
        fig_dir = self.model_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_stats['train_loss'], label='Train')
        plt.plot(self.training_stats['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_dir / "loss.png")
        
        # Plot metrics
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_stats['val_accuracy'], label='Accuracy')
        plt.plot(self.training_stats['val_f1'], label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_dir / "metrics.png")
        
        logger.info(f"Training plots saved to {fig_dir}") 