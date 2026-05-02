"""
Main training script for transfer learning.

This script orchestrates the entire transfer learning process, from data sampling
to model training and inference.
"""

import os
import sys
import json
import logging
import argparse
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Import modules
from .data_sampler import DataSampler
from .models.section_classifier import SectionClassifier
from .models.figure_detector import FigureDetector
from .models.reference_parser import ReferenceParser
from .inference_pipeline import EnhancedPDFProcessor

# Import logging utilities
from ..utils.logging import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models for enhanced PDF processing")
    
    # General arguments
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Base model directory")
    parser.add_argument("--log-file", type=str, default="logs/transfer_learning.log", help="Log file path")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of CORD-19 papers to sample")
    
    # Phases to run
    parser.add_argument("--prepare-data", action="store_true", help="Run data preparation phase")
    parser.add_argument("--train-section", action="store_true", help="Train section classifier")
    parser.add_argument("--train-figure", action="store_true", help="Train figure detector")
    parser.add_argument("--train-reference", action="store_true", help="Train reference parser")
    parser.add_argument("--run-inference", action="store_true", help="Run inference pipeline")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--test-dir", type=str, default="data/raw/papers", help="Directory with test papers for inference")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--early-stopping", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # If --all is specified, enable all phases
    if args.all:
        args.prepare_data = True
        args.train_section = True
        args.train_figure = True
        args.train_reference = True
        args.run_inference = True
    
    return args


def prepare_data(args):
    """Prepare datasets for transfer learning."""
    logger.info("Starting data preparation")
    
    # Create data sampler
    data_sampler = DataSampler(
        cord19_dir=f"{args.data_dir}/training_datasets/cord19",
        publaynet_dir=f"{args.data_dir}/training_datasets/publaynet",
        output_dir=f"{args.data_dir}/transfer_learning"
    )
    
    # Prepare all datasets
    data_sampler.prepare_all_datasets(args.sample_size)
    
    logger.info("Data preparation completed")


def train_section_classifier(args):
    """Train section classifier model."""
    logger.info("Starting section classifier training")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    # Initialize model
    model = SectionClassifier(
        model_dir=f"{args.model_dir}/section_classifier",
        freeze_base_model=True,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=device
    )
    
    # Train model
    train_data_path = f"{args.data_dir}/transfer_learning/prepared/section_classification/train.json"
    val_data_path = f"{args.data_dir}/transfer_learning/prepared/section_classification/val.json"
    
    logger.info(f"Training with data: {train_data_path}")
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    stats = model.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Section classifier training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    test_data_path = f"{args.data_dir}/transfer_learning/prepared/section_classification/test.json"
    
    try:
        from torch.utils.data import DataLoader
        test_dataset = model.SectionClassificationDataset(
            test_data_path, 
            model.tokenizer, 
            max_length=model.max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        test_metrics = model.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Error evaluating on test set: {str(e)}")
    
    return model


def train_figure_detector(args):
    """Train figure detector model."""
    logger.info("Starting figure detector training")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    # Initialize model
    model = FigureDetector(
        model_dir=f"{args.model_dir}/figure_detector",
        freeze_base_model=False,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size // 2,  # Reduce batch size for LayoutLM
        device=device
    )
    
    # Train model
    train_data_path = f"{args.data_dir}/transfer_learning/prepared/figure_detection/train.json"
    val_data_path = f"{args.data_dir}/transfer_learning/prepared/figure_detection/val.json"
    
    logger.info(f"Training with data: {train_data_path}")
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    stats = model.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Figure detector training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    test_data_path = f"{args.data_dir}/transfer_learning/prepared/figure_detection/test.json"
    
    try:
        from torch.utils.data import DataLoader
        test_dataset = model.FigureDetectionDataset(
            test_data_path, 
            model.tokenizer, 
            max_length=model.max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size // 2,
            shuffle=False
        )
        
        test_metrics = model.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Error evaluating on test set: {str(e)}")
    
    return model


def train_reference_parser(args):
    """Train reference parser model."""
    logger.info("Starting reference parser training")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    # Initialize model
    model = ReferenceParser(
        model_dir=f"{args.model_dir}/reference_parser",
        freeze_base_model=False,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=device
    )
    
    # Train model
    train_data_path = f"{args.data_dir}/transfer_learning/prepared/reference_parsing/train.json"
    val_data_path = f"{args.data_dir}/transfer_learning/prepared/reference_parsing/val.json"
    
    logger.info(f"Training with data: {train_data_path}")
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    stats = model.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Reference parser training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    test_data_path = f"{args.data_dir}/transfer_learning/prepared/reference_parsing/test.json"
    
    try:
        from torch.utils.data import DataLoader
        test_dataset = model.ReferenceLabelingDataset(
            test_data_path, 
            model.tokenizer, 
            max_length=model.max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        test_metrics = model.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Error evaluating on test set: {str(e)}")
    
    return model


def run_inference(args):
    """Run inference pipeline on test papers."""
    logger.info("Starting inference pipeline")
    
    # Set device
    use_gpu = torch.cuda.is_available() and not args.no_cuda
    
    # Initialize processor
    processor = EnhancedPDFProcessor(
        section_classifier_path=f"{args.model_dir}/section_classifier",
        figure_detector_path=f"{args.model_dir}/figure_detector",
        reference_parser_path=f"{args.model_dir}/reference_parser",
        output_dir=f"{args.data_dir}/processed/structured",
        use_gpu=use_gpu
    )
    
    # Process papers in test directory
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.warning(f"Test directory {test_dir} does not exist")
        return []
    
    # Process all papers
    results = processor.process_directory(test_dir)
    
    logger.info(f"Processed {len(results)} papers")
    
    return results


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    log_dir = Path(args.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=args.log_file)
    
    logger.info("Starting transfer learning process")
    logger.info(f"Arguments: {args}")
    
    # Check GPU availability
    if torch.cuda.is_available() and not args.no_cuda:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Create directories
    for directory in [args.data_dir, args.model_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Track overall time
    start_time = time.time()
    
    # Run phases
    if args.prepare_data:
        prepare_data(args)
    
    if args.train_section:
        train_section_classifier(args)
    
    if args.train_figure:
        train_figure_detector(args)
    
    if args.train_reference:
        train_reference_parser(args)
    
    if args.run_inference:
        run_inference(args)
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Transfer learning process completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 