# Enhanced PDF Processing with Transfer Learning

This module implements transfer learning models for enhanced PDF processing capabilities in LangChainGPT.

## Features

- **Section Classification**: SciBERT-based model to classify document sections
- **Figure Detection**: LayoutLM-based model to detect figures, tables, and other layout elements
- **Reference Parsing**: Sequence labeling model for parsing reference components
- **Unified Inference Pipeline**: Process PDFs with all models in a unified pipeline

## Directory Structure

```
src/langchain_gpt/transfer_learning/
├── data_sampler.py                 # Data sampling and preparation
├── models/
│   ├── section_classifier.py       # Section classification model
│   ├── figure_detector.py          # Figure detection model
│   └── reference_parser.py         # Reference parser model
├── inference_pipeline.py           # Unified inference pipeline
└── train_all.py                    # Main training script
```

## Prerequisites

Before running the transfer learning system, make sure you have extracted the necessary datasets as described in [Phase 2.1 Dataset Setup](../../docs/phase_2.1_setup.md).

Required datasets:
- PubLayNet (under `data/training_datasets/publaynet/`)
- CORD-19 (under `data/training_datasets/cord19/`)

## Installation

Install the required dependencies:

```bash
pip install torch transformers tqdm scikit-learn matplotlib
```

## Usage

The main script `train_all.py` orchestrates the entire transfer learning process:

```bash
# Run all phases (data preparation, training, inference)
python -m langchain_gpt.transfer_learning.train_all --all

# Prepare data only
python -m langchain_gpt.transfer_learning.train_all --prepare-data

# Train specific models
python -m langchain_gpt.transfer_learning.train_all --train-section --train-figure --train-reference

# Run inference on test papers
python -m langchain_gpt.transfer_learning.train_all --run-inference
```

### Command Line Arguments

```
  --data-dir DATA_DIR     Base data directory (default: data)
  --model-dir MODEL_DIR   Base model directory (default: models)
  --log-file LOG_FILE     Log file path (default: logs/transfer_learning.log)
  --sample-size SAMPLE_SIZE
                          Number of CORD-19 papers to sample (default: 5000)
  --prepare-data          Run data preparation phase
  --train-section         Train section classifier
  --train-figure          Train figure detector
  --train-reference       Train reference parser
  --run-inference         Run inference pipeline
  --all                   Run all phases
  --test-dir TEST_DIR     Directory with test papers for inference (default: data/raw/papers)
  --epochs EPOCHS         Number of training epochs (default: 5)
  --batch-size BATCH_SIZE
                          Batch size for training (default: 16)
  --learning-rate LEARNING_RATE
                          Learning rate for training (default: 2e-5)
  --early-stopping EARLY_STOPPING
                          Early stopping patience in epochs (default: 3)
  --no-cuda               Disable CUDA even if available
```

## Training Details

### Section Classification

The section classifier uses SciBERT as the base model with a classification head for predicting section types:
- Introduction, Methods, Results, Discussion, etc.
- Freezes base model layers, only trains the classification head
- Displays validation metrics after each epoch

### Figure Detection

The figure detector uses LayoutLM to detect figures, tables, and other layout elements:
- Processes document layout information
- Detects figure and table bounding boxes
- Fine-tunes with PubLayNet or synthetic data

### Reference Parsing

The reference parser uses a sequence labeling model to parse reference components:
- Authors, Title, Year, Venue, etc.
- Fine-tunes for parsing reference strings
- Implements citation linking logic

## Inference Pipeline

The unified inference pipeline (`EnhancedPDFProcessor`) combines all models to process scientific papers:

```python
from langchain_gpt.transfer_learning.inference_pipeline import EnhancedPDFProcessor

# Initialize processor
processor = EnhancedPDFProcessor()

# Process a single PDF
result = processor.process_pdf("path/to/paper.pdf")

# Process all PDFs in a directory
results = processor.process_directory("path/to/papers/")
```

## Outputs

The processed documents are saved in JSON format under `data/processed/structured/`. Each document includes:
- Classified sections with confidence scores
- Detected figures and tables with bounding boxes
- Parsed references with component identification
- Citation links between in-text citations and references

## Model Training and Validation

Each model is trained with:
- Progress bar and loss indicators
- Validation metrics after each epoch
- Early stopping based on validation performance
- Learning curves visualization
- Comprehensive logging

## GPU Acceleration

The system automatically uses GPU acceleration if available. You can disable it with the `--no-cuda` flag. 