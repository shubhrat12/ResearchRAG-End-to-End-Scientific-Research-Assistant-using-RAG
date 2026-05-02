"""
Transfer learning for enhanced PDF processing.

This module implements transfer learning models for section classification,
figure detection, and reference parsing.
"""

from .data_sampler import DataSampler
from .inference_pipeline import EnhancedPDFProcessor
from .models.section_classifier import SectionClassifier
from .models.figure_detector import FigureDetector
from .models.reference_parser import ReferenceParser

__all__ = [
    'DataSampler',
    'EnhancedPDFProcessor',
    'SectionClassifier',
    'FigureDetector',
    'ReferenceParser'
] 