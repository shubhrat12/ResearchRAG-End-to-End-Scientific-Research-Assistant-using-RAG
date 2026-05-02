"""
Transfer learning models for enhanced PDF processing.

This module contains models for section classification, figure detection,
and reference parsing.
"""

from .section_classifier import SectionClassifier
from .figure_detector import FigureDetector
from .reference_parser import ReferenceParser

__all__ = [
    'SectionClassifier',
    'FigureDetector',
    'ReferenceParser'
] 