"""Directory management utilities for LangChainGPT."""

import os
from pathlib import Path
from typing import List, Optional, Union

from ..config.settings import get_settings
from .logging import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_project_directories() -> None:
    """Create all required project directories from settings."""
    settings = get_settings()
    
    # Create main data directories
    ensure_directory(settings.data_dir)
    ensure_directory(settings.data_dir / "raw")
    ensure_directory(settings.data_dir / "processed")
    ensure_directory(settings.data_dir / "samples")
    
    # Create vector database directory
    ensure_directory(settings.vector_db.db_dir)
    
    # Create embedding cache directory
    ensure_directory(settings.embedding.cache_dir)
    
    # Create semantic scholar cache directory
    ensure_directory(settings.semantic_scholar.cache_dir)
    
    # Create general cache directory
    ensure_directory(settings.cache_dir)
    
    # Create temp directory
    ensure_directory(settings.temp_dir)
    
    # Create logs directory
    log_dir = Path("logs")
    ensure_directory(log_dir)
    
    # Create models directory
    models_dir = Path("models")
    ensure_directory(models_dir)
    
    logger.info("Created all required project directories")


def list_project_directories() -> List[Path]:
    """List all project directories.
    
    Returns:
        List[Path]: List of project directory paths
    """
    settings = get_settings()
    
    directories = [
        settings.data_dir,
        settings.data_dir / "raw",
        settings.data_dir / "processed",
        settings.data_dir / "samples",
        settings.vector_db.db_dir,
        settings.embedding.cache_dir,
        settings.semantic_scholar.cache_dir,
        settings.cache_dir,
        settings.temp_dir,
        Path("logs"),
        Path("models"),
    ]
    
    return [path for path in directories if path.exists()] 