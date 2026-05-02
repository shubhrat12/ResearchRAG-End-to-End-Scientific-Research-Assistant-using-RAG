"""Configuration settings for the LangChainGPT application."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if it exists
load_dotenv()

# Helper function to clean environment variable values
def clean_env_value(value, default=None):
    """Remove comments from environment variable values.
    
    Args:
        value: The environment variable value
        default: Default value if the environment variable is not set
        
    Returns:
        The cleaned value or default
    """
    if value is None:
        return default
    
    # Remove comments (anything after #)
    if "#" in value:
        value = value.split("#")[0].strip()
    
    return value


class LogConfig(BaseModel):
    """Configuration settings for logging."""
    
    level: str = Field(
        default=clean_env_value(os.getenv("LOG_LEVEL"), "INFO"),
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file: Optional[str] = Field(
        default=os.getenv("LOG_FILE"),
        description="Log file path (if None, logs to console only)"
    )


class DocumentProcessingConfig(BaseModel):
    """Configuration settings for document processing."""
    
    max_document_size_mb: int = Field(
        default=int(clean_env_value(os.getenv("MAX_DOCUMENT_SIZE_MB"), 10)),
        description="Maximum document size in MB"
    )
    max_tokens_per_document: int = Field(
        default=int(clean_env_value(os.getenv("MAX_TOKENS_PER_DOCUMENT"), 10000)),
        description="Maximum tokens per document"
    )


class GrobidConfig(BaseModel):
    """Configuration settings for Grobid document processing."""
    
    host: str = Field(
        default=clean_env_value(os.getenv("GROBID_HOST"), "http://localhost"),
        description="Grobid server host"
    )
    port: int = Field(
        default=int(clean_env_value(os.getenv("GROBID_PORT"), 8070)),
        description="Grobid server port"
    )
    timeout: int = Field(
        default=int(clean_env_value(os.getenv("GROBID_TIMEOUT"), 300)),
        description="Grobid request timeout in seconds"
    )
    threads: int = Field(
        default=int(clean_env_value(os.getenv("GROBID_THREADS"), 4)),
        description="Number of threads for Grobid parallel processing"
    )


class VectorDBConfig(BaseModel):
    """Configuration settings for vector database."""
    
    db_type: str = Field(
        default=clean_env_value(os.getenv("VECTOR_DB_TYPE"), "in_memory"),
        description="Vector database type (in_memory, chromadb, faiss)"
    )
    db_dir: Path = Field(
        default=Path(clean_env_value(os.getenv("VECTOR_DB_DIR"), "./data/vector_db")),
        description="Directory for vector database storage"
    )
    collection_name: str = Field(
        default=clean_env_value(os.getenv("VECTOR_DB_COLLECTION"), "research_papers"),
        description="Collection name for vector database"
    )


class EmbeddingConfig(BaseModel):
    """Configuration settings for embeddings."""
    
    model: str = Field(
        default=clean_env_value(os.getenv("EMBEDDING_MODEL"), "all-MiniLM-L6-v2"),
        description="SentenceTransformer model name"
    )
    dimension: int = Field(
        default=int(clean_env_value(os.getenv("EMBEDDING_DIMENSION"), 384)),
        description="Embedding dimension"
    )
    batch_size: int = Field(
        default=int(clean_env_value(os.getenv("EMBEDDING_BATCH_SIZE"), 32)),
        description="Batch size for embedding processing"
    )
    cache_dir: Path = Field(
        default=Path(clean_env_value(os.getenv("EMBEDDING_CACHE_DIR"), "./data/embeddings")),
        description="Directory for embedding cache"
    )
    use_cuda: bool = Field(
        default=clean_env_value(os.getenv("USE_CUDA"), "false").lower() in ("true", "1", "yes"),
        description="Whether to use CUDA for embedding (if available)"
    )


class ArxivConfig(BaseModel):
    """Configuration settings for arXiv API."""
    
    query_limit: int = Field(
        default=int(clean_env_value(os.getenv("ARXIV_QUERY_LIMIT"), 100)),
        description="Maximum number of results per query"
    )
    wait_time: int = Field(
        default=int(clean_env_value(os.getenv("ARXIV_WAIT_TIME"), 3)),
        description="Wait time between requests in seconds"
    )


class SemanticScholarConfig(BaseModel):
    """Configuration settings for Semantic Scholar API."""
    
    api_key: Optional[str] = Field(
        default=clean_env_value(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        description="Semantic Scholar API key (optional)"
    )
    wait_time: int = Field(
        default=int(clean_env_value(os.getenv("SEMANTIC_SCHOLAR_WAIT_TIME"), 3)),
        description="Wait time between requests in seconds"
    )
    cache_dir: Path = Field(
        default=Path(clean_env_value(os.getenv("SEMANTIC_SCHOLAR_CACHE_DIR"), "./data/semantic_scholar_cache")),
        description="Directory for Semantic Scholar cache"
    )


class Settings(BaseModel):
    """Main application settings."""
    
    environment: str = Field(
        default=clean_env_value(os.getenv("ENVIRONMENT"), "development"),
        description="Application environment (development, testing, production)"
    )
    data_dir: Path = Field(
        default=Path(clean_env_value(os.getenv("DATA_DIR"), "./data")),
        description="Directory for data storage"
    )
    cache_dir: Path = Field(
        default=Path(clean_env_value(os.getenv("CACHE_DIR"), "./cache")),
        description="Directory for cache storage"
    )
    temp_dir: Path = Field(
        default=Path(clean_env_value(os.getenv("TEMP_DIR"), "./tmp")),
        description="Directory for temporary files"
    )
    logging: LogConfig = Field(
        default_factory=LogConfig,
        description="Logging configuration"
    )
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig,
        description="Document processing configuration"
    )
    grobid: GrobidConfig = Field(
        default_factory=GrobidConfig,
        description="Grobid configuration"
    )
    vector_db: VectorDBConfig = Field(
        default_factory=VectorDBConfig,
        description="Vector database configuration"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding configuration"
    )
    arxiv: ArxivConfig = Field(
        default_factory=ArxivConfig,
        description="arXiv API configuration"
    )
    semantic_scholar: SemanticScholarConfig = Field(
        default_factory=SemanticScholarConfig,
        description="Semantic Scholar API configuration"
    )
    use_cuda: bool = Field(
        default=clean_env_value(os.getenv("USE_CUDA"), "false").lower() in ("true", "1", "yes"),
        description="Whether to use CUDA (if available)"
    )

    def is_development(self) -> bool:
        """Check if the environment is development."""
        return self.environment.lower() == "development"
    
    def is_testing(self) -> bool:
        """Check if the environment is testing."""
        return self.environment.lower() == "testing"
    
    def is_production(self) -> bool:
        """Check if the environment is production."""
        return self.environment.lower() == "production"
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings.
    
    Returns:
        Settings: The application settings
    """
    return settings 