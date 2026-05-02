from typing import List, Dict, Any, Optional, Union
import os
import numpy as np
import warnings
import logging
import uuid
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback import mechanism with compatibility handling
try:
    from sentence_transformers import SentenceTransformer
    logger.info("Successfully imported SentenceTransformer")
except ImportError as e:
    logger.error(f"Error importing SentenceTransformer: {str(e)}")
    raise

# Import models from phase3_1
from essentials.phase3_1.models import Chunk, Section

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class ModelDownloadManager:
    """Handles model downloading with compatibility across different huggingface_hub versions."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the model download manager.
        
        Args:
            cache_dir: Directory to cache models
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "langchain_scientific_models")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using model cache directory: {self.cache_dir}")
        
        # Check huggingface_hub version and capabilities
        self.hf_hub_available = self._check_hf_hub()
    
    def _check_hf_hub(self) -> bool:
        """Check if huggingface_hub is available and which version."""
        try:
            import huggingface_hub
            logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
            
            # Check for list_repo_tree function
            if hasattr(huggingface_hub, "list_repo_tree"):
                logger.info("huggingface_hub.list_repo_tree is available")
            else:
                logger.warning("huggingface_hub.list_repo_tree is not available")
                
            return True
        except ImportError:
            logger.warning("huggingface_hub is not available")
            return False
    
    def get_model_path(self, model_name: str) -> str:
        """Get the path to a model, downloading it if necessary.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Path to the model
        """
        # Check if it's a local path
        if os.path.exists(model_name):
            logger.info(f"Using local model: {model_name}")
            return model_name
        
        # Check if it's already in cache
        model_cache_dir = os.path.join(self.cache_dir, model_name.replace("/", "_"))
        if os.path.exists(model_cache_dir):
            logger.info(f"Using cached model: {model_cache_dir}")
            return model_cache_dir
        
        # Model needs to be downloaded - use sentence_transformers directly
        # This avoids calling huggingface_hub directly
        logger.info(f"Model will be downloaded via SentenceTransformer: {model_name}")
        return model_name

class ScientificEmbedding:
    """Specialized embeddings for scientific content."""
    
    AVAILABLE_MODELS = {
        "general": "all-MiniLM-L6-v2",  # General purpose, balanced model
        "scientific": "allenai/specter",  # Specialized for scientific papers
        "biomedical": "pritamdeka/S-PubMedBert-MS-MARCO",  # Specialized for biomedical content
        "large": "sentence-transformers/all-mpnet-base-v2"  # Larger, more powerful model
    }
    
    def __init__(self, model_name: str = "general", cache_dir: Optional[str] = None):
        """Initialize scientific embedding model with compatibility handling.
        
        Args:
            model_name: Name of the model to use (one of AVAILABLE_MODELS keys)
            cache_dir: Directory to cache models
        """
        # Set up model downloader
        self.downloader = ModelDownloadManager(cache_dir)
        
        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            self.model_name = model_name
            self.model_path = self.AVAILABLE_MODELS[model_name]
        else:
            self.model_name = model_name
            self.model_path = model_name  # Assume direct model path
        
        logger.info(f"Initializing scientific embedding with model: {self.model_path}")
        
        # Get model path with download handling
        try:
            resolved_path = self.downloader.get_model_path(self.model_path)
            logger.info(f"Loading model from: {resolved_path}")
            self.model = SentenceTransformer(resolved_path)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to general model if specified model fails
            if model_name != "general" and model_name in self.AVAILABLE_MODELS:
                logger.warning(f"Falling back to general model: {self.AVAILABLE_MODELS['general']}")
                self.model_path = self.AVAILABLE_MODELS["general"]
                self.model = SentenceTransformer(self.model_path)
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
    
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Process in batches to handle memory constraints
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings.tolist())
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch of texts: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in texts]
    
    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 16, debug: bool = False, debug_dump_path: str = None) -> List[Dict]:
        """Embed document chunks with diagnostics and debug mode."""
        total_chunks = len(chunks)
        non_empty_chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
        if total_chunks != len(non_empty_chunks):
            logger.warning(f"Skipped {total_chunks - len(non_empty_chunks)} empty chunks during embedding.")
        chunks = non_empty_chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size)
        all_zero_count = sum(1 for emb in embeddings if np.allclose(emb, 0))
        logger.info(f"Embedding diagnostics: total={total_chunks}, non_empty={len(chunks)}, all_zero={all_zero_count}")
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = getattr(chunk, 'id', None)
            if not chunk_id:
                chunk_id = chunk.metadata.get('document_id') if hasattr(chunk, 'metadata') and chunk.metadata else None
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
            result.append({
                "id": chunk_id,
                "embedding": embedding,
                "metadata": chunk.metadata
            })
        if debug and debug_dump_path:
            try:
                with open(debug_dump_path, "w", encoding="utf-8") as f:
                    for i, chunk in enumerate(chunks[:20]):
                        f.write(f"Chunk {i}: {chunk.text[:200]}\n\n")
                logger.info(f"Dumped top-{min(20, len(chunks))} chunk texts to {debug_dump_path}")
            except Exception as e:
                logger.error(f"Failed to dump debug texts: {e}")
        return result
    
    def embed_section_weighted(self, section: Section, 
                              title_weight: float = 2.0,
                              first_para_weight: float = 1.5) -> List[float]:
        """Create a weighted embedding for a document section.
        
        Args:
            section: Document section
            title_weight: Weight for the section title
            first_para_weight: Weight for the first paragraph
            
        Returns:
            Weighted embedding vector
        """
        # Embed title and content separately
        title_embedding = np.array(self.embed_text(section.title))
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
        
        if not paragraphs:
            # Only title available
            return title_embedding.tolist()
        
        # Embed first paragraph
        first_para_embedding = np.array(self.embed_text(paragraphs[0]))
        
        # Embed remaining content if available
        remaining_content = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
        
        if remaining_content:
            content_embedding = np.array(self.embed_text(remaining_content))
            
            # Calculate weighted average
            total_weight = title_weight + first_para_weight + 1.0
            weighted_embedding = (
                (title_weight * title_embedding) + 
                (first_para_weight * first_para_embedding) + 
                content_embedding
            ) / total_weight
        else:
            # Only title and first paragraph
            total_weight = title_weight + first_para_weight
            weighted_embedding = (
                (title_weight * title_embedding) + 
                (first_para_weight * first_para_embedding)
            ) / total_weight
        
        return weighted_embedding.tolist()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # Handle zero vectors
            
        vec1 = vec1 / norm1
        vec2 = vec2 / norm2
        
        # Calculate cosine similarity
        return float(np.dot(vec1, vec2)) 