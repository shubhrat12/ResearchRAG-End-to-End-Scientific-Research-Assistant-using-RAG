"""
Basic embeddings module using spaCy to avoid huggingface_hub dependency issues.
"""

from typing import List, Dict, Any, Optional, Union
import os
import numpy as np
import logging
from essentials.phase3_1.models import Chunk, Section
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicEmbedding:
    """Basic embedding implementation using spaCy to avoid huggingface_hub dependency issues."""
    
    def __init__(self, model_name: str = "en_core_web_md"):
        """Initialize basic embedding model.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.model_name = model_name
        
        # Try to load spaCy
        try:
            import spacy
            logger.info(f"Loading spaCy model: {model_name}")
            
            # Try to load the model
            try:
                self.model = spacy.load(model_name)
                self.dimension = self.model.vocab.vectors.shape[1]
                logger.info(f"Loaded spaCy model with dimension: {self.dimension}")
            except OSError:
                # Model not found, try to download it
                logger.warning(f"spaCy model {model_name} not found. Attempting to download...")
                try:
                    os.system(f"python -m spacy download {model_name}")
                    self.model = spacy.load(model_name)
                    self.dimension = self.model.vocab.vectors.shape[1]
                    logger.info(f"Downloaded and loaded spaCy model with dimension: {self.dimension}")
                except Exception as e:
                    logger.error(f"Error downloading spaCy model: {str(e)}")
                    # Try to fall back to a smaller model
                    if model_name != "en_core_web_sm":
                        logger.warning("Falling back to en_core_web_sm")
                        os.system("python -m spacy download en_core_web_sm")
                        self.model = spacy.load("en_core_web_sm")
                        self.dimension = self.model.vocab.vectors.shape[1]
                        logger.info(f"Loaded fallback model with dimension: {self.dimension}")
                    else:
                        raise
        except ImportError:
            logger.error("spaCy not installed. Please install it with: pip install spacy")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text:
            return [0.0] * self.dimension
            
        try:
            # Process the text with spaCy
            doc = self.model(text)
            
            # If the text is long enough, use the document vector
            if len(doc) > 0:
                # Use the average of word vectors as the document vector
                vector = np.mean([token.vector for token in doc if token.has_vector], axis=0)
                
                # If no tokens had vectors, return zeros
                if np.isnan(vector).any():
                    return [0.0] * self.dimension
                    
                return vector.tolist()
            else:
                return [0.0] * self.dimension
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
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
                batch_embeddings = [self.embed_text(text) for text in batch]
                embeddings.extend(batch_embeddings)
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
            result.append({
                "id": chunk.id,
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