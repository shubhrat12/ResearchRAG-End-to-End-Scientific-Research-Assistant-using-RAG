from typing import List, Dict
from sentence_transformers import SentenceTransformer
from essentials.phase3_1.models import Chunk
from tqdm import tqdm

# Load the SentenceTransformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to embed chunks
def embed_chunks(chunks: List[Chunk], batch_size: int = 16) -> List[Dict]:
    embeddings = []
    cache = {}
    for i in tqdm(range(0, len(chunks), batch_size), desc='Embedding Chunks'):
        batch = chunks[i:i + batch_size]
        texts = [chunk.text for chunk in batch]
        # Check cache
        batch_embeddings = []
        for text in texts:
            if text in cache:
                batch_embeddings.append(cache[text])
            else:
                embedding = model.encode(text)
                cache[text] = embedding
                batch_embeddings.append(embedding)
        # Append results
        for chunk, embedding in zip(batch, batch_embeddings):
            embeddings.append({
                'id': chunk.id,
                'embedding': embedding.tolist(),
                'metadata': chunk.metadata
            })
    return embeddings 