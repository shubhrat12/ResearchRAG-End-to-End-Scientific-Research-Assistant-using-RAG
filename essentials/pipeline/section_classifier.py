import logging
import glob
import os
import pickle
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import torch
from essentials.phase3_1.chunking import chunk_document
from essentials.phase3_2.retrieval import Retriever
from essentials.phase3_3.vector_store import ChromaVectorStore
from essentials.phase3_4.context_builder import ContextBuilder
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define a mapping from numerical labels to section names
label_mapping = {
    0: "Introduction",
    1: "Methods",
    2: "Results",
    3: "Discussion",
    4: "Conclusion"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def classify_sections(sections):
    # Initialize the ChromaVectorStore
    vector_store = ChromaVectorStore(persist_directory=str(PROJECT_ROOT / "data/chroma_db"), collection_name="document_collection")
    retriever = Retriever(vector_store)
    context_builder = ContextBuilder(max_tokens=2048)
    
    model_path = str(PROJECT_ROOT / "models/section_classifier/best_model/pytorch_model.bin")
    config_path = str(PROJECT_ROOT / "models/section_classifier/best_model/config.json")
    tokenizer_dir = str(PROJECT_ROOT / "models/section_classifier/best_model/")
    if not os.path.exists(model_path):
        logging.warning("No section classification model found.")
        raise FileNotFoundError("Section classification model not found.")
    logging.info(f"Loading model from {model_path}")
    
    # Load model and tokenizer
    config = BertConfig.from_json_file(config_path)
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(model_path), strict=False)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    model.eval()
    
    classified_sections = []
    embedded_documents = []  # List to store embedded documents for the vector store
    for section in sections:
        if section['text']:
            # Chunk the section text
            chunks = chunk_document(section['text'], strategy='fixed', chunk_size=512, overlap=50)
            for chunk in chunks:
                inputs = tokenizer(chunk.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_label = torch.argmax(outputs.logits, dim=1).item()
                section_name = label_mapping.get(predicted_label, "Unknown")
                classified_sections.append({**section, 'label': section_name, 'chunk': chunk})
                
                # Embed the chunk and prepare for vector store
                embedding = retriever.embedding_model.embed_text(chunk.text)
                embedded_documents.append({
                    "id": chunk.id,
                    "embedding": embedding,
                    "text": chunk.text,
                    "metadata": {
                        "label": section_name,
                        "text": chunk.text
                    }
                })
        else:
            logging.warning("Empty section text encountered.")
            classified_sections.append({**section, 'label': 'Unknown'})
    
    # Add embedded documents to the vector store
    vector_store.add_documents(embedded_documents)
    
    # Retrieve additional documents for each classified section
    for section in classified_sections:
        if section['label'] != 'Unknown':
            query = section['chunk'].text
            results = retriever.retrieve(query, k=3)
            section['additional_documents'] = results
            
            # Build context using ContextBuilder
            context = context_builder.build_context(results, query=query)
            section['context'] = context
    
    return classified_sections 