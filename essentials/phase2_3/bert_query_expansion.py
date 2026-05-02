from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class BERTQueryExpander:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """Initialize the BERTQueryExpander with a pre-trained BERT model."""
        # Check if model_name is a local path or a HuggingFace model name
        local_path = PROJECT_ROOT / model_name
        if local_path.exists():
            model_path = str(local_path)
        else:
            model_path = model_name  # Use as HuggingFace Hub model name

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

    def expand_query(self, keywords: List[str], top_n: int = 3) -> List[str]:
        """Expand the query by finding semantically similar terms for each keyword using BERT embeddings."""
        expanded_keywords = set(keywords)
        keyword_embeddings = self._get_embeddings(keywords)

        # For simplicity, using the same keywords as potential expansion terms
        # In practice, you might use a larger vocabulary or corpus
        for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
            similarities = cosine_similarity([keyword_embedding], keyword_embeddings)[0]
            similar_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Exclude the keyword itself
            expanded_keywords.update([keywords[i] for i in similar_indices])

        return list(expanded_keywords)

    def _get_embeddings(self, words: List[str]) -> List[torch.Tensor]:
        """Get BERT embeddings for a list of words."""
        inputs = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Use the embeddings from the [CLS] token
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

# Example usage
if __name__ == "__main__":
    expander = BERTQueryExpander()
    keywords = ["virus", "vaccine"]
    expanded_keywords = expander.expand_query(keywords)
    print("Expanded Keywords:", expanded_keywords) 