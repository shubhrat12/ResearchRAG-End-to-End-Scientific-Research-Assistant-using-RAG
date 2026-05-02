from gensim.models import KeyedVectors
from typing import List

class SemanticQueryExpander:
    def __init__(self, model_path: str):
        """Initialize the SemanticQueryExpander with a pre-trained Word2Vec model."""
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def expand_query(self, keywords: List[str], top_n: int = 3) -> List[str]:
        """Expand the query by finding semantically similar terms for each keyword."""
        expanded_keywords = set(keywords)
        for keyword in keywords:
            if keyword in self.model:
                similar_words = self.model.most_similar(keyword, topn=top_n)
                expanded_keywords.update([word for word, _ in similar_words])
        return list(expanded_keywords)

# Example usage
if __name__ == "__main__":
    # Note: You need to download a pre-trained Word2Vec model and provide its path
    model_path = "path/to/word2vec/model.bin"
    expander = SemanticQueryExpander(model_path)
    keywords = ["virus", "vaccine"]
    expanded_keywords = expander.expand_query(keywords)
    print("Expanded Keywords:", expanded_keywords) 