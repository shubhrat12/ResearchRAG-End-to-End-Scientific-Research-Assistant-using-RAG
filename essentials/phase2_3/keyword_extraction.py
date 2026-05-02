import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class KeywordExtractor:
    def __init__(self, max_features: int = 10):
        """Initialize the KeywordExtractor with a maximum number of features."""
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the given text using TF-IDF."""
        # Preprocess text
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)

        # Fit and transform the text
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_array = self.vectorizer.get_feature_names_out()
        tfidf_sorting = tfidf_matrix.toarray().flatten().argsort()[::-1]

        # Get top keywords
        top_keywords = [feature_array[i] for i in tfidf_sorting[:self.vectorizer.max_features]]
        return top_keywords

# Example usage
if __name__ == "__main__":
    extractor = KeywordExtractor(max_features=5)
    sample_text = "The SARS-CoV-2 virus, responsible for COVID-19, is a novel coronavirus that emerged in Wuhan, China."
    keywords = extractor.extract_keywords(sample_text)
    print("Extracted Keywords:", keywords) 