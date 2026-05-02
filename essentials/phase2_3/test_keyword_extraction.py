import unittest
from keyword_extraction import KeywordExtractor

class TestKeywordExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = KeywordExtractor(max_features=5)

    def test_extract_keywords(self):
        sample_text = "The SARS-CoV-2 virus, responsible for COVID-19, is a novel coronavirus that emerged in Wuhan, China."
        expected_keywords = ["novel", "emerged", "covid19", "coronavirus", "china"]
        keywords = self.extractor.extract_keywords(sample_text)
        self.assertEqual(len(keywords), 5)
        for keyword in expected_keywords:
            self.assertIn(keyword, keywords)

if __name__ == "__main__":
    unittest.main() 