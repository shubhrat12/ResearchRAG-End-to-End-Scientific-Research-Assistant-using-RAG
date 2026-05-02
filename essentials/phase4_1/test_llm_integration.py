import unittest
from llm_runner import LLMRunner
from llm_interface import LLMInterface

class TestLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.primary_model = LLMRunner()
        self.fallback_model = LLMRunner()  # Assuming a second model for fallback
        self.llm_interface = LLMInterface(self.primary_model, self.fallback_model)

    def test_generate_response(self):
        prompt = "What is the capital of France?"
        response = self.llm_interface.generate_response(prompt)
        self.assertTrue(self.llm_interface.validate_response(response))

    def test_fallback_mechanism(self):
        # Simulate primary model failure
        self.primary_model.generate = lambda x: ""
        prompt = "What is the capital of France?"
        response = self.llm_interface.generate_response(prompt)
        self.assertTrue(self.llm_interface.validate_response(response))

if __name__ == '__main__':
    unittest.main() 