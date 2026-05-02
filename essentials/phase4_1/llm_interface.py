class LLMInterface:
    def __init__(self, primary_model, fallback_model=None):
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def generate_response(self, prompt: str) -> str:
        response = self.primary_model.generate(prompt)
        if not response and self.fallback_model:
            response = self.fallback_model.generate(prompt)
        return response if response else "Error: No response generated."

    def validate_response(self, response: str) -> bool:
        return bool(response.strip())

    # Optional: Implement streaming output
    def generate_streaming_response(self, prompt: str):
        # This is a placeholder for streaming logic
        pass 