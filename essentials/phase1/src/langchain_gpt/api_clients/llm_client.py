"""LLM client interface for language model API."""

import abc
from typing import Any, Dict, List, Optional, Union

from ..utils.types import APIResponse, EmbeddingVector, ModelName
from .base_client import BaseAPIClient, MockAPIClient


class BaseLLMClient(BaseAPIClient):
    """Abstract base class for LLM clients."""
    
    @abc.abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: Optional[ModelName] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model parameters
            
        Returns:
            str: Generated text
        """
        pass
    
    @abc.abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[ModelName] = None,
        **kwargs,
    ) -> List[EmbeddingVector]:
        """Generate embeddings for texts.
        
        Args:
            texts: Input texts
            model: Model name
            **kwargs: Additional model parameters
            
        Returns:
            List[EmbeddingVector]: Text embeddings
        """
        pass


class MockLLMClient(MockAPIClient, BaseLLMClient):
    """Mock LLM client for testing."""
    
    def __init__(
        self,
        mock_responses: Optional[Dict[str, Any]] = None,
        default_text_response: str = "This is a mock response from the LLM API.",
        default_embedding_dimension: int = 384,
        **kwargs,
    ):
        """Initialize mock LLM client.
        
        Args:
            mock_responses: Predefined mock responses
            default_text_response: Default text response
            default_embedding_dimension: Default embedding dimension
            **kwargs: Additional arguments
        """
        super().__init__(mock_responses=mock_responses, **kwargs)
        self.default_text_response = default_text_response
        self.default_embedding_dimension = default_embedding_dimension
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[ModelName] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate mock text response.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model parameters
            
        Returns:
            str: Generated text
        """
        endpoint = "generate_text"
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        
        response = await self.call_api(endpoint, payload)
        
        if isinstance(response, dict) and "text" in response:
            return response["text"]
        return self.default_text_response
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[ModelName] = None,
        **kwargs,
    ) -> List[EmbeddingVector]:
        """Generate mock embeddings.
        
        Args:
            texts: Input texts
            model: Model name
            **kwargs: Additional model parameters
            
        Returns:
            List[EmbeddingVector]: Mock embeddings
        """
        endpoint = "generate_embeddings"
        payload = {
            "texts": texts,
            "model": model,
            **kwargs,
        }
        
        response = await self.call_api(endpoint, payload)
        
        if isinstance(response, dict) and "embeddings" in response:
            return response["embeddings"]
        
        # Generate random embeddings as mock
        import random
        return [
            [random.uniform(-1, 1) for _ in range(self.default_embedding_dimension)]
            for _ in texts
        ]


# Factory function to get LLM client
def get_llm_client(
    client_type: str = "mock",
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Get LLM client instance.
    
    Args:
        client_type: Client type ('mock' only for now)
        api_key: API key
        **kwargs: Additional client parameters
        
    Returns:
        BaseLLMClient: LLM client instance
    """
    if client_type == "mock":
        return MockLLMClient(api_key=api_key, **kwargs)
    else:
        # Only mock client is implemented for Phase 1
        # In future phases, we can add support for real LLM clients
        return MockLLMClient(api_key=api_key, **kwargs) 