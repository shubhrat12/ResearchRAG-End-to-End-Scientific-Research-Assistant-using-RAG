"""Unit tests for API clients."""

import pytest
from unittest import mock

from langchain_gpt.api_clients.base_client import MockAPIClient
from langchain_gpt.api_clients.llm_client import MockLLMClient, get_llm_client
from langchain_gpt.utils.errors import APIClientError


class TestMockAPIClient:
    """Tests for MockAPIClient."""
    
    def test_init(self):
        """Test initialization."""
        client = MockAPIClient(api_key="test_key", base_url="https://api.example.com")
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.example.com"
        assert client.timeout == 30  # Default value
        assert client.max_retries == 3  # Default value
        assert client.requests == []  # Empty request list initially
        
        # Test with custom timeout and retries
        client = MockAPIClient(timeout=60, max_retries=5)
        assert client.timeout == 60
        assert client.max_retries == 5
    
    @pytest.mark.asyncio
    async def test_call_api(self):
        """Test API call."""
        # Test with default response
        client = MockAPIClient()
        response = await client.call_api("test_endpoint", {"param": "value"})
        assert "status" in response
        assert response["status"] == "success"
        assert len(client.requests) == 1
        
        # Test with custom response
        mock_responses = {"custom_endpoint": {"result": "custom_response"}}
        client = MockAPIClient(mock_responses=mock_responses)
        response = await client.call_api("custom_endpoint", {})
        assert "result" in response
        assert response["result"] == "custom_response"
    
    def test_validate_api_key(self):
        """Test API key validation."""
        client = MockAPIClient()
        assert client.validate_api_key() is True
    
    def test_reset(self):
        """Test reset method."""
        client = MockAPIClient()
        
        # Add some requests
        client.requests = [{"endpoint": "test", "payload": {}}]
        assert len(client.requests) == 1
        
        # Reset client
        client.reset()
        assert len(client.requests) == 0
    
    def test_add_mock_response(self):
        """Test adding mock responses."""
        client = MockAPIClient()
        
        # Add mock response
        client.add_mock_response("test_endpoint", {"result": "test"})
        assert "test_endpoint" in client.mock_responses
        assert client.mock_responses["test_endpoint"]["result"] == "test"


class TestMockLLMClient:
    """Tests for MockLLMClient."""
    
    @pytest.mark.asyncio
    async def test_generate_text(self):
        """Test text generation."""
        client = MockLLMClient(default_text_response="This is a test response")
        
        # Test with default response
        response = await client.generate_text("Test prompt")
        assert response == "This is a test response"
        assert len(client.requests) == 1
        assert client.requests[0]["endpoint"] == "generate_text"
        
        # Test with custom response
        client = MockLLMClient(
            mock_responses={"generate_text": {"text": "Custom response"}}
        )
        response = await client.generate_text("Test prompt")
        assert response == "Custom response"
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self):
        """Test embedding generation."""
        client = MockLLMClient(default_embedding_dimension=5)
        
        # Test with default response (random embeddings)
        texts = ["text1", "text2"]
        embeddings = await client.generate_embeddings(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 5  # Default dimension
        
        # Test with custom response
        custom_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        client = MockLLMClient(
            mock_responses={"generate_embeddings": {"embeddings": custom_embeddings}}
        )
        embeddings = await client.generate_embeddings(texts)
        assert embeddings == custom_embeddings


class TestGetLLMClient:
    """Tests for get_llm_client factory function."""
    
    def test_get_llm_client(self):
        """Test factory function."""
        # Test with mock client type
        client = get_llm_client("mock", api_key="test_key")
        assert isinstance(client, MockLLMClient)
        assert client.api_key == "test_key"
        
        # Test with unsupported client type
        # Should return MockLLMClient for Phase 1
        client = get_llm_client("unsupported", api_key="test_key")
        assert isinstance(client, MockLLMClient) 