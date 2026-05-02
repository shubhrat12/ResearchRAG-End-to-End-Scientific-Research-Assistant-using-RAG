"""Abstract base class for API clients."""

import abc
from typing import Any, Dict, List, Optional, Union

from ..utils.errors import APIClientError
from ..utils.logging import get_logger
from ..utils.types import APIResponse

logger = get_logger(__name__)


class BaseAPIClient(abc.ABC):
    """Abstract base class for API clients."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize API client.
        
        Args:
            api_key: API key
            base_url: Base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_name = self.__class__.__name__
    
    @abc.abstractmethod
    async def call_api(self, endpoint: str, payload: Dict[str, Any], **kwargs) -> APIResponse:
        """Make an API call.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            **kwargs: Additional arguments
            
        Returns:
            APIResponse: API response
            
        Raises:
            APIClientError: If API call fails
        """
        pass
    
    @abc.abstractmethod
    def validate_api_key(self) -> bool:
        """Validate API key.
        
        Returns:
            bool: Whether API key is valid
        """
        pass
    
    def raise_error(self, message: str, **kwargs) -> None:
        """Raise API client error.
        
        Args:
            message: Error message
            **kwargs: Additional error information
            
        Raises:
            APIClientError: API client error
        """
        error_message = f"{message}"
        if kwargs:
            error_details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            error_message = f"{error_message} ({error_details})"
        
        logger.error(f"{self.client_name}: {error_message}")
        raise APIClientError(error_message, client_name=self.client_name)
    
    def log_request(self, endpoint: str, payload: Dict[str, Any], **kwargs) -> None:
        """Log API request.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            **kwargs: Additional information to log
        """
        # Don't log the actual API key
        safe_payload = payload.copy()
        if "api_key" in safe_payload:
            safe_payload["api_key"] = "***"
        
        logger.debug(
            f"{self.client_name} request: endpoint={endpoint}, payload={safe_payload}, kwargs={kwargs}"
        )
    
    def log_response(self, response: APIResponse, **kwargs) -> None:
        """Log API response.
        
        Args:
            response: API response
            **kwargs: Additional information to log
        """
        logger.debug(f"{self.client_name} response: {response}")


class MockAPIClient(BaseAPIClient):
    """Mock API client for testing."""
    
    def __init__(
        self,
        mock_responses: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize mock API client.
        
        Args:
            mock_responses: Predefined mock responses
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.mock_responses = mock_responses or {}
        self.requests = []
    
    async def call_api(self, endpoint: str, payload: Dict[str, Any], **kwargs) -> APIResponse:
        """Make a mock API call.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            **kwargs: Additional arguments
            
        Returns:
            APIResponse: Mock API response
        """
        self.log_request(endpoint, payload, **kwargs)
        
        # Record request for testing
        self.requests.append({
            "endpoint": endpoint,
            "payload": payload,
            "kwargs": kwargs,
        })
        
        # Get mock response
        if endpoint in self.mock_responses:
            response = self.mock_responses[endpoint]
        else:
            response = {"status": "success", "message": "Mock response", "data": {}}
        
        self.log_response(response)
        return response
    
    def validate_api_key(self) -> bool:
        """Validate API key.
        
        Returns:
            bool: Always True for mock client
        """
        return True
    
    def add_mock_response(self, endpoint: str, response: Any) -> None:
        """Add a mock response.
        
        Args:
            endpoint: API endpoint
            response: Mock response
        """
        self.mock_responses[endpoint] = response
    
    def reset(self) -> None:
        """Reset mock client state."""
        self.requests = [] 