"""Error handling utilities for LangChainGPT."""

from typing import Any, Dict, Optional, Type, Union, Callable

from .logging import get_logger

logger = get_logger(__name__)


class LangChainGPTError(Exception):
    """Base exception class for LangChainGPT."""
    
    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class ConfigurationError(LangChainGPTError):
    """Raised when there's an issue with configuration settings."""
    
    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(f"Configuration error: {message}")


class DocumentProcessingError(LangChainGPTError):
    """Raised when there's an issue processing a document."""
    
    def __init__(self, message: str = "Error processing document", document_path: Optional[str] = None):
        self.document_path = document_path
        doc_info = f" for document: {document_path}" if document_path else ""
        super().__init__(f"Document processing error{doc_info}: {message}")


class APIClientError(LangChainGPTError):
    """Raised when there's an issue with an API client."""
    
    def __init__(self, message: str = "API client error", client_name: Optional[str] = None):
        self.client_name = client_name
        client_info = f" in {client_name}" if client_name else ""
        super().__init__(f"API client error{client_info}: {message}")


def handle_error(
    error: Exception,
    error_map: Optional[Dict[Type[Exception], Union[str, Callable[[Exception], Any]]]] = None,
    default_message: str = "An unexpected error occurred",
    log_error: bool = True,
    raise_error: bool = False,
) -> Dict[str, Any]:
    """Handle exceptions in a consistent way.
    
    Args:
        error: The exception to handle
        error_map: Mapping of exception types to messages or handler functions
        default_message: Default error message for unhandled exceptions
        log_error: Whether to log the error
        raise_error: Whether to re-raise the error after handling
        
    Returns:
        Dict[str, Any]: Error information
        
    Raises:
        Exception: Re-raises the original exception if raise_error is True
    """
    error_map = error_map or {}
    error_type = type(error)
    
    # Get error message from map or use default
    if error_type in error_map:
        handler = error_map[error_type]
        if callable(handler):
            message = handler(error)
        else:
            message = handler
    elif isinstance(error, LangChainGPTError):
        message = error.message
    else:
        message = default_message
    
    # Build error response
    error_info = {
        "error": True,
        "error_type": error_type.__name__,
        "message": message,
    }
    
    # Log error
    if log_error:
        logger.error(f"{error_type.__name__}: {message}", exc_info=error)
    
    # Optionally re-raise
    if raise_error:
        raise error
    
    return error_info 