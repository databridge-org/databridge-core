from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.models.completion import CompletionResponse
from core.models.documents import Document


class BaseCache(ABC):
    """Base class for cache implementations.

    This class defines the interface for cache implementations that support
    document ingestion and cache-augmented querying.
    """

    def __init__(
        self, name: str, model: str, gguf_file: str, filters: Dict[str, Any], docs: List[Document]
    ):
        """Initialize the cache with the given parameters.

        Args:
            name: Name of the cache instance
            model: Model identifier
            gguf_file: Path to the GGUF model file
            filters: Filters used to create the cache context
            docs: Initial documents to ingest into the cache
        """
        self.name = name
        self.filters = filters
        self.docs = []  # List of document IDs that have been ingested
        self._initialize(model, gguf_file, docs)

    @abstractmethod
    def _initialize(self, model: str, gguf_file: str, docs: List[Document]) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass

    @abstractmethod
    async def add_docs(self, docs: List[Document]) -> bool:
        """Add documents to the cache.

        Args:
            docs: List of documents to add to the cache

        Returns:
            bool: True if documents were successfully added
        """
        pass

    @abstractmethod
    async def query(self, query: str) -> CompletionResponse:
        """Query the cache for relevant documents and generate a response.

        Args:
            query: Query string to search for relevant documents

        Returns:
            CompletionResponse: Generated response based on cached context
        """
        pass

    @property
    @abstractmethod
    def saveable_state(self) -> bytes:
        """Get the saveable state of the cache as bytes.

        Returns:
            bytes: Serialized state that can be used to restore the cache
        """
        pass
