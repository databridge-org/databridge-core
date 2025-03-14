from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..models.documents import Document
from ..models.auth import AuthContext
from ..models.graph import Graph


class BaseDatabase(ABC):
    """Base interface for document metadata storage."""

    @abstractmethod
    async def store_document(self, document: Document) -> bool:
        """
        Store document metadata.
        Returns: Success status
        """
        pass

    @abstractmethod
    async def get_document(self, document_id: str, auth: AuthContext) -> Optional[Document]:
        """
        Retrieve document metadata by ID if user has access.
        Returns: Document if found and accessible, None otherwise
        """
        pass
        
    @abstractmethod
    async def get_document_by_filename(self, filename: str, auth: AuthContext) -> Optional[Document]:
        """
        Retrieve document metadata by filename if user has access.
        If multiple documents have the same filename, returns the most recently updated one.
        
        Args:
            filename: The filename to search for
            auth: Authentication context
            
        Returns:
            Document if found and accessible, None otherwise
        """
        pass
        
    @abstractmethod
    async def get_documents_by_id(self, document_ids: List[str], auth: AuthContext) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.
        Only returns documents the user has access to.
        
        Args:
            document_ids: List of document IDs to retrieve
            auth: Authentication context
            
        Returns:
            List of Document objects that were found and user has access to
        """
        pass

    @abstractmethod
    async def get_documents(
        self,
        auth: AuthContext,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        List documents the user has access to.
        Supports pagination and filtering.
        """
        pass

    @abstractmethod
    async def update_document(
        self, document_id: str, updates: Dict[str, Any], auth: AuthContext
    ) -> bool:
        """
        Update document metadata if user has access.
        Returns: Success status
        """
        pass

    @abstractmethod
    async def delete_document(self, document_id: str, auth: AuthContext) -> bool:
        """
        Delete document metadata if user has admin access.
        Returns: Success status
        """
        pass

    @abstractmethod
    async def find_authorized_and_filtered_documents(
        self, auth: AuthContext, filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Find document IDs matching filters that user has access to."""
        pass

    @abstractmethod
    async def check_access(
        self, document_id: str, auth: AuthContext, required_permission: str = "read"
    ) -> bool:
        """
        Check if user has required permission for document.
        Returns: True if user has required access, False otherwise
        """
        pass

    @abstractmethod
    async def store_cache_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Store metadata for a cache.

        Args:
            name: Name of the cache
            metadata: Cache metadata including model info and storage location

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def get_cache_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cache.

        Args:
            name: Name of the cache

        Returns:
            Optional[Dict[str, Any]]: Cache metadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def store_graph(self, graph: Graph) -> bool:
        """Store a graph.

        Args:
            graph: Graph to store

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def get_graph(self, name: str, auth: AuthContext) -> Optional[Graph]:
        """Get a graph by name.

        Args:
            name: Name of the graph
            auth: Authentication context

        Returns:
            Optional[Graph]: Graph if found and accessible, None otherwise
        """
        pass

    @abstractmethod
    async def list_graphs(self, auth: AuthContext) -> List[Graph]:
        """List all graphs the user has access to.

        Args:
            auth: Authentication context

        Returns:
            List[Graph]: List of graphs
        """
        pass
