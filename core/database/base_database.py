from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..models.documents import Document
from ..models.auth import AuthContext


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
