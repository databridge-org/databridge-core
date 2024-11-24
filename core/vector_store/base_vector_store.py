from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.models.auth import AuthContext
from core.models.documents import DocumentChunk


class BaseVectorStore(ABC):
    @abstractmethod
    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks and their embeddings"""
        pass

    @abstractmethod
    def query_similar(self, query_embedding: List[float], k: int, auth: AuthContext, filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Find similar chunks"""
        pass
