from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from core.models.auth import AuthContext
from core.models.documents import DocumentChunk


class BaseVectorStore(ABC):
    @abstractmethod
    def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks and their embeddings"""
        pass

    @abstractmethod
    def query_similar(self, query_embedding: List[float], k: int, auth: AuthContext, doc_ids: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Find similar chunks"""
        pass
