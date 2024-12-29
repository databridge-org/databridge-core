from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from datetime import UTC, datetime
from pydantic import BaseModel, Field, field_validator
import uuid
import logging

logger = logging.getLogger(__name__)


class QueryReturnType(str, Enum):
    CHUNKS = "chunks"
    DOCUMENTS = "documents"


class Document(BaseModel):
    """Represents a document stored in MongoDB documents collection"""

    external_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner: Dict[str, str]
    content_type: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """user-defined metadata"""
    storage_info: Dict[str, str] = Field(default_factory=dict)
    system_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "version": 1,
        }
    )
    """metadata such as creation date etc."""
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)
    """metadata to help with querying eg. frame descriptions and time-stamped transcript for videos"""
    access_control: Dict[str, List[str]] = Field(
        default_factory=lambda: {"readers": [], "writers": [], "admins": []}
    )
    chunk_ids: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """Represents a chunk stored in VectorStore"""

    document_id: str  # external_id of parent document
    content: str
    embedding: List[float]
    chunk_number: int
    # chunk-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class Chunk(BaseModel):
    """Represents a chunk containing content and metadata"""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_document_chunk(
        self, document_id: str, chunk_number: int, embedding: List[float]
    ) -> DocumentChunk:
        return DocumentChunk(
            document_id=document_id,
            content=self.content,
            embedding=embedding,
            chunk_number=chunk_number,
            metadata=self.metadata,
        )


class ChunkResult(BaseModel):
    """Query result at chunk level"""

    content: str
    score: float
    document_id: str  # external_id
    chunk_number: int
    metadata: Dict[str, Any]
    content_type: str
    filename: Optional[str] = None
    download_url: Optional[str] = None


class DocumentContent(BaseModel):
    """Represents either a URL or content string"""

    type: Literal["url", "string"]
    value: str
    filename: Optional[str] = Field(None, description="Filename when type is url")

    @field_validator("filename")
    def filename_only_for_url(cls, v, values):
        logger.debug(f"Value looks like: {values}")
        if values.data.get("type") == "string" and v is not None:
            raise ValueError("filename can only be set when type is url")
        if values.data.get("type") == "url" and v is None:
            raise ValueError("filename is required when type is url")
        return v


class DocumentResult(BaseModel):
    """Query result at document level"""

    score: float  # Highest chunk score
    document_id: str  # external_id
    metadata: Dict[str, Any]
    content: DocumentContent
