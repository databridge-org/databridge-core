from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    """Document metadata model"""

    external_id: str = Field(..., description="Unique document identifier")
    content_type: str = Field(..., description="Content type of the document")
    filename: Optional[str] = Field(None, description="Original filename if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
    storage_info: Dict[str, str] = Field(
        default_factory=dict, description="Storage-related information"
    )
    system_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="System-managed metadata"
    )
    access_control: Dict[str, Any] = Field(
        default_factory=dict, description="Access control information"
    )
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of document chunks")


class ChunkResult(BaseModel):
    """Query result at chunk level"""

    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Parent document ID")
    chunk_number: int = Field(..., description="Chunk sequence number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    content_type: str = Field(..., description="Content type")
    filename: Optional[str] = Field(None, description="Original filename")
    download_url: Optional[str] = Field(None, description="URL to download full document")


class DocumentContent(BaseModel):
    """Represents either a URL or content string"""

    type: Literal["url", "string"] = Field(..., description="Content type (url or string)")
    value: str = Field(..., description="The actual content or URL")
    filename: Optional[str] = Field(None, description="Filename when type is url")

    @field_validator("filename")
    def filename_only_for_url(cls, v, values):
        if values.data.get("type") == "string" and v is not None:
            raise ValueError("filename can only be set when type is url")
        if values.data.get("type") == "url" and v is None:
            raise ValueError("filename is required when type is url")
        return v


class DocumentResult(BaseModel):
    """Query result at document level"""

    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    content: DocumentContent = Field(..., description="Document content or URL")


class CompletionResponse(BaseModel):
    """Completion response model"""

    completion: str
    usage: Dict[str, int]
