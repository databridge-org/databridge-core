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
    
    # Client reference for update methods
    _client = None
    
    def update_with_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> "Document":
        """
        Update this document with new text content using the specified strategy.
        
        Args:
            content: The new content to add
            filename: Optional new filename for the document
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding
            
        Returns:
            Document: Updated document metadata
        """
        if self._client is None:
            raise ValueError("Document instance not connected to a client. Use a document returned from a DataBridge client method.")
            
        return self._client.update_document_with_text(
            document_id=self.external_id,
            content=content,
            filename=filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali
        )
        
    def update_with_file(
        self,
        file: "Union[str, bytes, BinaryIO, Path]",
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> "Document":
        """
        Update this document with content from a file using the specified strategy.
        
        Args:
            file: File to add (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding
            
        Returns:
            Document: Updated document metadata
        """
        if self._client is None:
            raise ValueError("Document instance not connected to a client. Use a document returned from a DataBridge client method.")
            
        return self._client.update_document_with_file(
            document_id=self.external_id,
            file=file,
            filename=filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali
        )
        
    def update_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> "Document":
        """
        Update this document's metadata only.
        
        Args:
            metadata: Metadata to update
            
        Returns:
            Document: Updated document metadata
        """
        if self._client is None:
            raise ValueError("Document instance not connected to a client. Use a document returned from a DataBridge client method.")
            
        return self._client.update_document_metadata(
            document_id=self.external_id,
            metadata=metadata
        )


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


class ChunkSource(BaseModel):
    """Source information for a chunk used in completion"""
    
    document_id: str = Field(..., description="ID of the source document")
    chunk_number: int = Field(..., description="Chunk number within the document")
    score: Optional[float] = Field(None, description="Relevance score")


class CompletionResponse(BaseModel):
    """Completion response model"""

    completion: str
    usage: Dict[str, int]
    sources: List[ChunkSource] = Field(
        default_factory=list, description="Sources of chunks used in the completion"
    )
    metadata: Optional[Dict[str, Any]] = None


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    content: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    use_colpali: bool = Field(default=False)
