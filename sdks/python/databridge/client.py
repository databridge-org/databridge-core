import json
from typing import Dict, Any, List, Optional, Union, BinaryIO
import httpx
from urllib.parse import urlparse
import jwt
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO


class IngestTextRequest(BaseModel):
    """Request model for text ingestion"""
    content: str
    metadata: Dict[str, Any] = {}


class Document(BaseModel):
    """Document metadata model"""
    external_id: str
    content_type: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = {}
    storage_info: Dict[str, str] = {}
    system_metadata: Dict[str, Any] = {}
    access_control: Dict[str, Any] = {}
    chunk_ids: List[str] = []


class ChunkResult(BaseModel):
    """Query result at chunk level"""
    content: str
    score: float
    document_id: str
    chunk_number: int
    metadata: Dict[str, Any]
    content_type: str
    filename: Optional[str] = None
    download_url: Optional[str] = None


class DocumentResult(BaseModel):
    """Query result at document level"""
    score: float
    document_id: str
    metadata: Dict[str, Any]
    content: Dict[str, str]


class DataBridge:
    """
    DataBridge client for document operations.
    
    Args:
        uri (str): DataBridge URI in the format "databridge://<owner_id>:<token>@<host>"
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
    
    Examples:
        ```python
        async with DataBridge("databridge://owner_id:token@api.databridge.ai") as db:
            # Ingest text
            doc = await db.ingest_text(
                "Sample content",
                metadata={"category": "sample"}
            )
            
            # Query documents
            results = await db.query("search query")
        ```
    """

    def __init__(self, uri: str, timeout: int = 30):
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._setup_auth(uri)

    def _setup_auth(self, uri: str) -> None:
        """Setup authentication from URI"""
        parsed = urlparse(uri)
        if not parsed.netloc:
            raise ValueError("Invalid URI format")

        # Split host and auth parts
        auth, host = parsed.netloc.split('@')
        self._owner_id, self._auth_token = auth.split(':')
            
        # Set base URL
        self._base_url = f"{'http' if 'localhost' in host else 'https'}://{host}"

        # Basic token validation
        jwt.decode(self._auth_token, options={"verify_signature": False})

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make authenticated HTTP request"""
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        
        if not files:
            headers["Content-Type"] = "application/json"

        response = await self._client.request(
            method,
            f"{self._base_url}/{endpoint.lstrip('/')}",
            json=data if not files else None,
            files=files,
            data=data if files else None,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    async def ingest_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Ingest a text document into DataBridge.
        
        Args:
            content: Text content to ingest
            metadata: Optional metadata dictionary
        
        Returns:
            Document: Metadata of the ingested document
        
        Example:
            ```python
            doc = await db.ingest_text(
                "Machine learning is fascinating...",
                metadata={
                    "title": "ML Introduction",
                    "category": "tech"
                }
            )
            ```
        """
        request = IngestTextRequest(
            content=content,
            metadata=metadata or {}
        )

        response = await self._request(
            "POST",
            "ingest/text",
            request.model_dump()
        )
        return Document(**response)

    async def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Ingest a file document into DataBridge.
        
        Args:
            file: File to ingest (path string, bytes, file object, or Path)
            filename: Name of the file
            content_type: MIME type (optional, will be guessed if not provided)
            metadata: Optional metadata dictionary
        
        Returns:
            Document: Metadata of the ingested document
        
        Example:
            ```python
            # From file path
            doc = await db.ingest_file(
                "document.pdf",
                filename="document.pdf",
                content_type="application/pdf",
                metadata={"department": "research"}
            )
            
            # From file object
            with open("document.pdf", "rb") as f:
                doc = await db.ingest_file(f, "document.pdf")
            ```
        """
        # Handle different file input types
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError(f"File not found: {file}")
            with open(file_path, "rb") as f:
                content = f.read()
                file_obj = BytesIO(content)
        elif isinstance(file, bytes):
            file_obj = BytesIO(file)
        else:
            file_obj = file

        try:
            # Prepare multipart form data
            files = {
                "file": (filename, file_obj, content_type or "application/octet-stream")
            }

            # Add metadata
            data = {"metadata": json.dumps(metadata or {})}

            response = await self._request(
                "POST",
                "ingest/file",
                data=data,
                files=files
            )
            return Document(**response)
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    async def query(
        self,
        query: str,
        return_type: str = "chunks",
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0
    ) -> Union[List[ChunkResult], List[DocumentResult]]:
        """
        Query documents in DataBridge.

        Args:
            query: Search query text
            return_type: Type of results ("chunks" or "documents")
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)

        Returns:
            List[ChunkResult] or List[DocumentResult] depending on return_type

        Example:
            ```python
            # Query for chunks
            chunks = await db.query(
                "What are the key findings?",
                return_type="chunks",
                filters={"department": "research"}
            )

            # Query for documents
            docs = await db.query(
                "machine learning",
                return_type="documents",
                k=5
            )
            ```
        """
        request = {
            "query": query,
            "return_type": return_type,
            "filters": filters,
            "k": k,
            "min_score": min_score
        }

        response = await self._request("POST", "query", request)

        if return_type == "chunks":
            return [ChunkResult(**r) for r in response]
        return [DocumentResult(**r) for r in response]

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        List accessible documents.
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters

        Returns:
            List[Document]: List of accessible documents
        
        Example:
            ```python
            # Get first page
            docs = await db.list_documents(limit=10)
            
            # Get next page
            next_page = await db.list_documents(skip=10, limit=10, filters={"department": "research"})
            ```
        """
        response = await self._request(
            "GET",
            f"documents?skip={skip}&limit={limit}&filters={filters}"
        )
        return [Document(**doc) for doc in response]

    async def get_document(self, document_id: str) -> Document:
        """
        Get document metadata by ID.
        
        Args:
            document_id: ID of the document
        
        Returns:
            Document: Document metadata
        
        Example:
            ```python
            doc = await db.get_document("doc_123")
            print(f"Title: {doc.metadata.get('title')}")
            ```
        """
        response = await self._request("GET", f"documents/{document_id}")
        return Document(**response)

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
