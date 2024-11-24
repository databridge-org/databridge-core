from typing import Dict, Any, List, Optional, Union
import httpx
from urllib.parse import urlparse
import jwt
from pydantic import BaseModel
import logging
import base64


logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    """Structure for document ingestion"""
    content: str
    content_type: str
    metadata: Dict[str, Any] = {}
    filename: Optional[str] = None


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
    
    Usage:
        async with DataBridge("databridge://owner123:token@databridge.local") as db:
            doc_id = await db.ingest_document("content", content_type="text/plain")
            results = await db.query("What is...")
    """

    def __init__(self, uri: str, timeout: int = 30):
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._setup_auth(uri)

    def _setup_auth(self, uri: str) -> None:
        """Setup authentication from URI"""
        try:
            parsed = urlparse(uri)
            if not parsed.netloc:
                raise ValueError("Invalid URI format")

            split_uri = parsed.netloc.split('@')
            self._base_url = (
                f"{'http' if 'localhost' in split_uri[1] else 'https'}"
                f"://{split_uri[1]}"
            )

            auth_parts = split_uri[0].split(':')
            if len(auth_parts) != 2:
                raise ValueError("URI must include owner_id and auth_token")

            self._owner_id = auth_parts[0]
            self._auth_token = auth_parts[1]

            # Basic token validation
            try:
                jwt.decode(self._auth_token, options={"verify_signature": False})
            except jwt.InvalidTokenError as e:
                raise ValueError(f"Invalid auth token format: {str(e)}")

        except Exception as e:
            raise ValueError(f"Failed to setup authentication: {str(e)}")

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make authenticated HTTP request"""
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json"
        }

        try:
            response = await self._client.request(
                method,
                f"{self._base_url}/{endpoint.lstrip('/')}",
                json=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Authentication failed")
            raise ConnectionError(f"Request failed: {e.response.text}")

        except Exception as e:
            raise ConnectionError(f"Request failed: {str(e)}")

    async def ingest_document(
        self,
        content: Union[str, bytes],
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> Document:
        """
        Ingest a document into DataBridge.
        
        Args:
            content: Document content (string or bytes)
            content_type: MIME type of the content
            metadata: Optional document metadata
            filename: Optional filename
            
        Returns:
            Document object with metadata
        """
        # Handle content encoding
        if isinstance(content, bytes):
            encoded_content = base64.b64encode(content).decode('utf-8')
        else:
            encoded_content = content

        request = IngestRequest(
            content=encoded_content,
            content_type=content_type,
            metadata=metadata or {},
            filename=filename
        )

        response = await self._request("POST", "documents", request.model_dump())
        return Document(**response)

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
            query: Query string
            return_type: Type of results ("chunks" or "documents")
            filters: Optional metadata filters
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of ChunkResult or DocumentResult objects
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
        limit: int = 100
    ) -> List[Document]:
        """List accessible documents with pagination"""
        response = await self._request(
            "GET",
            f"documents?skip={skip}&limit={limit}"
        )
        return [Document(**doc) for doc in response]

    async def get_document(self, document_id: str) -> Document:
        """Get document by ID"""
        response = await self._request("GET", f"documents/{document_id}")
        return Document(**response)

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
