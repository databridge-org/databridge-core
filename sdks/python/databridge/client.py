from typing import Dict, Any, List, Optional, Union
import httpx
from urllib.parse import urlparse
import jwt
from datetime import datetime, UTC
import asyncio
from dataclasses import dataclass
from .exceptions import AuthenticationError
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured query result"""
    content: str
    doc_id: str
    score: Optional[float]
    metadata: Dict[str, Any]

# Request/Response Models
class Document(BaseModel):
    id: str
    name: str
    type: str
    source: str
    uploaded_at: str
    size: str
    redaction_level: str
    stats: Dict[str, Union[int, str]] = Field(
        default_factory=lambda: {
            "ai_queries": 0,
            "time_saved": "0h",
            "last_accessed": ""
        }
    )
    accessed_by: List[Dict[str, str]] = Field(default_factory=list)
    sensitive_content: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    presigned_url: Optional[str] = None


class DataBridge:
    """
    DataBridge client for document ingestion and querying.

    Usage:
        db = DataBridge("databridge://owner123:token@databridge.local")
        doc_id = await db.ingest_document("content", {"title": "My Doc"})
        results = await db.query("What is...")
    """
    def __init__(
        self,
        uri: str,
        timeout: int = 120,  # Increased timeout for long-running operations
        max_retries: int = 3
    ):
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=timeout, read=timeout),  # Set both connect and read timeouts
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self._setup_auth(uri)

    def _setup_auth(self, uri: str) -> None:
        """Setup authentication from URI"""
        try:
            parsed = urlparse(uri)
            if not parsed.netloc:
                raise ValueError("Invalid URI format")
            
            split_uri = parsed.netloc.split('@')
            self._base_url = f"{"http" if "localhost" in split_uri[1] else "https"}://{split_uri[1]}"
            auth_parts = split_uri[0].split(':')
            if len(auth_parts) != 2:
                raise ValueError("URI must include owner_id and auth_token")

            if '.' in auth_parts[0]:
                self._owner_id = auth_parts[0]  # dev_id.app_id format
                self._auth_token = auth_parts[1]
            else:
                self._owner_id = auth_parts[0]  # eu_id format
                self._auth_token = auth_parts[1]

            # Validate token structure (not signature)
            try:
                decoded = jwt.decode(self._auth_token, options={"verify_signature": False})
                self._token_expiry = datetime.fromtimestamp(decoded['exp'], UTC)
            except jwt.InvalidTokenError as e:
                raise ValueError(f"Invalid auth token format: {str(e)}")

        except Exception as e:
            raise AuthenticationError(f"Failed to setup authentication: {str(e)}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Make authenticated HTTP request with retries"""
        # if datetime.now(UTC) > self._token_expiry:
        #     raise AuthenticationError("Authentication token has expired")
        headers = {
            "X-Owner-ID": self._owner_id,
            "X-Auth-Token": self._auth_token,
            "Content-Type": "application/json"
        }

        try:
            response = await self._client.request(
                method,
                f"{self._base_url}/{endpoint.lstrip('/')}",
                json=data,
                headers=headers
            )

            # For long-running operations, check if request was accepted
            if response.status_code == 202:  # Accepted
                logger.info("Request accepted, waiting for completion...")
                # Could implement polling here if needed
                return {"status": "accepted"}

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            logger.warning(f"Request timed out but may still be processing: {str(e)}")
            return {"status": "timeout", "message": "Request is still processing on the server"}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Authentication failed: " + str(e))
            elif e.response.status_code >= 500 and retry_count < self._max_retries:
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._make_request(method, endpoint, data, retry_count + 1)
            else:
                raise ConnectionError(f"Request failed: {e.response.text}")
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            raise ConnectionError(f"Request failed: {str(e)}")

    async def ingest_document(
        self,
        content: Union[str, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Ingest a document into DataBridge.
        
        Args:
            content: Document content (string or bytes)
            metadata: Optional document metadata
            filename: Optional filename - defaults to doc_id if not provided
        Returns:
            Document ID of the ingested document
        """
        try:
            metadata = metadata or {}
            if filename:
                metadata["filename"] = filename

            if isinstance(content, bytes):
                import base64
                content = base64.b64encode(content).decode()
                metadata["is_base64"] = True

            response = await self._make_request(
                "POST",
                "ingest",
                {
                    "content": content,
                    "metadata": metadata
                }
            )

            if response.get("status") in ["accepted", "timeout"]:
                logger.info("Document ingestion is being processed asynchronously")
                return "processing"  # Or return a job ID if the API provides one

            if not response or "document_id" not in response:
                raise ValueError("Invalid response from server")

            return response["document_id"]

        except Exception as e:
            logger.error(f"Ingestion error: {str(e)}")
            raise ConnectionError(f"Document ingestion failed: {str(e)}")

    async def query(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """
        Query documents in DataBridge.
        
        Args:
            query: Query string
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of QueryResult objects
        """
        response = await self._make_request(
            "POST",
            "query",
            {
                "query": query,
                "k": k,
                "filters": filters
            }
        )

        return [
            QueryResult(
                content=result["content"],
                doc_id=result["doc_id"],
                score=result.get("score"),
                metadata=result.get("metadata", {})
            )
            for result in response["results"]
        ]
    
    async def get_documents(self) -> List[Document]:
        """Get all documents"""
        response = await self._make_request("GET", "documents")
        return [Document(**doc) for doc in response]

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def __repr__(self) -> str:
        """Safe string representation"""
        return f"DataBridge(owner_id='{self._owner_id}')"
