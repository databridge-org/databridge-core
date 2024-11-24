import base64
import pytest
from pathlib import Path
import jwt
from datetime import datetime, timedelta, UTC
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient
from fastapi import FastAPI
from core.api import app, get_settings
from core.models.auth import EntityType
from core.database.mongo_database import MongoDatabase
from core.vector_store.mongo_vector_store import MongoDBAtlasVectorStore

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
JWT_SECRET = "your-secret-key-for-signing-tokens"
TEST_USER_ID = "test_user"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment and create test files"""
    # Create test data directory if it doesn't exist
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Create a test text file
    text_file = TEST_DATA_DIR / "test.txt"
    if not text_file.exists():
        text_file.write_text("This is a test document for DataBridge testing.")
    
    # Create a small test PDF if it doesn't exist
    pdf_file = TEST_DATA_DIR / "test.pdf"
    if not pdf_file.exists():
        # Create a minimal PDF for testing
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(str(pdf_file))
            c.drawString(100, 750, "Test PDF Document")
            c.save()
        except ImportError:
            pytest.skip("reportlab not installed, skipping PDF tests")


def create_test_token(
    entity_type: str = "developer",
    entity_id: str = TEST_USER_ID,
    permissions: list = None,
    app_id: str = None,
    expired: bool = False
) -> str:
    """Create a test JWT token"""
    if permissions is None:
        permissions = ["read", "write", "admin"]
        
    payload = {
        "type": entity_type,
        "entity_id": entity_id,
        "permissions": permissions,
        "exp": datetime.now(UTC) + timedelta(days=-1 if expired else 1)
    }

    if app_id:
        payload["app_id"] = app_id

    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def create_auth_header(
    entity_type: str = "developer",
    permissions: list = None,
    expired: bool = False
) -> Dict[str, str]:
    """Create authorization header with test token"""
    token = create_test_token(entity_type, permissions=permissions, expired=expired)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def test_app() -> FastAPI:
    """Create test FastAPI application"""
    # Configure test settings
    settings = get_settings()
    settings.JWT_SECRET_KEY = JWT_SECRET
    return app


@pytest.fixture
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_ingest_text_document(client: AsyncClient):
    """Test ingesting a text document"""
    headers = create_auth_header()

    response = await client.post(
        "/documents/text",
        json={
            "content": "Test content for document ingestion",
            "metadata": {"test": True, "type": "text"}
        },
        headers=headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "text/plain"
    assert data["metadata"]["test"] is True

    return data["external_id"]


@pytest.mark.asyncio
async def test_ingest_file_document(client: AsyncClient):
    """Test ingesting a file (PDF) document"""
    headers = create_auth_header()
    pdf_path = TEST_DATA_DIR / "test.pdf"
    
    if not pdf_path.exists():
        pytest.skip("Test PDF file not available")
    
    # Create form data with file and metadata
    files = {
        "file": ("test.pdf", open(pdf_path, "rb"), "application/pdf")
    }
    data = {
        "metadata": json.dumps({"test": True, "type": "pdf"})
    }
    
    response = await client.post(
        "/documents/file",
        files=files,
        data=data,
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "application/pdf"
    assert "storage_info" in data
    
    return data["external_id"]


@pytest.mark.asyncio
async def test_ingest_error_handling(client: AsyncClient):
    """Test ingestion error cases"""
    headers = create_auth_header()
    
    # Test invalid text request
    response = await client.post(
        "/documents/text",
        json={
            "wrong_field": "Test content"  # Missing required content field
        },
        headers=headers
    )
    assert response.status_code == 422  # Validation error
    
    # Test invalid file request
    response = await client.post(
        "/documents/file",
        files={},  # Missing file
        data={"metadata": "{}"},
        headers=headers
    )
    assert response.status_code == 422  # Validation error
    
    # Test invalid metadata JSON
    pdf_path = TEST_DATA_DIR / "test.pdf"
    if pdf_path.exists():
        files = {
            "file": ("test.pdf", open(pdf_path, "rb"), "application/pdf")
        }
        response = await client.post(
            "/documents/file",
            files=files,
            data={"metadata": "invalid json"},
            headers=headers
        )
        assert response.status_code == 400  # Bad request
    
    # Test oversized content
    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    response = await client.post(
        "/documents/text",
        json={
            "content": large_content,
            "metadata": {}
        },
        headers=headers
    )
    assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
async def test_auth_errors(client: AsyncClient):
    """Test authentication error cases"""
    # Test missing auth header
    response = await client.post("/documents/text")
    assert response.status_code == 401
    
    # Test invalid token
    headers = {"Authorization": "Bearer invalid_token"}
    response = await client.post("/documents/file", headers=headers)
    assert response.status_code == 401
    
    # Test expired token
    headers = create_auth_header(expired=True)
    response = await client.post("/documents/text", headers=headers)
    assert response.status_code == 401
    
    # Test insufficient permissions
    headers = create_auth_header(permissions=["read"])
    response = await client.post(
        "/documents/text",
        json={
            "content": "Test content",
            "metadata": {}
        },
        headers=headers
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_query_chunks(client: AsyncClient):
    """Test querying document chunks"""
    # First ingest a document to query
    doc_id = await test_ingest_text_document(client)
    
    headers = create_auth_header()
    response = await client.post(
        "/query",
        json={
            "query": "test document",
            "return_type": "chunks",
            "k": 2
        },
        headers=headers
    )
    
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert all(isinstance(r["score"], (int, float)) for r in results)
    assert all(r["document_id"] == doc_id for r in results)


@pytest.mark.asyncio
async def test_query_documents(client: AsyncClient):
    """Test querying for full documents"""
    # First ingest a document to query
    doc_id = await test_ingest_text_document(client)
    
    headers = create_auth_header()
    response = await client.post(
        "/query",
        json={
            "query": "test document",
            "return_type": "documents",
            "filters": {"test": True}
        },
        headers=headers
    )
    
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert results[0]["document_id"] == doc_id
    assert "score" in results[0]
    assert "metadata" in results[0]


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient):
    """Test listing documents"""
    # First ingest some documents
    doc_id1 = await test_ingest_text_document(client)
    doc_id2 = await test_ingest_text_document(client)
    
    headers = create_auth_header()
    response = await client.get("/documents", headers=headers)
    
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) >= 2
    doc_ids = [doc["external_id"] for doc in docs]
    assert doc_id1 in doc_ids
    assert doc_id2 in doc_ids


@pytest.mark.asyncio
async def test_get_document(client: AsyncClient):
    """Test getting a specific document"""
    # First ingest a document
    doc_id = await test_ingest_text_document(client)
    
    headers = create_auth_header()
    response = await client.get(f"/documents/{doc_id}", headers=headers)
    
    assert response.status_code == 200
    doc = response.json()
    assert doc["external_id"] == doc_id
    assert "metadata" in doc
    assert "content_type" in doc


@pytest.mark.asyncio
async def test_error_handling(client: AsyncClient):
    """Test error handling scenarios"""
    headers = create_auth_header()
    
    # Test invalid document ID
    response = await client.get("/documents/invalid_id", headers=headers)
    assert response.status_code == 404
    
    # Test invalid query parameters
    response = await client.post(
        "/query",
        json={
            "query": "",  # Empty query
            "k": -1  # Invalid k
        },
        headers=headers
    )
    assert response.status_code == 400
    
    # Test oversized content
    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    response = await client.post(
        "/documents",
        json={
            "content": large_content,
            "content_type": "text/plain"
        },
        headers=headers
    )
    assert response.status_code == 400
