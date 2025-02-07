import asyncio
import json
import pytest
from pathlib import Path
import jwt
from datetime import datetime, timedelta, UTC
from typing import AsyncGenerator, Dict
from httpx import AsyncClient
from fastapi import FastAPI
from core.api import app
from core.shared import settings
import mimetypes
import logging
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger(__name__)


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
JWT_SECRET = "your-secret-key-for-signing-tokens"
TEST_USER_ID = "test_user"
TEST_POSTGRES_URI = "postgresql+asyncpg://postgres:postgres@localhost:5432/databridge_test"


async def setup_test_postgres():
    """Setup test PostgreSQL database"""
    engine = create_async_engine(TEST_POSTGRES_URI)
    try:
        async with engine.begin() as conn:
            # Drop all tables first
            from core.database.postgres_database import Base

            await conn.run_sync(Base.metadata.drop_all)

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)

            logger.info("Test PostgreSQL database setup completed")
    except Exception as e:
        logger.error(f"Failed to setup test PostgreSQL database: {e}")
        raise
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment(event_loop):
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
        pytest.skip("PDF file not available, skipping PDF tests")

    # Setup test PostgreSQL database
    if settings.DATABASE_PROVIDER == "postgres":
        await setup_test_postgres()


def create_test_token(
    entity_type: str = "developer",
    entity_id: str = TEST_USER_ID,
    permissions: list = None,
    app_id: str = None,
    expired: bool = False,
) -> str:
    """Create a test JWT token"""
    if not permissions:
        permissions = ["read", "write", "admin"]

    payload = {
        "type": entity_type,
        "entity_id": entity_id,
        "permissions": permissions,
        "exp": datetime.now(UTC) + timedelta(days=-1 if expired else 1),
    }

    if app_id:
        payload["app_id"] = app_id

    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def create_auth_header(
    entity_type: str = "developer", permissions: list = None, expired: bool = False
) -> Dict[str, str]:
    """Create authorization header with test token"""
    token = create_test_token(entity_type, permissions=permissions, expired=expired)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def test_app(event_loop: asyncio.AbstractEventLoop) -> FastAPI:
    """Create test FastAPI application"""
    # Configure test settings
    settings.JWT_SECRET_KEY = JWT_SECRET
    return app


@pytest.fixture
async def client(
    test_app: FastAPI, event_loop: asyncio.AbstractEventLoop
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_ingest_text_document(
    client: AsyncClient, content: str = "Test content for document ingestion"
):
    """Test ingesting a text document"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/text",
        json={"content": content, "metadata": {"test": True, "type": "text"}},
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "text/plain"
    assert data["metadata"]["test"] is True

    return data["external_id"]


@pytest.mark.asyncio
async def test_ingest_pdf(client: AsyncClient):
    """Test ingesting a pdf"""
    headers = create_auth_header()
    pdf_path = TEST_DATA_DIR / "test.pdf"

    if not pdf_path.exists():
        pytest.skip("Test PDF file not available")

    content_type, _ = mimetypes.guess_type(pdf_path)
    if not content_type:
        content_type = "application/octet-stream"

    with open(pdf_path, "rb") as f:
        response = await client.post(
            "/ingest/file",
            files={"file": (pdf_path.name, f, content_type)},
            data={"metadata": json.dumps({"test": True, "type": "pdf"})},
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "application/pdf"
    assert "storage_info" in data

    return data["external_id"]


@pytest.mark.asyncio
async def test_ingest_invalid_text_request(client: AsyncClient):
    """Test ingestion with invalid text request missing required content field"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/text",
        json={"wrong_field": "Test content"},  # Missing required content field
        headers=headers,
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_invalid_file_request(client: AsyncClient):
    """Test ingestion with invalid file request missing file"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/file",
        files={},  # Missing file
        data={"metadata": "{}"},
        headers=headers,
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_invalid_metadata(client: AsyncClient):
    """Test ingestion with invalid metadata JSON"""
    headers = create_auth_header()

    pdf_path = TEST_DATA_DIR / "test.pdf"
    if pdf_path.exists():
        files = {"file": ("test.pdf", open(pdf_path, "rb"), "application/pdf")}
        response = await client.post(
            "/ingest/file",
            files=files,
            data={"metadata": "invalid json"},
            headers=headers,
        )
        assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
@pytest.mark.skipif(
    settings.EMBEDDING_PROVIDER == "ollama",
    reason="local embedding models do not have size limits",
)
async def test_ingest_oversized_content(client: AsyncClient):
    """Test ingestion with oversized content"""
    headers = create_auth_header()
    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    response = await client.post(
        "/ingest/text", json={"content": large_content, "metadata": {}}, headers=headers
    )
    assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
async def test_auth_missing_header(client: AsyncClient):
    """Test authentication with missing auth header"""
    if settings.dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    response = await client.post("/ingest/text")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_invalid_token(client: AsyncClient):
    """Test authentication with invalid token"""
    if settings.dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    headers = {"Authorization": "Bearer invalid_token"}
    response = await client.post("/ingest/file", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_expired_token(client: AsyncClient):
    """Test authentication with expired token"""
    if settings.dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    headers = create_auth_header(expired=True)
    response = await client.post("/ingest/text", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_insufficient_permissions(client: AsyncClient):
    """Test authentication with insufficient permissions"""
    if settings.dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    headers = create_auth_header(permissions=["read"])
    response = await client.post(
        "/ingest/text",
        json={"content": "Test content", "metadata": {}},
        headers=headers,
    )
    assert response.status_code == 403


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
async def test_invalid_document_id(client: AsyncClient):
    """Test error handling for invalid document ID"""
    headers = create_auth_header()
    response = await client.get("/documents/invalid_id", headers=headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retrieve_chunks(client: AsyncClient):
    """Test retrieving document chunks"""
    upload_string = "The quick brown fox jumps over the lazy dog"
    # First ingest a document to search
    doc_id = await test_ingest_text_document(client, content=upload_string)

    headers = create_auth_header()

    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "jumping fox",
            "k": 1,
            "min_score": 0.0,
            "filters": {"external_id": doc_id},  # Add filter for specific document
        },
        headers=headers,
    )

    assert response.status_code == 200
    results = list(response.json())
    assert len(results) > 0
    assert (not settings.USE_RERANKING) or results[0]["score"] > 0.5
    assert any(upload_string == result["content"] for result in results)


@pytest.mark.asyncio
async def test_retrieve_docs(client: AsyncClient):
    """Test retrieving full documents"""
    # First ingest a document to search
    content = (
        "Headaches can significantly impact daily life and wellbeing. "
        "Common triggers include stress, dehydration, and poor sleep habits. "
        "While over-the-counter pain relievers may provide temporary relief, "
        "it's important to identify and address the root causes. "
        "Maintaining good health through proper nutrition, regular exercise, "
        "and stress management can help prevent chronic headaches."
    )
    doc_id = await test_ingest_text_document(client, content=content)

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/docs",
        json={
            "query": "Headaches, dehydration",
            "filters": {"test": True, "external_id": doc_id},  # Add filter for specific document
        },
        headers=headers,
    )

    assert response.status_code == 200
    results = list(response.json())
    assert len(results) > 0
    assert results[0]["document_id"] == doc_id
    assert "score" in results[0]
    assert "metadata" in results[0]


@pytest.mark.asyncio
async def test_query_completion(client: AsyncClient):
    """Test generating completions from context"""
    # First ingest a document to use as context
    content = (
        "The benefits of exercise are numerous. Regular physical activity "
        "can improve cardiovascular health, strengthen muscles, enhance mental "
        "wellbeing, and help maintain a healthy weight. Studies show that "
        "even moderate exercise like walking can significantly reduce the risk "
        "of various health conditions."
    )
    await test_ingest_text_document(client, content=content)

    headers = create_auth_header()
    response = await client.post(
        "/query",
        json={
            "query": "What are the main benefits of exercise?",
            "k": 2,
            "temperature": 0.7,
            "max_tokens": 100,
        },
        headers=headers,
    )

    assert response.status_code == 200
    result = response.json()
    assert "completion" in result
    assert "usage" in result
    assert len(result["completion"]) > 0


@pytest.mark.asyncio
async def test_invalid_retrieve_params(client: AsyncClient):
    """Test error handling for invalid retrieve parameters"""
    headers = create_auth_header()

    # Test empty query
    response = await client.post(
        "/retrieve/chunks", json={"query": "", "k": 1}, headers=headers  # Empty query
    )
    assert response.status_code == 422

    # Test invalid k
    response = await client.post(
        "/retrieve/docs", json={"query": "test", "k": -1}, headers=headers  # Invalid k
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_completion_params(client: AsyncClient):
    """Test error handling for invalid completion parameters"""
    headers = create_auth_header()

    # Test empty query
    response = await client.post(
        "/query",
        json={"query": "", "temperature": 2.0},  # Empty query  # Invalid temperature
        headers=headers,
    )
    assert response.status_code == 422


@pytest.fixture(autouse=True)
async def cleanup_database():
    """Clean up database before each test"""
    if settings.DATABASE_PROVIDER == "postgres":
        await setup_test_postgres()


@pytest.mark.asyncio
async def test_retrieve_chunks_default_reranking(client: AsyncClient):
    """Test retrieving chunks with default reranking behavior"""
    # First ingest some test documents
    _ = await test_ingest_text_document(
        client, "The quick brown fox jumps over the lazy dog. This is a test document."
    )
    _ = await test_ingest_text_document(
        client, "The lazy dog sleeps while the quick brown fox runs. Another test document."
    )

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "What does the fox do?",
            "k": 2,
            "min_score": 0.0,
            # Not specifying use_reranking - should use default from config
        },
        headers=headers,
    )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0
    # Verify chunks are ordered by score
    scores = [chunk["score"] for chunk in chunks]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


@pytest.mark.asyncio
async def test_retrieve_chunks_explicit_reranking(client: AsyncClient):
    """Test retrieving chunks with explicitly enabled reranking"""
    # First ingest some test documents
    _ = await test_ingest_text_document(
        client, "The quick brown fox jumps over the lazy dog. This is a test document."
    )
    _ = await test_ingest_text_document(
        client, "The lazy dog sleeps while the quick brown fox runs. Another test document."
    )

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "What does the fox do?",
            "k": 2,
            "min_score": 0.0,
            "use_reranking": True,
        },
        headers=headers,
    )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0
    # Verify chunks are ordered by score
    scores = [chunk["score"] for chunk in chunks]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


@pytest.mark.asyncio
async def test_retrieve_chunks_disabled_reranking(client: AsyncClient):
    """Test retrieving chunks with explicitly disabled reranking"""
    # First ingest some test documents
    await test_ingest_text_document(
        client, "The quick brown fox jumps over the lazy dog. This is a test document."
    )
    await test_ingest_text_document(
        client, "The lazy dog sleeps while the quick brown fox runs. Another test document."
    )

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "What does the fox do?",
            "k": 2,
            "min_score": 0.0,
            "use_reranking": False,
        },
        headers=headers,
    )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_reranking_affects_results(client: AsyncClient):
    """Test that reranking actually changes the order of results"""
    # First ingest documents with clearly different semantic relevance
    await test_ingest_text_document(
        client, "The capital of France is Paris. The city is known for the Eiffel Tower."
    )
    await test_ingest_text_document(
        client, "Paris is a city in France. It has many famous landmarks and museums."
    )
    await test_ingest_text_document(
        client, "Paris Hilton is a celebrity and businesswoman. She has nothing to do with France."
    )

    headers = create_auth_header()

    # Get results without reranking
    response_no_rerank = await client.post(
        "/retrieve/chunks",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": False,
        },
        headers=headers,
    )

    # Get results with reranking
    response_with_rerank = await client.post(
        "/retrieve/chunks",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": True,
        },
        headers=headers,
    )

    assert response_no_rerank.status_code == 200
    assert response_with_rerank.status_code == 200

    chunks_no_rerank = response_no_rerank.json()
    chunks_with_rerank = response_with_rerank.json()

    # Verify we got results in both cases
    assert len(chunks_no_rerank) > 0
    assert len(chunks_with_rerank) > 0

    # The order or scores should be different between reranked and non-reranked results
    # This test might be a bit flaky depending on the exact scoring, but it should work most of the time
    # given our carefully crafted test data
    scores_no_rerank = [c["score"] for c in chunks_no_rerank]
    scores_with_rerank = [c["score"] for c in chunks_with_rerank]
    assert scores_no_rerank != scores_with_rerank, "Reranking should affect the scores"


@pytest.mark.asyncio
async def test_retrieve_docs_with_reranking(client: AsyncClient):
    """Test document retrieval with reranking options"""
    # First ingest documents with clearly different semantic relevance
    await test_ingest_text_document(
        client, "The capital of France is Paris. The city is known for the Eiffel Tower."
    )
    await test_ingest_text_document(
        client, "Paris is a city in France. It has many famous landmarks and museums."
    )
    await test_ingest_text_document(
        client, "Paris Hilton is a celebrity and businesswoman. She has nothing to do with France."
    )

    headers = create_auth_header()

    # Test with default reranking (from config)
    response_default = await client.post(
        "/retrieve/docs",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
        },
        headers=headers,
    )
    assert response_default.status_code == 200
    docs_default = response_default.json()
    assert len(docs_default) > 0

    # Test with explicit reranking enabled
    response_rerank = await client.post(
        "/retrieve/docs",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": True,
        },
        headers=headers,
    )
    assert response_rerank.status_code == 200
    docs_rerank = response_rerank.json()
    assert len(docs_rerank) > 0

    # Test with reranking disabled
    response_no_rerank = await client.post(
        "/retrieve/docs",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": False,
        },
        headers=headers,
    )
    assert response_no_rerank.status_code == 200
    docs_no_rerank = response_no_rerank.json()
    assert len(docs_no_rerank) > 0

    # Verify that reranking affects the order
    scores_rerank = [doc["score"] for doc in docs_rerank]
    scores_no_rerank = [doc["score"] for doc in docs_no_rerank]
    assert scores_rerank != scores_no_rerank, "Reranking should affect document scores"


@pytest.mark.asyncio
async def test_query_with_reranking(client: AsyncClient):
    """Test query completion with reranking options"""
    # First ingest documents with clearly different semantic relevance
    await test_ingest_text_document(
        client, "The capital of France is Paris. The city is known for the Eiffel Tower."
    )
    await test_ingest_text_document(
        client, "Paris is a city in France. It has many famous landmarks and museums."
    )
    await test_ingest_text_document(
        client, "Paris Hilton is a celebrity and businesswoman. She has nothing to do with France."
    )

    headers = create_auth_header()

    # Test with default reranking (from config)
    response_default = await client.post(
        "/query",
        json={
            "query": "What is the capital of France?",
            "k": 3,
            "min_score": 0.0,
            "max_tokens": 50,
        },
        headers=headers,
    )
    assert response_default.status_code == 200
    completion_default = response_default.json()
    assert "completion" in completion_default

    # Test with explicit reranking enabled
    response_rerank = await client.post(
        "/query",
        json={
            "query": "What is the capital of France?",
            "k": 3,
            "min_score": 0.0,
            "max_tokens": 50,
            "use_reranking": True,
        },
        headers=headers,
    )
    assert response_rerank.status_code == 200
    completion_rerank = response_rerank.json()
    assert "completion" in completion_rerank

    # Test with reranking disabled
    response_no_rerank = await client.post(
        "/query",
        json={
            "query": "What is the capital of France?",
            "k": 3,
            "min_score": 0.0,
            "max_tokens": 50,
            "use_reranking": False,
        },
        headers=headers,
    )
    assert response_no_rerank.status_code == 200
    completion_no_rerank = response_no_rerank.json()
    assert "completion" in completion_no_rerank

    # The actual responses might be different due to different chunk ordering,
    # but all should mention Paris as the capital
    assert "Paris" in completion_default["completion"]
    assert "Paris" in completion_rerank["completion"]
    assert "Paris" in completion_no_rerank["completion"]
