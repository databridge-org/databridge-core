import asyncio
import os
from pathlib import Path
import sys
import pytest
from typing import Generator
from datetime import datetime, timedelta, UTC
import jwt

sdk_path = str(Path(__file__).parent.parent.parent / "sdks" / "python")
sys.path.append(sdk_path)

from databridge import DataBridge
# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

def create_test_token(entity_type: str = "developer", entity_id: str = "test_dev", 
                     app_id: str = None) -> str:
    """Create a test JWT token"""
    secret = os.getenv("JWT_SECRET_KEY", "test-secret")
    payload = {
        "type": entity_type,
        "entity_id": entity_id,
        "permissions": ["read", "write", "admin"],
        "exp": datetime.now(UTC) + timedelta(days=1)
    }
    if app_id:
        payload["app_id"] = app_id
    
    return jwt.encode(payload, secret, algorithm="HS256")

def get_test_uri(token: str = None) -> str:
    """Get test DataBridge URI"""
    if not token:
        token = create_test_token()
    host = os.getenv("DATABRIDGE_HOST", "localhost:8000")
    return f"databridge://test_dev:{token}@{host}"

@pytest.fixture
async def db() -> Generator[DataBridge, None, None]:
    """DataBridge client fixture"""
    client = DataBridge(get_test_uri())
    try:
        yield client
    finally:
        await client.close()

# Test Files
@pytest.fixture(scope="session")
def setup_test_files():
    """Create test files if they don't exist"""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Text file
    text_file = TEST_DATA_DIR / "test.txt"
    if not text_file.exists():
        text_file.write_text("This is a test document for DataBridge testing.")
    
    # PDF file (create a simple one)
    pdf_file = TEST_DATA_DIR / "test.pdf"
    if not pdf_file.exists():
        raise NotImplementedError("Test PDF not present")

async def test_text_ingestion(db: DataBridge, setup_test_files):
    """Test ingesting a text document"""
    content = (TEST_DATA_DIR / "test.txt").read_text()
    
    # Test ingest
    doc = await db.ingest_document(
        content=content,
        content_type="text/plain",
        metadata={"test": True, "type": "text"}
    )
    
    assert doc.external_id
    assert doc.content_type == "text/plain"
    assert doc.metadata["test"] is True
    
    # Verify storage
    stored_doc = await db.get_document(doc.external_id)
    assert stored_doc.external_id == doc.external_id
    
    return doc.external_id

async def test_binary_ingestion(db: DataBridge, setup_test_files):
    """Test ingesting a binary document (PDF)"""
    pdf_path = TEST_DATA_DIR / "test.pdf"
    if not pdf_path.exists():
        pytest.skip("Test PDF not available")
    
    with open(pdf_path, "rb") as f:
        content = f.read()
    
    doc = await db.ingest_document(
        content=content,
        content_type="application/pdf",
        filename="test.pdf",
        metadata={"test": True, "type": "pdf"}
    )
    
    assert doc.external_id
    assert doc.content_type == "application/pdf"
    assert doc.filename == "test.pdf"
    assert doc.storage_info  # Should have S3 info
    
    return doc.external_id

async def test_query_chunks(db: DataBridge):
    """Test querying for chunks"""
    results = await db.query(
        query="test document",
        return_type="chunks",
        k=2
    )
    
    assert len(results) > 0
    for r in results:
        assert r.content
        assert r.score > 0
        assert r.document_id
        assert r.metadata

async def test_query_documents(db: DataBridge):
    """Test querying for documents"""
    results = await db.query(
        query="test document",
        return_type="documents",
        filters={"test": True}
    )
    
    assert len(results) > 0
    for r in results:
        assert r.document_id
        assert r.score > 0
        assert r.metadata
        assert r.content["type"] in ["url", "string"]

async def test_list_and_get(db: DataBridge):
    """Test document listing and retrieval"""
    # List documents
    docs = await db.list_documents(limit=5)
    assert len(docs) > 0
    
    # Get specific document
    doc = await db.get_document(docs[0].external_id)
    assert doc.external_id == docs[0].external_id
    assert doc.metadata
    assert doc.content_type

async def test_auth_and_permissions():
    """Test authentication and permissions"""
    # Test invalid token
    with pytest.raises(ValueError):
        DataBridge("databridge://test:invalid_token@localhost:8000")
    
    # Test expired token
    expired_token = jwt.encode(
        {
            "type": "developer",
            "entity_id": "test_dev",
            "exp": datetime.now(UTC) - timedelta(days=1)
        },
        "test-secret"
    )
    
    db = DataBridge(get_test_uri(expired_token))
    with pytest.raises(Exception):  # Should fail auth
        await db.list_documents()

async def test_error_handling(db: DataBridge):
    """Test error handling scenarios"""
    # Test invalid document ID
    with pytest.raises(Exception):
        await db.get_document("invalid_id")
    
    # Test invalid query
    with pytest.raises(Exception):
        await db.query("", k=-1)  # Invalid k
    
    # Test large content (if size limits are implemented)
    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    with pytest.raises(Exception):
        await db.ingest_document(
            content=large_content,
            content_type="text/plain"
        )

# Run test suite
async def main():
    """Run all tests"""
    db = DataBridge(get_test_uri())
    try:
        setup_test_files()
        
        # Run ingestion tests
        doc_id = await test_text_ingestion(db)
        pdf_id = await test_binary_ingestion(db)
        print(f"✓ Ingestion tests passed (doc_id: {doc_id}, pdf_id: {pdf_id})")
        
        # Run query tests
        await test_query_chunks(db)
        await test_query_documents(db)
        print("✓ Query tests passed")
        
        # Run document management tests
        await test_list_and_get(db)
        print("✓ Document management tests passed")
        
        # Run auth tests
        await test_auth_and_permissions()
        print("✓ Auth tests passed")
        
        # Run error handling tests
        await test_error_handling(db)
        print("✓ Error handling tests passed")
        
    except Exception as e:
        print(f"× Test failed: {str(e)}")
        raise
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())
