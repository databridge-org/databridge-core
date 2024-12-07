import asyncio
import os
# import sys
from pathlib import Path

# pip install -e ./sdks/python
from databridge import DataBridge
import httpx


class LocalDataBridge(DataBridge):
    def __init__(self, uri: str, timeout: int = 30):
        # Initialize base class first
        super().__init__(uri, timeout)
        
        # Then override the client
        if "127.0.0.1" in uri or "localhost" in uri:
            self._client = httpx.AsyncClient(
                timeout=timeout,
                verify=False,  # Disable SSL for localhost
                http2=False    # Force HTTP/1.1
            )
            # Ensure base URL uses http for localhost
            self._base_url = self._base_url.replace("https://", "http://")


# Can be generated using generate_local_uri.py
DATABRIDGE_URI = ""


async def demonstrate_text_ingestion(db: LocalDataBridge):
    """Demonstrate text document ingestion"""
    print("\n=== Text Ingestion Example ===")

    # Ingest a text document with metadata
    doc = await db.ingest_text(
        content=(
            "Machine learning is a branch of artificial intelligence that allows "
            "systems to learn and improve from experience. Deep learning, a subset "
            "of machine learning, uses neural networks with multiple layers to "
            "analyze various factors of data."
        ),
        metadata={
            "title": "Introduction to Machine Learning",
            "category": "technical",
            "tags": ["ML", "AI", "deep learning"],
            "difficulty": "intermediate"
        }
    )

    print("✓ Text document ingested")
    print(f"  ID: {doc.external_id}")
    print(f"  Content Type: {doc.content_type}")
    print(f"  Tags: {doc.metadata.get('tags')}")

    return doc.external_id


async def demonstrate_file_ingestion(db: LocalDataBridge):
    """Demonstrate file document ingestion"""
    print("\n=== File Ingestion Example ===")

    # Create a sample PDF file
    pdf_path = Path("sample.pdf")
    if not pdf_path.exists():
        print("× Sample PDF not found, skipping file ingestion")
        return

    # Method 1: Ingest using file path
    doc1 = await db.ingest_file(
        file=pdf_path,
        filename="research_paper.pdf",
        content_type="application/pdf",
        metadata={
            "title": "Research Findings",
            "department": "R&D",
            "year": 2024
        }
    )
    print("✓ File ingested from path")
    print(f"  ID: {doc1.external_id}")
    print(f"  Storage Info: {doc1.storage_info}")

    # Method 2: Ingest using file object
    with open(pdf_path, "rb") as f:
        doc2 = await db.ingest_file(
            file=f,
            filename="research_paper_v2.pdf",
            content_type="application/pdf"
        )
    print("✓ File ingested from file object")
    print(f"  ID: {doc2.external_id}")

    return doc1.external_id


async def demonstrate_querying(db: LocalDataBridge, doc_id: str):
    """Demonstrate document querying"""
    print("\n=== Querying Example ===")

    # Query 1: Search for chunks
    print("\nSearching for chunks about machine learning:")
    chunks = await db.query(
        query="What is machine learning?",
        return_type="chunks",
        k=2,
        filters={"category": "technical"}
    )

    for chunk in chunks:
        print(f"\nChunk from document {chunk.document_id}")
        print(f"Score: {chunk.score:.2f}")
        print(f"Content: {chunk.content[:200]}...")

    # Query 2: Search for documents
    print("\nSearching for documents about deep learning:")
    docs = await db.query(
        query="deep learning applications",
        return_type="documents",
        filters={"tags": ["ML", "AI"]}
    )

    for doc in docs:
        print(f"\nDocument: {doc.document_id}")
        print(f"Score: {doc.score:.2f}")
        print(f"Content Type: {doc.content['type']}")


async def demonstrate_document_management(db: LocalDataBridge):
    """Demonstrate document management operations"""
    print("\n=== Document Management Example ===")

    # List documents with pagination
    print("\nListing first 5 documents:")
    docs = await db.list_documents(limit=5)
    for doc in docs:
        print(f"- {doc.external_id}: {doc.metadata.get('title', 'Untitled')}")

    if docs:
        # Get specific document details
        doc_id = docs[0].external_id
        print(f"\nFetching details for document {doc_id}")
        doc = await db.get_document(doc_id)
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Content Type: {doc.content_type}")
        print(f"Created: {doc.system_metadata.get('created_at')}")
        if doc.storage_info:
            bucket = doc.storage_info.get('bucket')
            key = doc.storage_info.get('key')
            print(f"Storage Location: {bucket}/{key}")


async def main():
    """Run all examples"""
    if not DATABRIDGE_URI:
        print("Please set DATABRIDGE_URI environment variable")
        return
    
    try:
        async with LocalDataBridge(DATABRIDGE_URI) as db:
            # Demonstrate text ingestion
            text_doc_id = await demonstrate_text_ingestion(db)
            
            # Demonstrate file ingestion
            await demonstrate_file_ingestion(db)
            
            # Demonstrate querying
            await demonstrate_querying(db, text_doc_id)
            
            # Demonstrate document management
            await demonstrate_document_management(db)
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
