import asyncio
import os
from pathlib import Path
from databridge import DataBridge


ADMIN_URI = os.getenv("DATABRIDGE_ADMIN_URI")  
DEV_URI = os.getenv("DATABRIDGE_DEV_URI")      
USER_URI = os.getenv("DATABRIDGE_USER_URI")    


async def get_access_token(entity_type: str, entity_id: str) -> str:
    """
    Get an access token from the DataBridge API.
    In production, this would typically be handled by your auth system.
    """
    async with DataBridge(ADMIN_URI) as admin_db:
        if entity_type == "developer":
            response = await admin_db._request(
                "POST",
                "auth/developer-token",
                {
                    "dev_id": entity_id,
                    "expiry_days": 30,
                    "permissions": ["read", "write"]
                }
            )
        else:
            response = await admin_db._request(
                "POST",
                "auth/user-token",
                {
                    "user_id": entity_id,
                    "expiry_days": 30,
                    "permissions": ["read"]
                }
            )
        return response["uri"]


async def example_text():
    """Example of handling text documents"""
    print("\n=== Text Document Example ===")
    
    # Get developer access
    dev_uri = await get_access_token("developer", "dev_123")
    
    async with DataBridge(dev_uri) as db:
        # Ingest a text document
        doc = await db.ingest_document(
            content="Machine learning is a type of artificial intelligence...",
            content_type="text/plain",
            metadata={
                "title": "ML Introduction",
                "category": "tech",
                "tags": ["ML", "AI"]
            }
        )
        print(f"✓ Document ingested: {doc.external_id}")
        
        # Query for chunks
        results = await db.query(
            query="What is machine learning?",
            return_type="chunks",
            k=2
        )
        
        print("\nChunk Results:")
        for r in results:
            print(f"Content: {r.content[:200]}...")
            print(f"Score: {r.score:.2f}\n")


async def example_pdf():
    """Example of handling PDF documents"""
    print("\n=== PDF Document Example ===")
    
    # Get user access
    user_uri = await get_access_token("user", "user_789")
    
    pdf_path = Path(__file__).parent / "sample.pdf"
    if not pdf_path.exists():
        print("× sample.pdf not found")
        return
        
    async with DataBridge(user_uri) as db:
        # Read and ingest PDF
        with open(pdf_path, "rb") as f:
            doc = await db.ingest_document(
                content=f.read(),
                content_type="application/pdf",
                filename="sample.pdf",
                metadata={
                    "source": "examples",
                    "department": "research"
                }
            )
        print(f"✓ PDF ingested: {doc.external_id}")
        
        # Query for full documents
        results = await db.query(
            query="Key findings",
            return_type="documents",
            filters={"department": "research"}
        )
        
        print("\nDocument Results:")
        for r in results:
            print(f"Document ID: {r.document_id}")
            print(f"Score: {r.score:.2f}")
            if r.content["type"] == "url":
                print(f"Download URL: {r.content['value']}\n")


async def example_document_management():
    """Example of document management operations"""
    print("\n=== Document Management Example ===")
    
    # Using an existing developer URI with appropriate permissions
    async with DataBridge(DEV_URI) as db:
        # List documents with pagination
        documents = await db.list_documents(limit=5)
        print(f"Found {len(documents)} documents:")
        for doc in documents:
            print(f"- {doc.filename or doc.external_id} ({doc.content_type})")
        
        if documents:
            # Get specific document details
            doc = await db.get_document(documents[0].external_id)
            print(f"\nDocument details for {doc.external_id}:")
            print(f"Content type: {doc.content_type}")
            print(f"Created: {doc.system_metadata.get('created_at')}")
            print(f"Owner: {doc.access_control.get('owner', {}).get('id')}")


async def example_app_integration():
    """Example of integrating DataBridge in an application"""
    print("\n=== Application Integration Example ===")
    
    # Get app-specific developer access
    app_uri = await get_access_token(
        "developer", 
        "dev_123",
        app_id="app_456"  # Optional: specific app context
    )
    
    async with DataBridge(app_uri) as db:
        # Application-specific document management
        await db.ingest_document(
            content="App-specific content...",
            content_type="text/plain",
            metadata={
                "app_id": "app_456",
                "type": "user_data"
            }
        )
        
        # Query with app-specific filters
        results = await db.query(
            query="Find relevant data",
            filters={"app_id": "app_456"}
        )
        
        print(f"Found {len(results)} relevant documents")


async def main():
    """Run example scenarios"""
    try:
        await example_text()
        await example_pdf()
        await example_document_management()
        await example_app_integration()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
