import asyncio
import jwt
from datetime import datetime, UTC, timedelta
from databridge import DataBridge
from pymongo import MongoClient
import os
from dotenv import load_dotenv

async def debug_databridge():
    # 1. Setup and Authentication
    load_dotenv()
    
    def create_test_token():
        return jwt.encode(
            {
                'type': 'developer',
                'exp': datetime.now(UTC) + timedelta(days=30)
            },
            "your-secret-key-for-signing-tokens",
            algorithm='HS256'
        )
    
    uri = f"databridge://673b64dcb6e40d739a9b6e2a:{create_test_token()}@localhost:8000"
    db = DataBridge(uri)
    
    # 2. Test Document Ingestion
    test_content = """
    This is a test document about machine learning.
    Machine learning is a subset of artificial intelligence.
    This text should be embedded and retrievable.
    """
    
    try:
        # Ingest document
        print("\n=== Testing Document Ingestion ===")
        doc_id = await db.ingest_document(
            content=test_content,
            metadata={
                "title": "Test Document",
                "test_type": "debug"
            }
        )
        print(f"✓ Document ingested with ID: {doc_id}")
        
        # 3. Verify MongoDB Storage
        print("\n=== Checking MongoDB Storage ===")
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            print("× ERROR: MONGODB_URI not set in environment")
            return
            
        client = MongoClient(mongodb_uri)
        collection = client.get_database(os.getenv("DATABRIDGE_DB")).get_collection("kb_chunked_embeddings")
        
        # Check document chunks
        chunks = list(collection.find({"system_metadata.doc_id": doc_id}))
        print(f"Found {len(chunks)} chunks in MongoDB")
        if chunks:
            print("Sample chunk data:")
            chunk = chunks[0]
            print(f"- Content length: {len(chunk.get('content', ''))}")
            print(f"- Has embedding: {'embedding' in chunk}")
            print(f"- Metadata present: {chunk.get('metadata') is not None}")
        else:
            print("× ERROR: No chunks found in MongoDB")
        
        # 4. Test Query
        print("\n=== Testing Query ===")
        results = await db.query(
            query="What is machine learning?",
            k=1
        )
        
        print(f"Query returned {len(results)} results")
        if results:
            print("\nFirst result:")
            print(f"Content: {results[0].content}")
            print(f"Score: {results[0].score}")
            print(f"Doc ID: {results[0].doc_id}")
        else:
            print("× ERROR: No results returned from query")
        
        # 5. Check Vector Search Configuration
        print("\n=== Checking Vector Search Configuration ===")
        
        # Check indexes
        indexes = collection.list_indexes()
        print("Available indexes:")
        for idx in indexes:
            print(f"- {idx['name']}: {idx.get('weights', {}).keys()}")
            
        # Try basic find operation
        print("\n=== Checking Basic Query Access ===")
        basic_query = {"system_metadata.doc_id": doc_id}
        doc = collection.find_one(basic_query)
        if doc:
            print("✓ Can read document directly")
            print("Document permissions:", doc.get("permissions", {}))
            print("Document system metadata:", doc.get("system_metadata", {}))
        else:
            print("× Cannot read document directly")
            
        # Try vector search with explicit auth filter
        print("\n=== Testing Vector Search with Auth Filter ===")
        auth_filter = {
            "$or": [
                {"system_metadata.dev_id": "673b64dcb6e40d739a9b6e2a"},
                {"permissions": {"$in": ["app123"]}}  # Replace with your actual app_id
            ]
        }
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": [0.1] * 1536,
                    "numCandidates": 10,
                    "limit": 1,
                    "filter": auth_filter
                }
            }
        ]
        
        try:
            vector_results = list(collection.aggregate(pipeline))
            print(f"Vector search returned {len(vector_results)} results")
        except Exception as e:
            print(f"× ERROR in vector search: {str(e)}")
        
    except Exception as e:
        print(f"× Error during testing: {str(e)}")
    finally:
        await db.close()
        client.close()

if __name__ == "__main__":
    asyncio.run(debug_databridge())
