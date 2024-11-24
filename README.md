# DataBridge

DataBridge is an extensible, open-source document processing and retrieval system designed for building document-based applications. It provides a modular architecture for integrating document parsing, embedding generation, and vector search capabilities.

## Features

- 🔌 **Extensible Architecture**: Built with modularity in mind - easily extend or replace any component:
  - Document Parsing: Currently integrated with Unstructured API
  - Vector Store: Currently using MongoDB Atlas
  - Embedding Model: Currently using OpenAI
  - Storage: Currently using AWS S3
- 🔍 **Vector Search**: Semantic search capabilities
- 🔐 **Authentication**: JWT-based auth with developer and end-user access modes
- 📊 **Metadata**: Rich metadata filtering and organization
- 🚀 **Python SDK**: Simple client SDK for quick integration

## Quick Start

1. Install the SDK:
```bash
pip install databridge-client
```

2. Set up your environment variables:
```env
MONGODB_URI=your_mongodb_connection_string
OPENAI_API_KEY=your_openai_api_key
UNSTRUCTURED_API_KEY=your_unstructured_api_key
JWT_SECRET_KEY=your_jwt_secret
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

3. Start the server:
```bash
python start_server.py
```

4. Use the SDK:
```python
import asyncio
from databridge import DataBridge

async def main():
    # Initialize client
    db = DataBridge("databridge://owner_id:auth_token@your-domain.com")
    
    # Ingest a document
    doc_id = await db.ingest_document(
        content="Your document content",
        metadata={"title": "My Document"}
    )
    
    # Query documents
    results = await db.query(
        query="What is...",
        k=4  # Number of results
    )
    
    await db.close()

asyncio.run(main())
```

## Architecture

DataBridge uses a modular architecture with the following base components that can be extended or replaced:

### Current Integrations

- **Document Parser**: Unstructured API integration for intelligent document processing
  - Extend `BaseParser` to add new parsing capabilities
- **Vector Store**: MongoDB Atlas Vector Search integration
  - Extend `BaseVectorStore` to add new vector stores
- **Embedding Model**: OpenAI embeddings integration
  - Extend `BaseEmbeddingModel` to add new embedding models
- **Storage**: AWS S3 integration
  - Storage utilities can be modified in `utils/`

### Adding New Components

1. Implement the relevant base class from `core/`
2. Register your implementation in the service configuration
3. Update environment variables if needed

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the complete OpenAPI documentation.

### Key Endpoints

- `POST /ingest`: Ingest new documents
- `POST /query`: Query documents using semantic search
- `GET /documents`: List all documents
- `GET /document/{doc_id}`: Get specific document details

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please open an issue or submit a pull request.

---

Built with ❤️ by DataBridge.