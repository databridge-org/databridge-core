from typing import List, Optional, Tuple
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from .base_vector_store import BaseVectorStore
from core.models.documents import DocumentChunk

logger = logging.getLogger(__name__)


class MongoDBAtlasVectorStore(BaseVectorStore):
    """MongoDB Atlas Vector Search implementation."""

    def __init__(
        self,
        uri: str,
        database_name: str,
        collection_name: str = "document_chunks",
        index_name: str = "vector_index"
    ):
        """Initialize MongoDB connection for vector storage."""
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.index_name = index_name

    async def initialize(self):
        """Initialize vector search index if needed."""
        try:
            # Create basic indexes
            await self.collection.create_index("document_id")
            await self.collection.create_index("chunk_number")

            # Note: Vector search index must be created via Atlas UI or API
            # as it requires specific configuration

            logger.info("MongoDB vector store indexes initialized")
            return True
        except PyMongoError as e:
            logger.error(f"Error initializing vector store indexes: {str(e)}")
            return False

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their embeddings."""
        try:
            if not chunks:
                return True, []

            # Convert chunks to dicts
            documents = []
            for chunk in chunks:
                doc = chunk.model_dump()
                # Ensure we have required fields
                if not doc.get('embedding'):
                    logger.error(
                        f"Missing embedding for chunk "
                        f"{chunk.document_id}-{chunk.chunk_number}"
                    )
                    continue
                documents.append(doc)

            if documents:
                # Use ordered=False to continue even if some inserts fail
                result = await self.collection.insert_many(
                    documents, ordered=False
                )
                return len(result.inserted_ids) > 0, [str(id) for id in result.inserted_ids]
            return False, []

        except PyMongoError as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False, []

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using MongoDB Atlas Vector Search."""
        try:
            logger.debug(
                f"Searching in database {self.db.name} "
                f"collection {self.collection.name}"
            )
            logger.debug(f"Query vector looks like: {query_embedding}")
            logger.debug(f"Doc IDs: {doc_ids}")
            logger.debug(f"K is: {k}")
            logger.debug(f"Index is: {self.index_name}")

            # Vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k*40,  # Get more candidates
                        "limit": k,
                        "filter": {"document_id": {"$in": doc_ids}} if doc_ids else {}
                    }
                },
                {
                    "$project": {
                        "score": {"$meta": "vectorSearchScore"},
                        "document_id": 1,
                        "chunk_number": 1,
                        "content": 1,
                        "metadata": 1,
                        "_id": 0
                    }
                }
            ]

            # Execute search
            cursor = self.collection.aggregate(pipeline)
            chunks = []

            async for result in cursor:
                chunk = DocumentChunk(
                    document_id=result["document_id"],
                    chunk_number=result["chunk_number"],
                    content=result["content"],
                    embedding=[],  # Don't send embeddings back
                    metadata=result.get("metadata", {}),
                    score=result.get("score", 0.0)
                )
                chunks.append(chunk)

            return chunks

        except PyMongoError as e:
            logger.error(f"MongoDB error: {e._message}")
            logger.error(f"Error querying similar chunks: {str(e)}")
            raise e
