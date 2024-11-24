from typing import List, Dict, Any, Optional
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from .base_vector_store import BaseVectorStore
from core.models.documents import DocumentChunk
from core.models.auth import AuthContext, EntityType

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

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks with their embeddings."""
        try:
            if not chunks:
                return True

            # Convert chunks to dicts
            documents = []
            for chunk in chunks:
                doc = chunk.model_dump()
                # Ensure we have required fields
                if not doc.get('embedding'):
                    logger.error(f"Missing embedding for chunk {chunk.document_id}-{chunk.chunk_number}")
                    continue
                documents.append(doc)

            if documents:
                # Use ordered=False to continue even if some inserts fail
                result = await self.collection.insert_many(documents, ordered=False)
                return len(result.inserted_ids) > 0
            return False

        except PyMongoError as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Find similar chunks using MongoDB Atlas Vector Search."""
        try:
            # Build access filter based on auth context
            access_filter = self._build_access_filter(auth)
            
            # Add metadata filters if provided
            filter_query = access_filter
            if filters:
                metadata_filter = self._build_metadata_filter(filters)
                if metadata_filter:
                    filter_query = {"$and": [access_filter, metadata_filter]}

            # Vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,  # Get more candidates for better results
                        "limit": k,
                        "filter": filter_query
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
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []

    def _build_access_filter(self, auth: AuthContext) -> Dict[str, Any]:
        """Build MongoDB filter for access control."""
        base_filter = {
            "$or": [
                {"owner.id": auth.entity_id},
                {"access_control.readers": auth.entity_id},
                {"access_control.writers": auth.entity_id},
                {"access_control.admins": auth.entity_id}
            ]
        }

        if auth.entity_type == EntityType.DEVELOPER and auth.app_id:
            # Add app-specific access for developers
            base_filter["$or"].append(
                {"access_control.app_access": auth.app_id}
            )
            
        return base_filter

    def _build_metadata_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build MongoDB filter for metadata fields."""
        if not filters:
            return {}
            
        metadata_filter = {}
        
        for key, value in filters.items():
            metadata_key = f"metadata.{key}"
            
            if isinstance(value, (str, int, float, bool)):
                metadata_filter[metadata_key] = value
            elif isinstance(value, list):
                metadata_filter[metadata_key] = {"$in": value}
            elif isinstance(value, dict):
                valid_ops = {
                    "gt": "$gt",
                    "gte": "$gte", 
                    "lt": "$lt",
                    "lte": "$lte",
                    "ne": "$ne"
                }
                mongo_ops = {}
                for op, val in value.items():
                    if op not in valid_ops:
                        logger.warning(f"Skipping invalid operator: {op}")
                        continue
                    mongo_ops[valid_ops[op]] = val
                if mongo_ops:
                    metadata_filter[metadata_key] = mongo_ops
            else:
                logger.warning(f"Skipping unsupported filter value type for key {key}")
                
        return metadata_filter
