from typing import List, Optional, Dict, Any
import logging
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from .base_database import BaseDatabase
from ..models.documents import Document
from ..models.auth import AuthContext, EntityType

logger = logging.getLogger(__name__)


class MongoDatabase(BaseDatabase):
    """MongoDB implementation for document metadata storage."""

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str,
    ):
        """Initialize MongoDB connection for document storage."""
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    async def initialize(self):
        """Initialize database indexes."""
        try:
            # Create indexes for common queries
            await self.collection.create_index("external_id", unique=True)
            await self.collection.create_index("owner.id")
            await self.collection.create_index("access_control.readers")
            await self.collection.create_index("access_control.writers")
            await self.collection.create_index("access_control.admins")
            await self.collection.create_index("system_metadata.created_at")
            
            logger.info("MongoDB indexes created successfully")
            return True
        except PyMongoError as e:
            logger.error(f"Error creating MongoDB indexes: {str(e)}")
            return False

    async def store_document(self, document: Document) -> bool:
        """Store document metadata."""
        try:
            doc_dict = document.model_dump()
            
            # Ensure system metadata
            doc_dict["system_metadata"]["created_at"] = datetime.now(UTC)
            doc_dict["system_metadata"]["updated_at"] = datetime.now(UTC)

            result = await self.collection.insert_one(doc_dict)
            return bool(result.inserted_id)
            
        except PyMongoError as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            return False

    async def get_document(self, document_id: str, auth: AuthContext) -> Optional[Document]:
        """Retrieve document metadata by ID if user has access."""
        try:
            # Build access filter
            access_filter = self._build_access_filter(auth)
            
            # Query document
            query = {
                "$and": [
                    {"external_id": document_id},
                    access_filter
                ]
            }

            doc_dict = await self.collection.find_one(query)
            return Document(**doc_dict) if doc_dict else None
            
        except PyMongoError as e:
            logger.error(f"Error retrieving document metadata: {str(e)}")
            return None

    async def get_documents(
        self,
        auth: AuthContext,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """List accessible documents with pagination and filtering."""
        try:
            # Build query
            query = self._build_access_filter(auth)
            if filters:
                query = {"$and": [query, filters]}

            # Execute paginated query
            cursor = self.collection.find(query).skip(skip).limit(limit)
            
            documents = []
            async for doc_dict in cursor:
                documents.append(Document(**doc_dict))
            
            return documents
            
        except PyMongoError as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    async def update_document(
        self,
        document_id: str,
        updates: Dict[str, Any],
        auth: AuthContext
    ) -> bool:
        """Update document metadata if user has write access."""
        try:
            # Verify write access
            if not await self.check_access(document_id, auth, "write"):
                return False

            # Update system metadata
            updates.setdefault("system_metadata", {})
            updates["system_metadata"]["updated_at"] = datetime.utcnow()

            result = await self.collection.find_one_and_update(
                {"external_id": document_id},
                {"$set": updates},
                return_document=ReturnDocument.AFTER
            )
            
            return bool(result)
            
        except PyMongoError as e:
            logger.error(f"Error updating document metadata: {str(e)}")
            return False

    async def delete_document(self, document_id: str, auth: AuthContext) -> bool:
        """Delete document if user has admin access."""
        try:
            # Verify admin access
            if not await self.check_access(document_id, auth, "admin"):
                return False

            result = await self.collection.delete_one({"external_id": document_id})
            return bool(result.deleted_count)
            
        except PyMongoError as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    async def find_documents(
        self,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Find document IDs matching filters and access permissions."""
        try:
            # Build query
            query = self._build_access_filter(auth)
            if filters:
                query = {"$and": [query, filters]}

            # Get matching document IDs
            cursor = self.collection.find(query, {"external_id": 1})
            
            document_ids = []
            async for doc in cursor:
                document_ids.append(doc["external_id"])
                
            return document_ids
            
        except PyMongoError as e:
            logger.error(f"Error finding documents: {str(e)}")
            return []

    async def check_access(
        self,
        document_id: str,
        auth: AuthContext,
        required_permission: str = "read"
    ) -> bool:
        """Check if user has required permission for document."""
        try:
            doc = await self.collection.find_one({"external_id": document_id})
            if not doc:
                return False

            access_control = doc.get("access_control", {})
            
            # Check owner access
            owner = doc.get("owner", {})
            if (owner.get("type") == auth.entity_type and owner.get("id") == auth.entity_id):
                return True

            # Check permission-specific access
            permission_map = {
                "read": "readers",
                "write": "writers",
                "admin": "admins"
            }
            
            permission_set = permission_map.get(required_permission)
            if not permission_set:
                return False
                
            return auth.entity_id in access_control.get(permission_set, set())
            
        except PyMongoError as e:
            logger.error(f"Error checking document access: {str(e)}")
            return False

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

        if auth.entity_type == EntityType.DEVELOPER:
            # Add app-specific access for developers
            base_filter["$or"].append(
                {"access_control.app_access": auth.app_id}
            )

        return base_filter
