import json
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC, timedelta
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Index, select, text
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)
Base = declarative_base()


class UserLimitsModel(Base):
    """SQLAlchemy model for user limits data."""

    __tablename__ = "user_limits"

    user_id = Column(String, primary_key=True)
    tier = Column(String, nullable=False)  # FREE, PRO, CUSTOM, SELF_HOSTED
    custom_limits = Column(JSONB, nullable=True)
    usage = Column(JSONB, default=dict)  # Holds all usage counters
    app_ids = Column(JSONB, default=list)  # List of app IDs registered by this user
    created_at = Column(String)  # ISO format string
    updated_at = Column(String)  # ISO format string

    # Create indexes
    __table_args__ = (Index("idx_user_tier", "tier"),)


class UserLimitsDatabase:
    """Database operations for user limits."""

    def __init__(self, uri: str):
        """Initialize database connection."""
        self.engine = create_async_engine(uri)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize database tables and indexes."""
        if self._initialized:
            return True

        try:
            logger.info("Initializing user limits database tables...")
            # Create tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info("User limits database tables initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize user limits database: {e}")
            return False

    async def get_user_limits(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user limits for a user.

        Args:
            user_id: The user ID to get limits for

        Returns:
            Dict with user limits if found, None otherwise
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(UserLimitsModel).where(UserLimitsModel.user_id == user_id)
            )
            user_limits = result.scalars().first()

            if not user_limits:
                return None

            return {
                "user_id": user_limits.user_id,
                "tier": user_limits.tier,
                "custom_limits": user_limits.custom_limits,
                "usage": user_limits.usage,
                "app_ids": user_limits.app_ids,
                "created_at": user_limits.created_at,
                "updated_at": user_limits.updated_at,
            }

    async def create_user_limits(self, user_id: str, tier: str = "free") -> bool:
        """
        Create user limits record.

        Args:
            user_id: The user ID
            tier: Initial tier (defaults to "free")

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                # Check if already exists
                result = await session.execute(
                    select(UserLimitsModel).where(UserLimitsModel.user_id == user_id)
                )
                if result.scalars().first():
                    return True  # Already exists

                # Create new record with properly initialized JSONB columns
                # Create JSON strings and parse them for consistency
                usage_json = json.dumps({
                    "storage_file_count": 0,
                    "storage_size_bytes": 0,
                    "hourly_query_count": 0,
                    "hourly_query_reset": now,
                    "monthly_query_count": 0,
                    "monthly_query_reset": now,
                    "hourly_ingest_count": 0,
                    "hourly_ingest_reset": now,
                    "monthly_ingest_count": 0,
                    "monthly_ingest_reset": now,
                    "graph_count": 0,
                    "cache_count": 0,
                })
                app_ids_json = json.dumps([])  # Empty array but as JSON string
                
                # Create the model with the JSON parsed
                user_limits = UserLimitsModel(
                    user_id=user_id,
                    tier=tier,
                    usage=json.loads(usage_json),
                    app_ids=json.loads(app_ids_json),
                    created_at=now,
                    updated_at=now,
                )

                session.add(user_limits)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to create user limits: {e}")
            return False

    async def update_user_tier(
        self, user_id: str, tier: str, custom_limits: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update user tier and custom limits.

        Args:
            user_id: The user ID
            tier: New tier
            custom_limits: Optional custom limits for CUSTOM tier

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                result = await session.execute(
                    select(UserLimitsModel).where(UserLimitsModel.user_id == user_id)
                )
                user_limits = result.scalars().first()

                if not user_limits:
                    return False

                user_limits.tier = tier
                user_limits.custom_limits = custom_limits
                user_limits.updated_at = now

                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update user tier: {e}")
            return False

    async def register_app(self, user_id: str, app_id: str) -> bool:
        """
        Register an app for a user.

        Args:
            user_id: The user ID
            app_id: The app ID to register

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                # First check if user exists
                result = await session.execute(
                    select(UserLimitsModel).where(UserLimitsModel.user_id == user_id)
                )
                user_limits = result.scalars().first()

                if not user_limits:
                    logger.error(f"User {user_id} not found in register_app")
                    return False
                
                # Use raw SQL with jsonb_array_append to update the app_ids array
                # This is the most reliable way to append to a JSONB array in PostgreSQL
                query = text(
                    """
                    UPDATE user_limits 
                    SET 
                        app_ids = CASE 
                            WHEN NOT (app_ids ? :app_id)  -- Check if app_id is not in the array
                            THEN app_ids || :app_id_json  -- Append it if not present
                            ELSE app_ids                  -- Keep it unchanged if already present
                        END,
                        updated_at = :now
                    WHERE user_id = :user_id
                    RETURNING app_ids;
                    """
                )
                
                # Execute the query
                result = await session.execute(
                    query, 
                    {
                        "app_id": app_id,                       # For the check
                        "app_id_json": f'["{app_id}"]',         # JSON array format for appending
                        "now": now,
                        "user_id": user_id
                    }
                )
                
                # Log the result for debugging
                updated_app_ids = result.scalar()
                logger.info(f"Updated app_ids for user {user_id}: {updated_app_ids}")
                
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to register app: {e}")
            return False

    async def update_usage(self, user_id: str, usage_type: str, increment: int = 1) -> bool:
        """
        Update usage counter for a user.

        Args:
            user_id: The user ID
            usage_type: Type of usage to update
            increment: Value to increment by

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC)
            now_iso = now.isoformat()
            
            async with self.async_session() as session:
                result = await session.execute(
                    select(UserLimitsModel).where(UserLimitsModel.user_id == user_id)
                )
                user_limits = result.scalars().first()
                
                if not user_limits:
                    return False
                
                # Create a new dictionary to force SQLAlchemy to detect the change
                usage = dict(user_limits.usage) if user_limits.usage else {}
                
                # Handle different usage types
                if usage_type == "query":
                    # Check hourly reset
                    hourly_reset_str = usage.get("hourly_query_reset", "")
                    if hourly_reset_str:
                        hourly_reset = datetime.fromisoformat(hourly_reset_str)
                        if now > hourly_reset + timedelta(hours=1):
                            usage["hourly_query_count"] = increment
                            usage["hourly_query_reset"] = now_iso
                        else:
                            usage["hourly_query_count"] = usage.get("hourly_query_count", 0) + increment
                    else:
                        usage["hourly_query_count"] = increment
                        usage["hourly_query_reset"] = now_iso

                    # Check monthly reset
                    monthly_reset_str = usage.get("monthly_query_reset", "")
                    if monthly_reset_str:
                        monthly_reset = datetime.fromisoformat(monthly_reset_str)
                        if now > monthly_reset + timedelta(days=30):
                            usage["monthly_query_count"] = increment
                            usage["monthly_query_reset"] = now_iso
                        else:
                            usage["monthly_query_count"] = usage.get("monthly_query_count", 0) + increment
                    else:
                        usage["monthly_query_count"] = increment
                        usage["monthly_query_reset"] = now_iso

                elif usage_type == "ingest":
                    # Similar pattern for ingest
                    hourly_reset_str = usage.get("hourly_ingest_reset", "")
                    if hourly_reset_str:
                        hourly_reset = datetime.fromisoformat(hourly_reset_str)
                        if now > hourly_reset + timedelta(hours=1):
                            usage["hourly_ingest_count"] = increment
                            usage["hourly_ingest_reset"] = now_iso
                        else:
                            usage["hourly_ingest_count"] = usage.get("hourly_ingest_count", 0) + increment
                    else:
                        usage["hourly_ingest_count"] = increment
                        usage["hourly_ingest_reset"] = now_iso

                    monthly_reset_str = usage.get("monthly_ingest_reset", "")
                    if monthly_reset_str:
                        monthly_reset = datetime.fromisoformat(monthly_reset_str)
                        if now > monthly_reset + timedelta(days=30):
                            usage["monthly_ingest_count"] = increment
                            usage["monthly_ingest_reset"] = now_iso
                        else:
                            usage["monthly_ingest_count"] = usage.get("monthly_ingest_count", 0) + increment
                    else:
                        usage["monthly_ingest_count"] = increment
                        usage["monthly_ingest_reset"] = now_iso

                elif usage_type == "storage_file":
                    usage["storage_file_count"] = usage.get("storage_file_count", 0) + increment

                elif usage_type == "storage_size":
                    usage["storage_size_bytes"] = usage.get("storage_size_bytes", 0) + increment

                elif usage_type == "graph":
                    usage["graph_count"] = usage.get("graph_count", 0) + increment

                elif usage_type == "cache":
                    usage["cache_count"] = usage.get("cache_count", 0) + increment
                
                # Force SQLAlchemy to recognize the change by assigning a new dict
                user_limits.usage = usage
                user_limits.updated_at = now_iso
                
                # Explicitly mark as modified
                session.add(user_limits)

                # Log the updated usage for debugging
                logger.info(f"Updated usage for user {user_id}, type: {usage_type}, value: {increment}")
                logger.info(f"New usage values: {usage}")
                logger.info(f"About to commit: user_id={user_id}, usage={user_limits.usage}")

                # Commit and flush to ensure changes are written
                await session.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
