from typing import Dict, Any, Optional
from datetime import datetime, UTC
from pydantic import BaseModel, Field

from .tiers import AccountTier


class UserUsage(BaseModel):
    """Tracks user's actual usage of the system."""

    # Storage usage
    storage_file_count: int = 0
    storage_size_bytes: int = 0

    # Query usage - hourly
    hourly_query_count: int = 0
    hourly_query_reset: Optional[datetime] = None

    # Query usage - monthly
    monthly_query_count: int = 0
    monthly_query_reset: Optional[datetime] = None

    # Ingest usage - hourly
    hourly_ingest_count: int = 0
    hourly_ingest_reset: Optional[datetime] = None

    # Ingest usage - monthly
    monthly_ingest_count: int = 0
    monthly_ingest_reset: Optional[datetime] = None

    # Graph usage
    graph_count: int = 0
    hourly_graph_query_count: int = 0
    hourly_graph_query_reset: Optional[datetime] = None
    monthly_graph_query_count: int = 0
    monthly_graph_query_reset: Optional[datetime] = None

    # Cache usage
    cache_count: int = 0
    hourly_cache_query_count: int = 0
    hourly_cache_query_reset: Optional[datetime] = None
    monthly_cache_query_count: int = 0
    monthly_cache_query_reset: Optional[datetime] = None


class UserLimits(BaseModel):
    """Stores user tier and usage information."""

    user_id: str
    tier: AccountTier = AccountTier.FREE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    usage: UserUsage = Field(default_factory=UserUsage)
    custom_limits: Optional[Dict[str, Any]] = None
    app_ids: list[str] = Field(default_factory=list)
