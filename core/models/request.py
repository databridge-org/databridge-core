from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .documents import QueryReturnType


class IngestTextRequest(BaseModel):
    """Request model for text ingestion"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Query request model - remains unchanged"""
    query: str
    return_type: QueryReturnType = QueryReturnType.CHUNKS
    filters: Optional[Dict[str, Any]] = None
    k: int = 4
    min_score: float = 0.0
