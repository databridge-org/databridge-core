from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Base retrieve request model"""

    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.0)
    use_reranking: Optional[bool] = None  # If None, use default from config


class CompletionQueryRequest(RetrieveRequest):
    """Request model for completion generation"""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
