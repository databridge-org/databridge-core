from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Base retrieve request model"""

    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.0)
    use_reranking: Optional[bool] = None  # If None, use default from config
    use_colpali: Optional[bool] = None
    graph_name: Optional[str] = Field(
        None, description="Name of the graph to use for knowledge graph-enhanced retrieval"
    )
    hop_depth: Optional[int] = Field(
        1, description="Number of relationship hops to traverse in the graph", ge=1, le=3
    )
    include_paths: Optional[bool] = Field(
        False, description="Whether to include relationship paths in the response"
    )


class CompletionQueryRequest(RetrieveRequest):
    """Request model for completion generation"""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    content: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    use_colpali: Optional[bool] = None


class CreateGraphRequest(BaseModel):
    """Request model for creating a graph"""

    name: str = Field(..., description="Name of the graph to create")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters to determine which documents to include"
    )
    documents: Optional[List[str]] = Field(
        None, description="Optional list of specific document IDs to include"
    )
