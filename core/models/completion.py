from pydantic import BaseModel
from typing import Dict, List, Optional


class CompletionResponse(BaseModel):
    """Response from completion generation"""

    completion: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None


class CompletionRequest(BaseModel):
    """Request for completion generation"""

    query: str
    context_chunks: List[str]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
