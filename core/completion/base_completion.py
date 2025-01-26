from abc import ABC, abstractmethod
from core.models.completion import CompletionRequest, CompletionResponse


class BaseCompletionModel(ABC):
    """Base class for completion models"""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion from query and context"""
        pass
