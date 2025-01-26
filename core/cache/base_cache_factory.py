from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
from .base_cache import BaseCache


class BaseCacheFactory(ABC):
    """Abstract base factory for creating and loading caches."""

    def __init__(self, storage_path: Path):
        """Initialize the cache factory.

        Args:
            storage_path: Base path for storing cache files
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_new_cache(
        self, name: str, model: str, model_file: str, **kwargs: Dict[str, Any]
    ) -> BaseCache:
        """Create a new cache instance.

        Args:
            name: Name of the cache
            model: Name/type of the model to use
            model_file: Path or identifier for the model file
            **kwargs: Additional arguments for cache creation

        Returns:
            BaseCache: The created cache instance
        """
        pass

    @abstractmethod
    def load_cache_from_bytes(
        self, name: str, cache_bytes: bytes, metadata: Dict[str, Any], **kwargs: Dict[str, Any]
    ) -> BaseCache:
        """Load a cache from its serialized bytes.

        Args:
            name: Name of the cache
            cache_bytes: Serialized cache data
            metadata: Cache metadata including model info
            **kwargs: Additional arguments for cache loading

        Returns:
            BaseCache: The loaded cache instance
        """
        pass

    def get_cache_path(self, name: str) -> Path:
        """Get the storage path for a cache.

        Args:
            name: Name of the cache

        Returns:
            Path: Directory path for the cache
        """
        path = self.storage_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path
