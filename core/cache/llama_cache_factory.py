from core.cache.base_cache_factory import BaseCacheFactory
from core.cache.llama_cache import LlamaCache
from typing import Dict, Any


class LlamaCacheFactory(BaseCacheFactory):
    def create_new_cache(
        self, name: str, model: str, model_file: str, **kwargs: Dict[str, Any]
    ) -> LlamaCache:
        return LlamaCache(name, model, model_file, **kwargs)

    def load_cache_from_bytes(
        self, name: str, cache_bytes: bytes, metadata: Dict[str, Any], **kwargs: Dict[str, Any]
    ) -> LlamaCache:
        return LlamaCache.from_bytes(name, cache_bytes, metadata, **kwargs)
