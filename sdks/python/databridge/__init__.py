"""
DataBridge Python SDK for document ingestion and querying.
"""

from .sync import DataBridge
from .async_ import AsyncDataBridge
from .models import Document, IngestTextRequest

__all__ = [
    "DataBridge",
    "AsyncDataBridge",
    "Document",
    "IngestTextRequest",
]

__version__ = "0.2.0"
