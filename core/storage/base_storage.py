from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseStorage(ABC):
    """Base interface for storage providers."""

    @abstractmethod
    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        """
        Upload base64 encoded content.

        Args:
            content: Base64 encoded content
            key: Storage key/path
            content_type: Optional MIME type
            bucket: Optional bucket/folder name
        Returns:
            Tuple[str, str]: (bucket/container name, storage key)
        """
        pass

    @abstractmethod
    async def download_file(self, bucket: str, key: str) -> bytes:
        """
        Download file from storage.

        Args:
            bucket: Bucket/container name
            key: Storage key/path

        Returns:
            bytes: File content
        """
        pass

    @abstractmethod
    async def get_download_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        """
        Get temporary download URL.

        Args:
            bucket: Bucket/container name
            key: Storage key/path
            expires_in: URL expiration in seconds

        Returns:
            str: Presigned download URL
        """
        pass

    @abstractmethod
    async def delete_file(self, bucket: str, key: str) -> bool:
        """
        Delete file from storage.

        Args:
            bucket: Bucket/container name
            key: Storage key/path

        Returns:
            bool: True if successful
        """
        pass
