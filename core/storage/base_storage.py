from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, BinaryIO
from pathlib import Path


class BaseStorage(ABC):
    """Base interface for storage providers."""
    
    @abstractmethod
    async def upload_file(self, 
                         file: Union[str, bytes, BinaryIO], 
                         key: str,
                         content_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Upload a file to storage.
        
        Args:
            file: File content as string, bytes or file object
            key: Storage key/path for the file
            content_type: Optional MIME type
            
        Returns:
            Tuple[str, str]: (bucket/container name, storage key)
        """
        pass

    @abstractmethod
    async def upload_from_base64(self,
                                content: str,
                                key: str,
                                content_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Upload base64 encoded content.
        
        Args:
            content: Base64 encoded content
            key: Storage key/path
            content_type: Optional MIME type
            
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
