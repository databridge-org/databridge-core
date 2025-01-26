import base64
from pathlib import Path
from typing import Tuple, Optional, BinaryIO
from .base_storage import BaseStorage


class LocalStorage(BaseStorage):
    def __init__(self, storage_path: str):
        """Initialize local storage with a base path."""
        self.storage_path = Path(storage_path)
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def download_file(self, bucket: str, key: str) -> BinaryIO:
        """Download a file from local storage."""
        file_path = self.storage_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return open(file_path, "rb")

    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        base64_content = content
        """Upload base64 encoded content to local storage."""
        # Decode base64 content
        file_content = base64.b64decode(base64_content)

        key = f"{bucket}/{key}" if bucket else key
        # Create file path
        file_path = self.storage_path / key

        # Write content to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.unlink(missing_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        return str(self.storage_path), key

    async def get_download_url(self, bucket: str, key: str) -> str:
        """Get local file path as URL."""
        file_path = self.storage_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return f"file://{file_path.absolute()}"

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete a file from local storage."""
        file_path = self.storage_path / key
        if file_path.exists():
            file_path.unlink()
        return True
