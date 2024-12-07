import base64
import logging
from typing import Tuple, Optional, Union, BinaryIO
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from .base_storage import BaseStorage
from .utils_file_extensions import detect_file_type

logger = logging.getLogger(__name__)


class S3Storage(BaseStorage):
    """AWS S3 storage implementation."""
    
    # TODO: Remove hardcoded values.
    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        region_name: str = "us-east-2",
        default_bucket: str = "databridge-storage"
    ):
        self.default_bucket = default_bucket
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )

    async def upload_file(
        self,
        file: Union[str, bytes, BinaryIO],
        key: str,
        content_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Upload a file to S3."""
        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            if isinstance(file, (str, bytes)):
                # Create temporary file for content
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    if isinstance(file, str):
                        temp_file.write(file.encode())
                    else:
                        temp_file.write(file)
                    temp_file_path = temp_file.name

                try:
                    self.s3_client.upload_file(
                        temp_file_path,
                        self.default_bucket,
                        key,
                        ExtraArgs=extra_args
                    )
                finally:
                    Path(temp_file_path).unlink()
            else:
                # File object
                self.s3_client.upload_fileobj(
                    file,
                    self.default_bucket,
                    key,
                    ExtraArgs=extra_args
                )

            return self.default_bucket, key

        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    async def upload_from_base64(
        self,
        content: str,
        key: str,
        content_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Upload base64 encoded content to S3."""
        try:
            decoded_content = base64.b64decode(content)
            extension = detect_file_type(content)
            key = f"{key}{extension}"
            
            return await self.upload_file(
                file=decoded_content,
                key=key,
                content_type=content_type
            )

        except Exception as e:
            logger.error(f"Error uploading base64 content to S3: {e}")
            raise e

    async def download_file(self, bucket: str, key: str) -> bytes:
        """Download file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise

    async def get_download_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        """Generate presigned download URL."""
        if not key or not bucket:
            return ""
            
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in
            )
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return ""

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"File {key} deleted from bucket {bucket}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False
