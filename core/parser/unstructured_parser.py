from typing import List, Union
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
import logging

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class UnstructuredAPIParser(BaseParser):
    def __init__(
        self,
        api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.api_key = api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    async def split_text(self, text: str) -> List[str]:
        """Split plain text into chunks"""
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Failed to split text: {str(e)}")
            raise

    async def parse_file(self, file: Union[UploadFile, bytes], content_type: str) -> List[str]:
        """Parse file content using unstructured"""
        try:
            # Handle different file input types
            if isinstance(file, UploadFile):
                file_content = await file.read()
            else:
                file_content = file

            # Parse with unstructured
            elements = partition(
                file=file_content,
                content_type=content_type,
                api_key=self.api_key
            )

            # Extract text from elements
            chunks = []
            for element in elements:
                if hasattr(element, 'text') and element.text:
                    chunks.append(element.text.strip())

            return chunks

        except Exception as e:
            logger.error(f"Failed to parse file: {str(e)}")
            raise
