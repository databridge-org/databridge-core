from typing import List
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
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
        return self.text_splitter.split_text(text)

    async def parse_file(self, file: bytes, content_type: str) -> List[str]:
        """Parse file content using unstructured"""
        # Parse with unstructured
        loader = UnstructuredLoader(
            file=io.BytesIO(file),
            content_type=content_type,
            partition_via_api=True,
            api_key=self.api_key,
            chunking_strategy="by_title"
        )
        elements = loader.load()
        return [element.page_content for element in elements]
