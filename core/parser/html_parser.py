from typing import List
import logging
from bs4 import BeautifulSoup
import markdownify
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.models.documents import Chunk
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Configure markdown splitter with common headers
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

        # Fallback character splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def split_text(self, text: str) -> List[Chunk]:
        """Split plain text into chunks"""
        return [
            Chunk(content=chunk, metadata={})
            for chunk in self.text_splitter.split_text(text)
        ]

    async def parse_file(self, file: bytes, content_type: str) -> List[Chunk]:
        """Parse HTML content into markdown chunks"""
        try:
            # Parse HTML
            html_content = file.decode("utf-8")
            soup = BeautifulSoup(html_content, "html.parser")

            # Convert to markdown
            markdown = markdownify.markdownify(str(soup), heading_style="ATX")

            # Try header-based splitting first
            try:
                md_chunks = self.md_splitter.split_text(markdown)
                chunks = []
                for md_chunk in md_chunks:
                    # Extract header metadata
                    metadata = {
                        k: v
                        for k, v in md_chunk.metadata.items()
                        if k in [h[1] for h in self.headers_to_split_on]
                    }

                    # Further split if chunk is too large
                    if len(md_chunk.page_content) > self.chunk_size:
                        sub_chunks = self.text_splitter.split_text(
                            md_chunk.page_content
                        )
                        # Add same header metadata to all sub-chunks
                        chunks.extend(
                            [Chunk(content=sc, metadata=metadata) for sc in sub_chunks]
                        )
                    else:
                        chunks.append(
                            Chunk(content=md_chunk.page_content, metadata=metadata)
                        )
                return chunks

            except Exception as e:
                logger.warning(
                    f"Header-based splitting failed: {str(e)}. Falling back to character splitting."
                )
                # Fallback to basic character splitting
                return await self.split_text(markdown)

        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            raise
