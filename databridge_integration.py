import sys
from pathlib import Path

# Add local SDK to path before other imports
_SDK_PATH = str(Path(__file__).parent / "sdks" / "python")
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from databridge import DataBridge  # noqa: E402


class DatabridgeIntegrator:
    def __init__(self):
        """Initialize DataBridge client"""
        self.client = DataBridge()

    def ingest_document_content(
        self, content: str, image_descriptions: list, full_text: str, metadata: dict = None
    ):
        """
        Ingest document content with image descriptions into DataBridge.

        Args:
            content: Combined content of document and image descriptions
            image_descriptions: List of image descriptions
            full_text: Original document text
            metadata: Optional metadata about the document
        """
        # Combine full text with image descriptions
        combined_content = f"Document Text:\n{full_text}\n\nImage Descriptions:\n"
        for idx, desc in enumerate(image_descriptions, 1):
            combined_content += f"\nImage {idx}:\n{desc}\n"
            combined_content += "-" * 80 + "\n"

        # Add metadata if provided, or create default
        doc_metadata = metadata or {}
        doc_metadata.update(
            {
                "has_images": len(image_descriptions) > 0,
                "num_images": len(image_descriptions),
                "content_type": "document_with_images",
            }
        )

        # Ingest the combined content
        try:
            response = self.client.ingest_text(content=combined_content, metadata=doc_metadata)
            return response
        except Exception as e:
            print(f"Error ingesting document: {str(e)}")
            return None

    def search_documents(self, query: str, filters: dict = None, k: int = 15):
        """
        Search through ingested documents and get an AI completion.

        Args:
            query: Search query string
            filters: Optional metadata filters
            k: Number of relevant chunks to use for context
        """
        try:
            # Get completion using relevant chunks as context
            response = self.client.query(
                query=query,
                filters=filters,
                k=k,
                max_tokens=500,  # Reasonable length for responses
                temperature=0.3,  # Some creativity while staying focused
            )
            return response.completion if response else None
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return None
