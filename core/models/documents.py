from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from datetime import UTC, datetime
from pydantic import BaseModel, Field, field_validator
import uuid
import logging

from core.models.video import TimeSeriesData

logger = logging.getLogger(__name__)


class QueryReturnType(str, Enum):
    CHUNKS = "chunks"
    DOCUMENTS = "documents"


class Document(BaseModel):
    """Represents a document stored in MongoDB documents collection"""

    external_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner: Dict[str, str]
    content_type: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """user-defined metadata"""
    storage_info: Dict[str, str] = Field(default_factory=dict)
    system_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "version": 1,
        }
    )
    """metadata such as creation date etc."""
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)
    """metadata to help with querying eg. frame descriptions and time-stamped transcript for videos"""
    access_control: Dict[str, List[str]] = Field(
        default_factory=lambda: {"readers": [], "writers": [], "admins": []}
    )
    chunk_ids: List[str] = Field(default_factory=list)

    def __hash__(self):
        return hash(self.external_id)

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.external_id == other.external_id


class DocumentContent(BaseModel):
    """Represents either a URL or content string"""

    type: Literal["url", "string"]
    value: str
    filename: Optional[str] = Field(None, description="Filename when type is url")

    @field_validator("filename")
    def filename_only_for_url(cls, v, values):
        logger.debug(f"Value looks like: {values}")
        if values.data.get("type") == "string" and v is not None:
            raise ValueError("filename can only be set when type is url")
        if values.data.get("type") == "url" and v is None:
            raise ValueError("filename is required when type is url")
        return v


class DocumentResult(BaseModel):
    """Query result at document level"""

    score: float  # Highest chunk score
    document_id: str  # external_id
    metadata: Dict[str, Any]
    content: DocumentContent
    additional_metadata: Dict[str, Any]


class ChunkResult(BaseModel):
    """Query result at chunk level"""

    content: str
    score: float
    document_id: str  # external_id
    chunk_number: int
    metadata: Dict[str, Any]
    content_type: str
    filename: Optional[str] = None
    download_url: Optional[str] = None

    def augmented_content(self, doc: DocumentResult) -> str:
        match self.metadata:
            case m if "timestamp" in m:
                # if timestamp present, then must be a video. In that case,
                # obtain the original document and augment the content with
                # frame/transcript information as well.
                frame_description = doc.additional_metadata.get("frame_description")
                transcript = doc.additional_metadata.get("transcript")
                if not isinstance(frame_description, dict) or not isinstance(transcript, dict):
                    logger.warning("Invalid frame description or transcript - not a dictionary")
                    return self.content
                ts_frame = TimeSeriesData(time_to_content=frame_description)
                ts_transcript = TimeSeriesData(time_to_content=transcript)
                timestamps = (
                    ts_frame.content_to_times[self.content]
                    + ts_transcript.content_to_times[self.content]
                )
                augmented_contents = [
                    f"Frame description: {ts_frame.at_time(t)} \n \n Transcript: {ts_transcript.at_time(t)}"
                    for t in timestamps
                ]
                return "\n\n".join(augmented_contents)
            case _:
                return self.content
