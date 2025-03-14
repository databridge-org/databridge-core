from typing import Dict, Any, List, Optional, Set
from datetime import UTC, datetime
from pydantic import BaseModel, Field
import uuid


class Entity(BaseModel):
    """Represents an entity in a knowledge graph"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    document_ids: List[str] = Field(default_factory=list)
    chunk_sources: Dict[str, List[int]] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


class Relationship(BaseModel):
    """Represents a relationship between entities in a knowledge graph"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    document_ids: List[str] = Field(default_factory=list)
    chunk_sources: Dict[str, List[int]] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return self.id == other.id


class Graph(BaseModel):
    """Represents a knowledge graph"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_ids: List[str] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    owner: Dict[str, str] = Field(default_factory=dict)
    access_control: Dict[str, List[str]] = Field(
        default_factory=lambda: {"readers": [], "writers": [], "admins": []}
    )
