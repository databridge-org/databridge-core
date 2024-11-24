from typing import Optional, Set
from pydantic import BaseModel
from enum import Enum


class EntityType(str, Enum):
    USER = "user"
    DEVELOPER = "developer"


class AuthContext(BaseModel):
    """JWT decoded context"""
    entity_type: EntityType
    entity_id: str  # uuid
    app_id: Optional[str] = None  # uuid, only for developers
    permissions: Set[str] = {"read"}
