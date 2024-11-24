# core/auth/uri_service.py
from typing import Optional, Set
from datetime import datetime, timedelta, UTC
import jwt
from functools import lru_cache

from ..models.auth import EntityType, AuthContext
from ..config import Settings


class URIService:
    """Service for creating and validating DataBridge URIs with authentication tokens"""
    
    def __init__(self, settings: Settings):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.host = settings.HOST
        self.port = settings.PORT
    
    def create_developer_uri(
        self, 
        dev_id: str,
        app_id: Optional[str] = None,
        expiry_days: int = 30,
        permissions: Optional[Set[str]] = None
    ) -> str:
        """
        Create URI for developer access.
        
        Args:
            dev_id: Developer ID
            app_id: Optional application ID for app-specific access
            expiry_days: Token validity in days
            permissions: Set of permissions to grant
        """
        payload = {
            "type": EntityType.DEVELOPER,
            "entity_id": dev_id,
            "permissions": list(permissions or {"read"}),
            "exp": datetime.now(UTC) + timedelta(days=expiry_days)
        }
        
        if app_id:
            payload["app_id"] = app_id
        
        token = jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        # Construct the URI
        owner_id = f"{dev_id}.{app_id}" if app_id else dev_id
        host_port = f"{self.host}:{self.port}" if self.port != 80 else self.host
        return f"databridge://{owner_id}:{token}@{host_port}"
        
    def create_user_uri(
        self,
        user_id: str,
        expiry_days: int = 30,
        permissions: Optional[Set[str]] = None
    ) -> str:
        """
        Create URI for end-user access.
        
        Args:
            user_id: User ID 
            expiry_days: Token validity in days
            permissions: Set of permissions to grant
        """
        token = jwt.encode(
            {
                "type": EntityType.USER,
                "entity_id": user_id,
                "permissions": list(permissions or {"read"}),
                "exp": datetime.now(UTC) + timedelta(days=expiry_days)
            },
            self.secret_key,
            algorithm=self.algorithm
        )
        
        host_port = f"{self.host}:{self.port}" if self.port != 80 else self.host
        return f"databridge://{user_id}:{token}@{host_port}"

    def validate_uri(self, uri: str) -> Optional[AuthContext]:
        """
        Validate a DataBridge URI and return auth context if valid.
        
        Args:
            uri: DataBridge URI to validate
            
        Returns:
            AuthContext if valid, None otherwise
        """
        try:
            # Extract token from URI
            token = uri.split("://")[1].split("@")[0].split(":")[1]
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            return AuthContext(
                entity_type=EntityType(payload["type"]),
                entity_id=payload["entity_id"],
                app_id=payload.get("app_id"),
                permissions=set(payload.get("permissions", ["read"]))
            )
            
        except (jwt.InvalidTokenError, IndexError, ValueError):
            return None


@lru_cache()
def get_uri_service(settings: Settings = None) -> URIService:
    """Get cached URIService instance."""
    if settings is None:
        from ..config import get_settings
        settings = get_settings()
    return URIService(settings)
