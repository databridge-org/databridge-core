from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """DataBridge configuration settings."""
    
    # MongoDB settings
    MONGODB_URI: str = Field(..., env="MONGODB_URI")
    DATABRIDGE_DB: str = Field(..., env="DATABRIDGE_DB")
    
    # Collection names
    DOCUMENTS_COLLECTION: str = Field("documents", env="DOCUMENTS_COLLECTION")
    CHUNKS_COLLECTION: str = Field("document_chunks", env="CHUNKS_COLLECTION")
    
    # Vector search settings
    VECTOR_INDEX_NAME: str = Field("vector_index", env="VECTOR_INDEX_NAME")
    
    # API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    UNSTRUCTURED_API_KEY: str = Field(..., env="UNSTRUCTURED_API_KEY")
    
    # Optional API keys for alternative models
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    COHERE_API_KEY: Optional[str] = Field(None, env="COHERE_API_KEY")
    VOYAGE_API_KEY: Optional[str] = Field(None, env="VOYAGE_API_KEY")
    
    # Model settings
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Document processing settings
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")
    DEFAULT_K: int = Field(4, env="DEFAULT_K")
    
    # Storage settings
    AWS_ACCESS_KEY: str = Field(..., env="AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field("us-east-2", env="AWS_REGION")
    S3_BUCKET: str = Field("databridge-storage", env="S3_BUCKET")
    
    # Auth settings
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    
    # Server settings
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    RELOAD: bool = Field(False, env="RELOAD")

    def get_mongodb_settings(self) -> Dict[str, Any]:
        """Get MongoDB related settings."""
        return {
            "uri": self.MONGODB_URI,
            "db_name": self.DATABRIDGE_DB,
            "collection_name": self.DOCUMENTS_COLLECTION
        }

    def get_vector_store_settings(self) -> Dict[str, Any]:
        """Get vector store related settings."""
        return {
            "uri": self.MONGODB_URI,
            "database_name": self.DATABRIDGE_DB,
            "collection_name": self.CHUNKS_COLLECTION,
            "index_name": self.VECTOR_INDEX_NAME
        }

    def get_storage_settings(self) -> Dict[str, Any]:
        """Get storage related settings."""
        return {
            "aws_access_key": self.AWS_ACCESS_KEY,
            "aws_secret_key": self.AWS_SECRET_ACCESS_KEY,
            "region_name": self.AWS_REGION,
            "default_bucket": self.S3_BUCKET
        }

    def get_parser_settings(self) -> Dict[str, Any]:
        """Get document parser settings."""
        return {
            "api_key": self.UNSTRUCTURED_API_KEY,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP
        }

    def get_embedding_settings(self) -> Dict[str, Any]:
        """Get embedding model settings."""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model_name": self.EMBEDDING_MODEL
        }

    def get_server_settings(self) -> Dict[str, Any]:
        """Get server related settings."""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "reload": self.RELOAD,
        }

    def get_auth_settings(self) -> Dict[str, Any]:
        """Get authentication related settings."""
        return {
            "secret_key": self.JWT_SECRET_KEY,
            "algorithm": self.JWT_ALGORITHM
        }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields in settings


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()