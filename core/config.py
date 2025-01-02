from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import tomli
from dotenv import load_dotenv


class Settings(BaseSettings):
    """DataBridge configuration settings."""

    # Required environment variables (referenced in config.toml)
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    MONGODB_URI: Optional[str] = Field(None, env="MONGODB_URI")
    POSTGRES_URI: Optional[str] = Field(None, env="POSTGRES_URI")

    UNSTRUCTURED_API_KEY: Optional[str] = Field(None, env="UNSTRUCTURED_API_KEY")
    AWS_ACCESS_KEY: Optional[str] = Field(None, env="AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    ASSEMBLYAI_API_KEY: Optional[str] = Field(None, env="ASSEMBLYAI_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Service settings
    HOST: str = "localhost"
    PORT: int = 8000
    RELOAD: bool = False

    # Component selection
    STORAGE_PROVIDER: str = "local"
    DATABASE_PROVIDER: str = "mongodb"
    VECTOR_STORE_PROVIDER: str = "mongodb"
    EMBEDDING_PROVIDER: str = "openai"
    COMPLETION_PROVIDER: str = "ollama"
    PARSER_PROVIDER: str = "combined"
    RERANKER_PROVIDER: str = "bge"

    # Storage settings
    STORAGE_PATH: str = "./storage"
    AWS_REGION: str = "us-east-2"
    S3_BUCKET: str = "databridge-s3-storage"

    # Database settings
    DATABRIDGE_DB: str = "DataBridgeTest"
    DOCUMENTS_TABLE: str = "documents"
    CHUNKS_TABLE: str = "document_chunks"
    DOCUMENTS_COLLECTION: str = "documents"
    CHUNKS_COLLECTION: str = "document_chunks"

    # Vector store settings
    VECTOR_INDEX_NAME: str = "vector_index"
    VECTOR_DIMENSIONS: int = 1536

    # Model settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    COMPLETION_MODEL: str = "llama3.1"
    COMPLETION_MAX_TOKENS: int = 1000
    COMPLETION_TEMPERATURE: float = 0.7
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-gemma"
    RERANKER_DEVICE: Optional[str] = None
    RERANKER_USE_FP16: bool = True
    RERANKER_QUERY_MAX_LENGTH: int = 256
    RERANKER_PASSAGE_MAX_LENGTH: int = 512

    # Processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DEFAULT_K: int = 4
    FRAME_SAMPLE_RATE: int = 120
    USE_UNSTRUCTURED_API: bool = False
    USE_RERANKING: bool = True

    # Auth settings
    JWT_ALGORITHM: str = "HS256"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    load_dotenv(override=True)

    # Load config.toml
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    # Map config.toml values to settings
    settings_dict = {
        # Service settings
        "HOST": config["service"]["host"],
        "PORT": config["service"]["port"],
        "RELOAD": config["service"]["reload"],
        # Component selection
        "STORAGE_PROVIDER": config["service"]["components"]["storage"],
        "DATABASE_PROVIDER": config["service"]["components"]["database"],
        "VECTOR_STORE_PROVIDER": config["service"]["components"]["vector_store"],
        "EMBEDDING_PROVIDER": config["service"]["components"]["embedding"],
        "COMPLETION_PROVIDER": config["service"]["components"]["completion"],
        "PARSER_PROVIDER": config["service"]["components"]["parser"],
        "RERANKER_PROVIDER": config["service"]["components"]["reranker"],
        # Storage settings
        "STORAGE_PATH": config["storage"]["local"]["path"],
        "AWS_REGION": config["storage"]["aws"]["region"],
        "S3_BUCKET": config["storage"]["aws"]["bucket_name"],
        # Database settings
        "DATABRIDGE_DB": config["database"][config["service"]["components"]["database"]][
            "database_name"
        ],
        "DOCUMENTS_TABLE": config["database"]
        .get("postgres", {})
        .get("documents_table", "documents"),
        "CHUNKS_TABLE": config["database"]
        .get("postgres", {})
        .get("chunks_table", "document_chunks"),
        "DOCUMENTS_COLLECTION": config["database"]
        .get("mongodb", {})
        .get("documents_collection", "documents"),
        "CHUNKS_COLLECTION": config["database"]
        .get("mongodb", {})
        .get("chunks_collection", "document_chunks"),
        # Vector store settings
        "VECTOR_INDEX_NAME": config["vector_store"]["mongodb"]["index_name"],
        "VECTOR_DIMENSIONS": config["vector_store"]["mongodb"]["dimensions"],
        # Model settings
        "EMBEDDING_MODEL": config["models"]["embedding"]["model_name"],
        "COMPLETION_MODEL": config["models"]["completion"]["model_name"],
        "COMPLETION_MAX_TOKENS": config["models"]["completion"]["default_max_tokens"],
        "COMPLETION_TEMPERATURE": config["models"]["completion"]["default_temperature"],
        "OLLAMA_BASE_URL": config["models"]["ollama"]["base_url"],
        "RERANKER_MODEL": config["models"]["reranker"]["model_name"],
        "RERANKER_DEVICE": config["models"]["reranker"].get("device"),
        "RERANKER_USE_FP16": config["models"]["reranker"].get("use_fp16", True),
        "RERANKER_QUERY_MAX_LENGTH": config["models"]["reranker"].get("query_max_length", 256),
        "RERANKER_PASSAGE_MAX_LENGTH": config["models"]["reranker"].get("passage_max_length", 512),
        # Processing settings
        "CHUNK_SIZE": config["processing"]["text"]["chunk_size"],
        "CHUNK_OVERLAP": config["processing"]["text"]["chunk_overlap"],
        "DEFAULT_K": config["processing"]["text"]["default_k"],
        "USE_RERANKING": config["processing"]["text"]["use_reranking"],
        "FRAME_SAMPLE_RATE": config["processing"]["video"]["frame_sample_rate"],
        "USE_UNSTRUCTURED_API": config["processing"]["unstructured"]["use_api"],
        # Auth settings
        "JWT_ALGORITHM": config["auth"]["jwt_algorithm"],
    }

    return Settings(**settings_dict)
