import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
import tomli
from dotenv import load_dotenv
from collections import ChainMap


class Settings(BaseSettings):
    """DataBridge configuration settings."""

    # Environment variables
    JWT_SECRET_KEY: str
    POSTGRES_URI: Optional[str] = None
    MONGODB_URI: Optional[str] = None
    UNSTRUCTURED_API_KEY: Optional[str] = None
    AWS_ACCESS_KEY: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ASSEMBLYAI_API_KEY: Optional[str] = None

    # API configuration
    HOST: str
    PORT: int
    RELOAD: bool

    # Auth configuration
    JWT_ALGORITHM: str
    dev_mode: bool = False
    dev_entity_type: str = "developer"
    dev_entity_id: str = "dev_user"
    dev_permissions: list = ["read", "write", "admin"]

    # Completion configuration
    COMPLETION_PROVIDER: Literal["ollama", "openai"]
    COMPLETION_MODEL: str
    COMPLETION_MAX_TOKENS: Optional[str] = None
    COMPLETION_TEMPERATURE: Optional[float] = None
    COMPLETION_OLLAMA_BASE_URL: Optional[str] = None

    # Database configuration
    DATABASE_PROVIDER: Literal["postgres", "mongodb"]
    DATABASE_NAME: Optional[str] = None
    DOCUMENTS_COLLECTION: Optional[str] = None

    # Embedding configuration
    EMBEDDING_PROVIDER: Literal["ollama", "openai"]
    EMBEDDING_MODEL: str
    VECTOR_DIMENSIONS: int
    EMBEDDING_SIMILARITY_METRIC: Literal["cosine", "dotProduct"]
    EMBEDDING_OLLAMA_BASE_URL: Optional[str] = None

    # Parser configuration
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    USE_UNSTRUCTURED_API: bool
    FRAME_SAMPLE_RATE: Optional[int] = None
    USE_CONTEXTUAL_CHUNKING: bool = False

    # Rules configuration
    RULES_PROVIDER: Literal["ollama", "openai"]
    RULES_MODEL: str
    RULES_BATCH_SIZE: int = 4096

    # Graph configuration
    GRAPH_PROVIDER: Literal["ollama", "openai"]
    GRAPH_MODEL: str
    ENABLE_ENTITY_RESOLUTION: bool = True

    # Reranker configuration
    USE_RERANKING: bool
    RERANKER_PROVIDER: Optional[Literal["flag"]] = None
    RERANKER_MODEL: Optional[str] = None
    RERANKER_QUERY_MAX_LENGTH: Optional[int] = None
    RERANKER_PASSAGE_MAX_LENGTH: Optional[int] = None
    RERANKER_USE_FP16: Optional[bool] = None
    RERANKER_DEVICE: Optional[str] = None

    # Storage configuration
    STORAGE_PROVIDER: Literal["local", "aws-s3"]
    STORAGE_PATH: Optional[str] = None
    AWS_REGION: Optional[str] = None
    S3_BUCKET: Optional[str] = None

    # Vector store configuration
    VECTOR_STORE_PROVIDER: Literal["pgvector", "mongodb"]
    VECTOR_STORE_DATABASE_NAME: Optional[str] = None
    VECTOR_STORE_COLLECTION_NAME: Optional[str] = None

    # Colpali configuration
    ENABLE_COLPALI: bool
    
    # Mode configuration     
    MODE: Literal["cloud", "self_hosted"] = "cloud"
    
    # Telemetry configuration
    TELEMETRY_ENABLED: bool = True
    HONEYCOMB_ENABLED: bool = True
    HONEYCOMB_ENDPOINT: str = "https://api.honeycomb.io"
    HONEYCOMB_PROXY_ENDPOINT: str = "https://otel-proxy.onrender.com/"
    SERVICE_NAME: str = "databridge-core"
    OTLP_TIMEOUT: int = 10
    OTLP_MAX_RETRIES: int = 3
    OTLP_RETRY_DELAY: int = 1
    OTLP_MAX_EXPORT_BATCH_SIZE: int = 512
    OTLP_SCHEDULE_DELAY_MILLIS: int = 5000
    OTLP_MAX_QUEUE_SIZE: int = 2048


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    load_dotenv(override=True)

    # Load config.toml
    with open("databridge.toml", "rb") as f:
        config = tomli.load(f)

    em = "'{missing_value}' needed if '{field}' is set to '{value}'"
    # load api config
    api_config = {
        "HOST": config["api"]["host"],
        "PORT": int(config["api"]["port"]),
        "RELOAD": bool(config["api"]["reload"]),
    }

    # load auth config
    auth_config = {
        "JWT_ALGORITHM": config["auth"]["jwt_algorithm"],
        "JWT_SECRET_KEY": os.environ.get(
            "JWT_SECRET_KEY", "dev-secret-key"
        ),  # Default for dev mode
        "dev_mode": config["auth"].get("dev_mode", False),
        "dev_entity_type": config["auth"].get("dev_entity_type", "developer"),
        "dev_entity_id": config["auth"].get("dev_entity_id", "dev_user"),
        "dev_permissions": config["auth"].get("dev_permissions", ["read", "write", "admin"]),
    }

    # Only require JWT_SECRET_KEY in non-dev mode
    if not auth_config["dev_mode"] and "JWT_SECRET_KEY" not in os.environ:
        raise ValueError("JWT_SECRET_KEY is required when dev_mode is disabled")

    # load completion config
    completion_config = {
        "COMPLETION_PROVIDER": config["completion"]["provider"],
        "COMPLETION_MODEL": config["completion"]["model_name"],
    }
    match completion_config["COMPLETION_PROVIDER"]:
        case "openai" if "OPENAI_API_KEY" in os.environ:
            completion_config.update({"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
        case "openai":
            msg = em.format(
                missing_value="OPENAI_API_KEY", field="completion.provider", value="openai"
            )
            raise ValueError(msg)
        case "ollama" if "base_url" in config["completion"]:
            completion_config.update(
                {"COMPLETION_OLLAMA_BASE_URL": config["completion"]["base_url"]}
            )
        case "ollama":
            msg = em.format(missing_value="base_url", field="completion.provider", value="ollama")
            raise ValueError(msg)
        case _:
            prov = completion_config["COMPLETION_PROVIDER"]
            raise ValueError(f"Unknown completion provider selected: '{prov}'")

    # load database config
    database_config = {"DATABASE_PROVIDER": config["database"]["provider"]}
    match database_config["DATABASE_PROVIDER"]:
        case "mongodb":
            database_config.update(
                {
                    "DATABASE_NAME": config["database"]["database_name"],
                    "COLLECTION_NAME": config["database"]["collection_name"],
                }
            )
        case "postgres" if "POSTGRES_URI" in os.environ:
            database_config.update({"POSTGRES_URI": os.environ["POSTGRES_URI"]})
        case "postgres":
            msg = em.format(
                missing_value="POSTGRES_URI", field="database.provider", value="postgres"
            )
            raise ValueError(msg)
        case _:
            prov = database_config["DATABASE_PROVIDER"]
            raise ValueError(f"Unknown database provider selected: '{prov}'")

    # load embedding config
    embedding_config = {
        "EMBEDDING_PROVIDER": config["embedding"]["provider"],
        "EMBEDDING_MODEL": config["embedding"]["model_name"],
        "VECTOR_DIMENSIONS": config["embedding"]["dimensions"],
        "EMBEDDING_SIMILARITY_METRIC": config["embedding"]["similarity_metric"],
    }
    match embedding_config["EMBEDDING_PROVIDER"]:
        case "openai" if "OPENAI_API_KEY" in os.environ:
            embedding_config.update({"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
        case "openai":
            msg = em.format(
                missing_value="OPENAI_API_KEY", field="embedding.provider", value="openai"
            )
            raise ValueError(msg)
        case "ollama" if "base_url" in config["embedding"]:
            embedding_config.update({"EMBEDDING_OLLAMA_BASE_URL": config["embedding"]["base_url"]})
        case "ollama":
            msg = em.format(missing_value="base_url", field="embedding.provider", value="ollama")
            raise ValueError(msg)
        case _:
            prov = embedding_config["EMBEDDING_PROVIDER"]
            raise ValueError(f"Unknown embedding provider selected: '{prov}'")

    # load parser config
    parser_config = {
        "CHUNK_SIZE": config["parser"]["chunk_size"],
        "CHUNK_OVERLAP": config["parser"]["chunk_overlap"],
        "USE_UNSTRUCTURED_API": config["parser"]["use_unstructured_api"],
        "USE_CONTEXTUAL_CHUNKING": config["parser"].get("use_contextual_chunking", False),
    }
    if parser_config["USE_UNSTRUCTURED_API"] and "UNSTRUCTURED_API_KEY" not in os.environ:
        msg = em.format(
            missing_value="UNSTRUCTURED_API_KEY", field="parser.use_unstructured_api", value="true"
        )
        raise ValueError(msg)
    elif parser_config["USE_UNSTRUCTURED_API"]:
        parser_config.update({"UNSTRUCTURED_API_KEY": os.environ["UNSTRUCTURED_API_KEY"]})

    # load reranker config
    reranker_config = {"USE_RERANKING": config["reranker"]["use_reranker"]}
    if reranker_config["USE_RERANKING"]:
        reranker_config.update(
            {
                "RERANKER_PROVIDER": config["reranker"]["provider"],
                "RERANKER_MODEL": config["reranker"]["model_name"],
                "RERANKER_QUERY_MAX_LENGTH": config["reranker"]["query_max_length"],
                "RERANKER_PASSAGE_MAX_LENGTH": config["reranker"]["passage_max_length"],
                "RERANKER_USE_FP16": config["reranker"]["use_fp16"],
                "RERANKER_DEVICE": config["reranker"]["device"],
            }
        )

    # load storage config
    storage_config = {"STORAGE_PROVIDER": config["storage"]["provider"], "STORAGE_PATH": config["storage"]["storage_path"]}
    match storage_config["STORAGE_PROVIDER"]:
        case "local":
            storage_config.update({"STORAGE_PATH": config["storage"]["storage_path"]})
        case "aws-s3" if all(
            key in os.environ for key in ["AWS_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"]
        ):
            storage_config.update(
                {
                    "AWS_REGION": config["storage"]["region"],
                    "S3_BUCKET": config["storage"]["bucket_name"],
                    "AWS_ACCESS_KEY": os.environ["AWS_ACCESS_KEY"],
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                }
            )
        case "aws-s3":
            msg = em.format(
                missing_value="AWS credentials", field="storage.provider", value="aws-s3"
            )
            raise ValueError(msg)
        case _:
            prov = storage_config["STORAGE_PROVIDER"]
            raise ValueError(f"Unknown storage provider selected: '{prov}'")

    # load vector store config
    vector_store_config = {"VECTOR_STORE_PROVIDER": config["vector_store"]["provider"]}
    match vector_store_config["VECTOR_STORE_PROVIDER"]:
        case "mongodb":
            vector_store_config.update(
                {
                    "VECTOR_STORE_DATABASE_NAME": config["vector_store"]["database_name"],
                    "VECTOR_STORE_COLLECTION_NAME": config["vector_store"]["collection_name"],
                }
            )
        case "pgvector":
            if "POSTGRES_URI" not in os.environ:
                msg = em.format(
                    missing_value="POSTGRES_URI", field="vector_store.provider", value="pgvector"
                )
                raise ValueError(msg)
        case _:
            prov = vector_store_config["VECTOR_STORE_PROVIDER"]
            raise ValueError(f"Unknown vector store provider selected: '{prov}'")

    # load rules config - simplified
    rules_config = {
        "RULES_PROVIDER": config["rules"]["provider"],
        "RULES_MODEL": config["rules"]["model_name"],
        "RULES_BATCH_SIZE": config["rules"]["batch_size"],
    }

    # load databridge config
    databridge_config = {
        "ENABLE_COLPALI": config["databridge"]["enable_colpali"],
        "MODE": config["databridge"].get("mode", "cloud"),  # Default to "cloud" mode
    }

    # load graph config
    graph_config = {
        "GRAPH_PROVIDER": config["graph"]["provider"],
        "GRAPH_MODEL": config["graph"]["model_name"],
        "ENABLE_ENTITY_RESOLUTION": config["graph"].get("enable_entity_resolution", True),
    }
    
    # load telemetry config
    telemetry_config = {}
    if "telemetry" in config:
        telemetry_config = {
            "TELEMETRY_ENABLED": config["telemetry"].get("enabled", True),
            "HONEYCOMB_ENABLED": config["telemetry"].get("honeycomb_enabled", True),
            "HONEYCOMB_ENDPOINT": config["telemetry"].get("honeycomb_endpoint", "https://api.honeycomb.io"),
            "SERVICE_NAME": config["telemetry"].get("service_name", "databridge-core"),
            "OTLP_TIMEOUT": config["telemetry"].get("otlp_timeout", 10),
            "OTLP_MAX_RETRIES": config["telemetry"].get("otlp_max_retries", 3),
            "OTLP_RETRY_DELAY": config["telemetry"].get("otlp_retry_delay", 1),
            "OTLP_MAX_EXPORT_BATCH_SIZE": config["telemetry"].get("otlp_max_export_batch_size", 512),
            "OTLP_SCHEDULE_DELAY_MILLIS": config["telemetry"].get("otlp_schedule_delay_millis", 5000),
            "OTLP_MAX_QUEUE_SIZE": config["telemetry"].get("otlp_max_queue_size", 2048),
        }
    
    settings_dict = dict(ChainMap(
        api_config,
        auth_config,
        completion_config,
        database_config,
        embedding_config,
        parser_config,
        reranker_config,
        storage_config,
        vector_store_config,
        rules_config,
        databridge_config,
        graph_config,
        telemetry_config,
    ))

    return Settings(**settings_dict)
