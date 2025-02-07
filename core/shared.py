"""Shared components used across multiple API modules."""

import logging
from pathlib import Path
from fastapi import HTTPException, Header
import jwt
from datetime import datetime, UTC

from core.models.auth import AuthContext, EntityType
from core.config import get_settings
from core.services.document_service import DocumentService
from core.database.mongo_database import MongoDatabase
from core.database.postgres_database import PostgresDatabase
from core.vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from core.storage.s3_storage import S3Storage
from core.storage.local_storage import LocalStorage
from core.embedding.openai_embedding_model import OpenAIEmbeddingModel
from core.completion.ollama_completion import OllamaCompletionModel
from core.parser.contextual_parser import ContextualParser
from core.reranker.flag_reranker import FlagReranker
from core.cache.llama_cache_factory import LlamaCacheFactory
from core.parser.combined_parser import CombinedParser
from core.parser.unstructured_parser import UnstructuredParser
from core.embedding.ollama_embedding_model import OllamaEmbeddingModel
from core.completion.openai_completion import OpenAICompletionModel

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize database
match settings.DATABASE_PROVIDER:
    case "postgres":
        if not settings.POSTGRES_URI:
            raise ValueError("PostgreSQL URI is required for PostgreSQL database")
        database = PostgresDatabase(uri=settings.POSTGRES_URI)
    case "mongodb":
        if not settings.MONGODB_URI:
            raise ValueError("MongoDB URI is required for MongoDB database")
        database = MongoDatabase(
            uri=settings.MONGODB_URI,
            db_name=settings.DATABRIDGE_DB,
            collection_name=settings.DOCUMENTS_COLLECTION,
        )
    case _:
        raise ValueError(f"Unsupported database provider: {settings.DATABASE_PROVIDER}")

# Initialize vector store
match settings.VECTOR_STORE_PROVIDER:
    case "mongodb":
        vector_store = MongoDBAtlasVectorStore(
            uri=settings.MONGODB_URI,
            database_name=settings.DATABRIDGE_DB,
            collection_name=settings.CHUNKS_COLLECTION,
            index_name=settings.VECTOR_INDEX_NAME,
        )
    case "pgvector":
        if not settings.POSTGRES_URI:
            raise ValueError("PostgreSQL URI is required for pgvector store")
        from core.vector_store.pgvector_store import PGVectorStore

        vector_store = PGVectorStore(
            uri=settings.POSTGRES_URI,
        )
    case _:
        raise ValueError(f"Unsupported vector store provider: {settings.VECTOR_STORE_PROVIDER}")

# Initialize storage
match settings.STORAGE_PROVIDER:
    case "local":
        storage = LocalStorage(storage_path=settings.STORAGE_PATH)
    case "aws-s3":
        if not settings.AWS_ACCESS_KEY or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS credentials are required for S3 storage")
        storage = S3Storage(
            aws_access_key=settings.AWS_ACCESS_KEY,
            aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            default_bucket=settings.S3_BUCKET,
        )
    case _:
        raise ValueError(f"Unsupported storage provider: {settings.STORAGE_PROVIDER}")

# Initialize parser
match settings.PARSER_PROVIDER:
    case "combined":
        if not settings.ASSEMBLYAI_API_KEY:
            raise ValueError("AssemblyAI API key is required for combined parser")
        parser = CombinedParser(
            use_unstructured_api=settings.USE_UNSTRUCTURED_API,
            unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
            assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            frame_sample_rate=settings.FRAME_SAMPLE_RATE,
        )
    case "unstructured":
        parser = UnstructuredParser(
            use_api=settings.USE_UNSTRUCTURED_API,
            api_key=settings.UNSTRUCTURED_API_KEY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    case "contextual":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key is required for contextual parser")
        parser = ContextualParser(
            use_unstructured_api=settings.USE_UNSTRUCTURED_API,
            unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
            assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            frame_sample_rate=settings.FRAME_SAMPLE_RATE,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
        )
    case _:
        raise ValueError(f"Unsupported parser provider: {settings.PARSER_PROVIDER}")

# Initialize embedding model
match settings.EMBEDDING_PROVIDER:
    case "ollama":
        embedding_model = OllamaEmbeddingModel(
            base_url=settings.EMBEDDING_OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL,
        )
    case "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for OpenAI embedding model")
        embedding_model = OpenAIEmbeddingModel(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )
    case _:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")

# Initialize completion model
match settings.COMPLETION_PROVIDER:
    case "ollama":
        completion_model = OllamaCompletionModel(
            model_name=settings.COMPLETION_MODEL,
            base_url=settings.COMPLETION_OLLAMA_BASE_URL,
        )
    case "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for OpenAI completion model")
        completion_model = OpenAICompletionModel(
            model_name=settings.COMPLETION_MODEL,
        )
    case _:
        raise ValueError(f"Unsupported completion provider: {settings.COMPLETION_PROVIDER}")

# Initialize reranker
reranker = None
if settings.USE_RERANKING:
    match settings.RERANKER_PROVIDER:
        case "flag":
            reranker = FlagReranker(
                model_name=settings.RERANKER_MODEL,
                device=settings.RERANKER_DEVICE,
                use_fp16=settings.RERANKER_USE_FP16,
                query_max_length=settings.RERANKER_QUERY_MAX_LENGTH,
                passage_max_length=settings.RERANKER_PASSAGE_MAX_LENGTH,
            )
        case _:
            raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")

# Initialize cache factory
cache_factory = LlamaCacheFactory(Path(settings.STORAGE_PATH))

# Initialize document service with configured components
document_service = DocumentService(
    storage=storage,
    database=database,
    vector_store=vector_store,
    embedding_model=embedding_model,
    completion_model=completion_model,
    parser=parser,
    reranker=reranker,
    cache_factory=cache_factory,
)


async def verify_token(authorization: str = Header(None)) -> AuthContext:
    """Verify JWT Bearer token or return dev context if dev_mode is enabled."""
    # Check if dev mode is enabled
    if settings.dev_mode:
        return AuthContext(
            entity_type=EntityType(settings.dev_entity_type),
            entity_id=settings.dev_entity_id,
            permissions=set(settings.dev_permissions),
        )

    # Normal token verification flow
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]  # Remove "Bearer "
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

        if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
            raise HTTPException(status_code=401, detail="Token expired")

        return AuthContext(
            entity_type=EntityType(payload["type"]),
            entity_id=payload["entity_id"],
            app_id=payload.get("app_id"),
            permissions=set(payload.get("permissions", ["read"])),
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))
