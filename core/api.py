import json
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Depends, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import jwt
import logging
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from core.completion.openai_completion import OpenAICompletionModel
from core.embedding.ollama_embedding_model import OllamaEmbeddingModel
from core.models.request import (
    IngestTextRequest,
    RetrieveRequest,
    CompletionQueryRequest,
)
from core.models.documents import Document, DocumentResult, ChunkResult
from core.models.auth import AuthContext, EntityType
from core.parser.combined_parser import CombinedParser
from core.completion.base_completion import CompletionResponse
from core.parser.unstructured_parser import UnstructuredAPIParser
from core.services.document_service import DocumentService
from core.services.telemetry import TelemetryService
from core.config import get_settings
from core.database.mongo_database import MongoDatabase
from core.vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from core.storage.s3_storage import S3Storage
from core.embedding.openai_embedding_model import OpenAIEmbeddingModel
from core.completion.ollama_completion import OllamaCompletionModel

# Initialize FastAPI app
app = FastAPI(title="DataBridge API")
logger = logging.getLogger(__name__)

# Initialize telemetry
telemetry = TelemetryService()

# Add OpenTelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
settings = get_settings()

# Initialize database
match settings.DATABASE_PROVIDER:
    case "mongodb":
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
    case _:
        raise ValueError(f"Unsupported vector store provider: {settings.VECTOR_STORE_PROVIDER}")

# Initialize storage
match settings.STORAGE_PROVIDER:
    case "aws-s3":
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
        parser = CombinedParser(
            unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
            assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            frame_sample_rate=settings.FRAME_SAMPLE_RATE,
        )
    case "unstructured":
        parser = UnstructuredAPIParser(
            unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    case _:
        raise ValueError(f"Unsupported parser provider: {settings.PARSER_PROVIDER}")

# Initialize embedding model
match settings.EMBEDDING_PROVIDER:
    case "openai":
        embedding_model = OpenAIEmbeddingModel(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )
    case "ollama":
        embedding_model = OllamaEmbeddingModel(
            model_name=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    case _:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")

# Initialize completion model
match settings.COMPLETION_PROVIDER:
    case "ollama":
        completion_model = OllamaCompletionModel(
            model_name=settings.COMPLETION_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    case "openai":
        completion_model = OpenAICompletionModel(
            model_name=settings.COMPLETION_MODEL,
        )
    case _:
        raise ValueError(f"Unsupported completion provider: {settings.COMPLETION_PROVIDER}")

# Initialize document service with configured components
document_service = DocumentService(
    database=database,
    vector_store=vector_store,
    storage=storage,
    parser=parser,
    embedding_model=embedding_model,
    completion_model=completion_model,
)


async def verify_token(authorization: str = Header(None)) -> AuthContext:
    """Verify JWT Bearer token."""
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


@app.post("/ingest/text", response_model=Document)
async def ingest_text(
    request: IngestTextRequest, auth: AuthContext = Depends(verify_token)
) -> Document:
    """Ingest a text document."""
    try:
        async with telemetry.track_operation(
            operation_type="ingest_text",
            user_id=auth.entity_id,
            tokens_used=len(request.text.split()),  # Approximate token count
            metadata=request.metadata.model_dump() if request.metadata else None,
        ):
            return await document_service.ingest_text(request, auth)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/ingest/file", response_model=Document)
async def ingest_file(
    file: UploadFile,
    metadata: str = Form("{}"),
    auth: AuthContext = Depends(verify_token),
) -> Document:
    """Ingest a file document."""
    try:
        metadata_dict = json.loads(metadata)
        async with telemetry.track_operation(
            operation_type="ingest_file",
            user_id=auth.entity_id,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "metadata": metadata_dict,
            },
        ):
            doc = await document_service.ingest_file(file, metadata_dict, auth)
            return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid metadata JSON")


@app.post("/retrieve/chunks", response_model=List[ChunkResult])
async def retrieve_chunks(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """Retrieve relevant chunks."""
    async with telemetry.track_operation(
        operation_type="retrieve_chunks",
        user_id=auth.entity_id,
        metadata=request.model_dump(),
    ):
        return await document_service.retrieve_chunks(
            request.query, auth, request.filters, request.k, request.min_score
        )


@app.post("/retrieve/docs", response_model=List[DocumentResult])
async def retrieve_documents(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """Retrieve relevant documents."""
    async with telemetry.track_operation(
        operation_type="retrieve_docs",
        user_id=auth.entity_id,
        metadata=request.model_dump(),
    ):
        return await document_service.retrieve_docs(
            request.query, auth, request.filters, request.k, request.min_score
        )


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest, auth: AuthContext = Depends(verify_token)
):
    """Generate completion using relevant chunks as context."""
    async with telemetry.track_operation(
        operation_type="query",
        user_id=auth.entity_id,
        metadata=request.model_dump(),
    ) as span:
        response = await document_service.query(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.max_tokens,
            request.temperature,
        )
        if hasattr(response, "usage"):
            span.set_attribute("tokens.completion", response.usage.completion_tokens)
            span.set_attribute("tokens.prompt", response.usage.prompt_tokens)
            span.set_attribute("tokens.total", response.usage.total_tokens)
        return response


@app.get("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None,
):
    """List accessible documents."""
    return await document_service.db.get_documents(auth, skip, limit, filters)


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """Get document by ID."""
    try:
        doc = await document_service.db.get_document(document_id, auth)
        logger.info(f"Found document: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document: {e}")
        raise e


# Usage tracking endpoints
@app.get("/usage/stats")
async def get_usage_stats(auth: AuthContext = Depends(verify_token)) -> Dict[str, int]:
    """Get usage statistics for the authenticated user."""
    async with telemetry.track_operation(operation_type="get_usage_stats", user_id=auth.entity_id):
        if not auth.permissions or "admin" not in auth.permissions:
            return telemetry.get_user_usage(auth.entity_id)
        return telemetry.get_user_usage(auth.entity_id)


@app.get("/usage/recent")
async def get_recent_usage(
    auth: AuthContext = Depends(verify_token),
    operation_type: Optional[str] = None,
    since: Optional[datetime] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """Get recent usage records."""
    async with telemetry.track_operation(
        operation_type="get_recent_usage",
        user_id=auth.entity_id,
        metadata={
            "operation_type": operation_type,
            "since": since.isoformat() if since else None,
            "status": status,
        },
    ):
        if not auth.permissions or "admin" not in auth.permissions:
            records = telemetry.get_recent_usage(
                user_id=auth.entity_id, operation_type=operation_type, since=since, status=status
            )
        else:
            records = telemetry.get_recent_usage(
                operation_type=operation_type, since=since, status=status
            )

        return [
            {
                "timestamp": record.timestamp,
                "operation_type": record.operation_type,
                "tokens_used": record.tokens_used,
                "user_id": record.user_id,
                "duration_ms": record.duration_ms,
                "status": record.status,
                "metadata": record.metadata,
            }
            for record in records
        ]
