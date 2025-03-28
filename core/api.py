import asyncio
import json
from datetime import datetime, UTC, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import jwt
import logging
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from core.completion.openai_completion import OpenAICompletionModel
from core.embedding.ollama_embedding_model import OllamaEmbeddingModel
from core.limits_utils import check_and_increment_limits
from core.models.request import GenerateUriRequest, RetrieveRequest, CompletionQueryRequest, IngestTextRequest, CreateGraphRequest, BatchIngestResponse
from core.models.completion import ChunkSource, CompletionResponse
from core.models.documents import Document, DocumentResult, ChunkResult
from core.models.graph import Graph
from core.models.auth import AuthContext, EntityType
from core.parser.databridge_parser import DatabridgeParser
from core.services.document_service import DocumentService
from core.services.telemetry import TelemetryService
from core.config import get_settings
from core.database.mongo_database import MongoDatabase
from core.database.postgres_database import PostgresDatabase
from core.vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from core.vector_store.multi_vector_store import MultiVectorStore
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.storage.s3_storage import S3Storage
from core.storage.local_storage import LocalStorage
from core.embedding.openai_embedding_model import OpenAIEmbeddingModel
from core.completion.ollama_completion import OllamaCompletionModel
from core.reranker.flag_reranker import FlagReranker
from core.cache.llama_cache_factory import LlamaCacheFactory
import tomli

# Initialize FastAPI app
app = FastAPI(title="DataBridge API")
logger = logging.getLogger(__name__)


# Add health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check that verifies the application is initialized."""
    return {
        "status": "ready",
        "components": {
            "database": settings.DATABASE_PROVIDER,
            "vector_store": settings.VECTOR_STORE_PROVIDER,
            "embedding": settings.EMBEDDING_PROVIDER,
            "completion": settings.COMPLETION_PROVIDER,
            "storage": settings.STORAGE_PROVIDER,
        },
    }


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


@app.on_event("startup")
async def initialize_database():
    """Initialize database tables and indexes on application startup."""
    logger.info("Initializing database...")
    success = await database.initialize()
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")
        # We don't raise an exception here to allow the app to continue starting
        # even if there are initialization errors

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
parser = DatabridgeParser(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    use_unstructured_api=settings.USE_UNSTRUCTURED_API,
    unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
    assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
    anthropic_api_key=settings.ANTHROPIC_API_KEY,
    use_contextual_chunking=settings.USE_CONTEXTUAL_CHUNKING,
)

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

# Initialize ColPali embedding model if enabled
colpali_embedding_model = ColpaliEmbeddingModel() if settings.ENABLE_COLPALI else None
colpali_vector_store = (
    MultiVectorStore(uri=settings.POSTGRES_URI) if settings.ENABLE_COLPALI else None
)

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
    enable_colpali=settings.ENABLE_COLPALI,
    colpali_embedding_model=colpali_embedding_model,
    colpali_vector_store=colpali_vector_store,
)


async def verify_token(authorization: str = Header(None)) -> AuthContext:
    """Verify JWT Bearer token or return dev context if dev_mode is enabled."""
    # Check if dev mode is enabled
    if settings.dev_mode:
        return AuthContext(
            entity_type=EntityType(settings.dev_entity_type),
            entity_id=settings.dev_entity_id,
            permissions=set(settings.dev_permissions),
            user_id=settings.dev_entity_id,  # In dev mode, entity_id is also the user_id
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

        # Support both "type" and "entity_type" fields for compatibility
        entity_type_field = payload.get("type") or payload.get("entity_type")
        if not entity_type_field:
            raise HTTPException(status_code=401, detail="Missing entity type in token")
            
        return AuthContext(
            entity_type=EntityType(entity_type_field),
            entity_id=payload["entity_id"],
            app_id=payload.get("app_id"),
            permissions=set(payload.get("permissions", ["read"])),
            user_id=payload.get("user_id", payload["entity_id"]),  # Use user_id if available, fallback to entity_id
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/ingest/text", response_model=Document)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token),
) -> Document:
    """
    Ingest a text document.

    Args:
        request: IngestTextRequest containing:
            - content: Text content to ingest
            - filename: Optional filename to help determine content type
            - metadata: Optional metadata dictionary
            - rules: Optional list of rules. Each rule should be either:
                   - MetadataExtractionRule: {"type": "metadata_extraction", "schema": {...}}
                   - NaturalLanguageRule: {"type": "natural_language", "prompt": "..."}
        auth: Authentication context

    Returns:
        Document: Metadata of ingested document
    """
    try:
        async with telemetry.track_operation(
            operation_type="ingest_text",
            user_id=auth.entity_id,
            tokens_used=len(request.content.split()),  # Approximate token count
            metadata={
                "metadata": request.metadata,
                "rules": request.rules,
                "use_colpali": request.use_colpali,
            },
        ):
            return await document_service.ingest_text(
                content=request.content,
                filename=request.filename,
                metadata=request.metadata,
                rules=request.rules,
                use_colpali=request.use_colpali,
                auth=auth,
            )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/ingest/file", response_model=Document)
async def ingest_file(
    file: UploadFile,
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    auth: AuthContext = Depends(verify_token),
    use_colpali: Optional[bool] = None,
) -> Document:
    """
    Ingest a file document.

    Args:
        file: File to ingest
        metadata: JSON string of metadata
        rules: JSON string of rules list. Each rule should be either:
               - MetadataExtractionRule: {"type": "metadata_extraction", "schema": {...}}
               - NaturalLanguageRule: {"type": "natural_language", "prompt": "..."}
        auth: Authentication context

    Returns:
        Document: Metadata of ingested document
    """
    try:
        metadata_dict = json.loads(metadata)
        rules_list = json.loads(rules)
        use_colpali = bool(use_colpali)

        async with telemetry.track_operation(
            operation_type="ingest_file",
            user_id=auth.entity_id,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "metadata": metadata_dict,
                "rules": rules_list,
                "use_colpali": use_colpali,
            },
        ):
            logger.debug(f"API: Ingesting file with use_colpali: {use_colpali}")
            return await document_service.ingest_file(
                file=file,
                metadata=metadata_dict,
                auth=auth,
                rules=rules_list,
                use_colpali=use_colpali,
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/ingest/files", response_model=BatchIngestResponse)
async def batch_ingest_files(
    files: List[UploadFile] = File(...),
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    use_colpali: Optional[bool] = Form(None),
    parallel: bool = Form(True),
    auth: AuthContext = Depends(verify_token),
) -> BatchIngestResponse:
    """
    Batch ingest multiple files.
    
    Args:
        files: List of files to ingest
        metadata: JSON string of metadata (either a single dict or list of dicts)
        rules: JSON string of rules list. Can be either:
               - A single list of rules to apply to all files
               - A list of rule lists, one per file
        use_colpali: Whether to use ColPali-style embedding
        parallel: Whether to process files in parallel
        auth: Authentication context

    Returns:
        BatchIngestResponse containing:
            - documents: List of successfully ingested documents
            - errors: List of errors encountered during ingestion
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided for batch ingestion"
        )

    try:
        metadata_value = json.loads(metadata)
        rules_list = json.loads(rules)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Validate metadata if it's a list
    if isinstance(metadata_value, list) and len(metadata_value) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Number of metadata items ({len(metadata_value)}) must match number of files ({len(files)})"
        )

    # Validate rules if it's a list of lists
    if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list):
        if len(rules_list) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of rule lists ({len(rules_list)}) must match number of files ({len(files)})"
            )

    documents = []
    errors = []

    async with telemetry.track_operation(
        operation_type="batch_ingest",
        user_id=auth.entity_id,
        metadata={
            "file_count": len(files),
            "metadata_type": "list" if isinstance(metadata_value, list) else "single",
            "rules_type": "per_file" if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list) else "shared",
        },
    ):
        if parallel:
            tasks = []
            for i, file in enumerate(files):
                metadata_item = metadata_value[i] if isinstance(metadata_value, list) else metadata_value
                file_rules = rules_list[i] if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list) else rules_list
                task = document_service.ingest_file(
                    file=file,
                    metadata=metadata_item,
                    auth=auth,
                    rules=file_rules,
                    use_colpali=use_colpali
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append({
                        "filename": files[i].filename,
                        "error": str(result)
                    })
                else:
                    documents.append(result)
        else:
            for i, file in enumerate(files):
                try:
                    metadata_item = metadata_value[i] if isinstance(metadata_value, list) else metadata_value
                    file_rules = rules_list[i] if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list) else rules_list
                    doc = await document_service.ingest_file(
                        file=file,
                        metadata=metadata_item,
                        auth=auth,
                        rules=file_rules,
                        use_colpali=use_colpali
                    )
                    documents.append(doc)
                except Exception as e:
                    errors.append({
                        "filename": file.filename,
                        "error": str(e)
                    })

    return BatchIngestResponse(documents=documents, errors=errors)


@app.post("/retrieve/chunks", response_model=List[ChunkResult])
async def retrieve_chunks(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """Retrieve relevant chunks."""
    try:
        async with telemetry.track_operation(
            operation_type="retrieve_chunks",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "use_reranking": request.use_reranking,
                "use_colpali": request.use_colpali,
            },
        ):
            return await document_service.retrieve_chunks(
                request.query,
                auth,
                request.filters,
                request.k,
                request.min_score,
                request.use_reranking,
                request.use_colpali,
            )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/retrieve/docs", response_model=List[DocumentResult])
async def retrieve_documents(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """Retrieve relevant documents."""
    try:
        async with telemetry.track_operation(
            operation_type="retrieve_docs",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "use_reranking": request.use_reranking,
                "use_colpali": request.use_colpali,
            },
        ):
            return await document_service.retrieve_docs(
                request.query,
                auth,
                request.filters,
                request.k,
                request.min_score,
                request.use_reranking,
                request.use_colpali,
            )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/documents", response_model=List[Document])
async def batch_get_documents(document_ids: List[str], auth: AuthContext = Depends(verify_token)):
    """Retrieve multiple documents by their IDs in a single batch operation."""
    try:
        async with telemetry.track_operation(
            operation_type="batch_get_documents",
            user_id=auth.entity_id,
            metadata={
                "document_count": len(document_ids),
            },
        ):
            return await document_service.batch_retrieve_documents(document_ids, auth)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/chunks", response_model=List[ChunkResult])
async def batch_get_chunks(chunk_ids: List[ChunkSource], auth: AuthContext = Depends(verify_token)):
    """Retrieve specific chunks by their document ID and chunk number in a single batch operation."""
    try:
        async with telemetry.track_operation(
            operation_type="batch_get_chunks",
            user_id=auth.entity_id,
            metadata={
                "chunk_count": len(chunk_ids),
            },
        ):
            return await document_service.batch_retrieve_chunks(chunk_ids, auth)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest, auth: AuthContext = Depends(verify_token)
):
    """Generate completion using relevant chunks as context.
    
    When graph_name is provided, the query will leverage the knowledge graph 
    to enhance retrieval by finding relevant entities and their connected documents.
    """
    try:
        # Check query limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "query", 1)
            
        async with telemetry.track_operation(
            operation_type="query",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "use_reranking": request.use_reranking,
                "use_colpali": request.use_colpali,
                "graph_name": request.graph_name,
                "hop_depth": request.hop_depth,
                "include_paths": request.include_paths,
            },
        ):
            return await document_service.query(
                request.query,
                auth,
                request.filters,
                request.k,
                request.min_score,
                request.max_tokens,
                request.temperature,
                request.use_reranking,
                request.use_colpali,
                request.graph_name,
                request.hop_depth,
                request.include_paths,
            )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 10000,
    filters: Optional[Dict[str, Any]] = None,
):
    """List accessible documents."""
    return await document_service.db.get_documents(auth, skip, limit, filters)


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """Get document by ID."""
    try:
        doc = await document_service.db.get_document(document_id, auth)
        logger.debug(f"Found document: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document: {e}")
        raise e


@app.get("/documents/filename/{filename}", response_model=Document)
async def get_document_by_filename(filename: str, auth: AuthContext = Depends(verify_token)):
    """Get document by filename."""
    try:
        doc = await document_service.db.get_document_by_filename(filename, auth)
        logger.debug(f"Found document by filename: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document by filename: {e}")
        raise e


@app.post("/documents/{document_id}/update_text", response_model=Document)
async def update_document_text(
    document_id: str,
    request: IngestTextRequest,
    update_strategy: str = "add",
    auth: AuthContext = Depends(verify_token)
):
    """
    Update a document with new text content using the specified strategy.
    
    Args:
        document_id: ID of the document to update
        request: Text content and metadata for the update
        update_strategy: Strategy for updating the document (default: 'add')
        
    Returns:
        Document: Updated document metadata
    """
    try:
        async with telemetry.track_operation(
            operation_type="update_document_text",
            user_id=auth.entity_id,
            metadata={
                "document_id": document_id,
                "update_strategy": update_strategy,
                "use_colpali": request.use_colpali,
                "has_filename": request.filename is not None,
            },
        ):
            doc = await document_service.update_document(
                document_id=document_id,
                auth=auth,
                content=request.content,
                file=None,
                filename=request.filename,
                metadata=request.metadata,
                rules=request.rules,
                update_strategy=update_strategy,
                use_colpali=request.use_colpali,
            )

            if not doc:
                raise HTTPException(status_code=404, detail="Document not found or update failed")

            return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/documents/{document_id}/update_file", response_model=Document)
async def update_document_file(
    document_id: str,
    file: UploadFile,
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    update_strategy: str = Form("add"),
    use_colpali: Optional[bool] = None,
    auth: AuthContext = Depends(verify_token)
):
    """
    Update a document with content from a file using the specified strategy.

    Args:
        document_id: ID of the document to update
        file: File to add to the document
        metadata: JSON string of metadata to merge with existing metadata
        rules: JSON string of rules to apply to the content
        update_strategy: Strategy for updating the document (default: 'add')
        use_colpali: Whether to use multi-vector embedding

    Returns:
        Document: Updated document metadata
    """
    try:
        metadata_dict = json.loads(metadata)
        rules_list = json.loads(rules)

        async with telemetry.track_operation(
            operation_type="update_document_file",
            user_id=auth.entity_id,
            metadata={
                "document_id": document_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "update_strategy": update_strategy,
                "use_colpali": use_colpali,
            },
        ):
            doc = await document_service.update_document(
                document_id=document_id,
                auth=auth,
                content=None,
                file=file,
                filename=file.filename,
                metadata=metadata_dict,
                rules=rules_list,
                update_strategy=update_strategy,
                use_colpali=use_colpali,
            )

            if not doc:
                raise HTTPException(status_code=404, detail="Document not found or update failed")

            return doc
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/documents/{document_id}/update_metadata", response_model=Document)
async def update_document_metadata(
    document_id: str,
    metadata: Dict[str, Any],
    auth: AuthContext = Depends(verify_token)
):
    """
    Update only a document's metadata.

    Args:
        document_id: ID of the document to update
        metadata: New metadata to merge with existing metadata

    Returns:
        Document: Updated document metadata
    """
    try:
        async with telemetry.track_operation(
            operation_type="update_document_metadata",
            user_id=auth.entity_id,
            metadata={
                "document_id": document_id,
            },
        ):
            doc = await document_service.update_document(
                document_id=document_id,
                auth=auth,
                content=None,
                file=None,
                filename=None,
                metadata=metadata,
                rules=[],
                update_strategy="add",
                use_colpali=None,
            )

            if not doc:
                raise HTTPException(status_code=404, detail="Document not found or update failed")

            return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


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


# Cache endpoints
@app.post("/cache/create")
async def create_cache(
    name: str,
    model: str,
    gguf_file: str,
    filters: Optional[Dict[str, Any]] = None,
    docs: Optional[List[str]] = None,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Create a new cache with specified configuration."""
    try:
        # Check cache creation limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "cache", 1)
            
        async with telemetry.track_operation(
            operation_type="create_cache",
            user_id=auth.entity_id,
            metadata={
                "name": name,
                "model": model,
                "gguf_file": gguf_file,
                "filters": filters,
                "docs": docs,
            },
        ):
            filter_docs = set(await document_service.db.get_documents(auth, filters=filters))
            additional_docs = (
                {
                    await document_service.db.get_document(document_id=doc_id, auth=auth)
                    for doc_id in docs
                }
                if docs
                else set()
            )
            docs_to_add = list(filter_docs.union(additional_docs))
            if not docs_to_add:
                raise HTTPException(status_code=400, detail="No documents to add to cache")
            response = await document_service.create_cache(
                name, model, gguf_file, docs_to_add, filters
            )
            return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/cache/{name}")
async def get_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, Any]:
    """Get cache configuration by name."""
    try:
        async with telemetry.track_operation(
            operation_type="get_cache",
            user_id=auth.entity_id,
            metadata={"name": name},
        ):
            exists = await document_service.load_cache(name)
            return {"exists": exists}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/update")
async def update_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, bool]:
    """Update cache with new documents matching its filter."""
    try:
        async with telemetry.track_operation(
            operation_type="update_cache",
            user_id=auth.entity_id,
            metadata={"name": name},
        ):
            if name not in document_service.active_caches:
                exists = await document_service.load_cache(name)
                if not exists:
                    raise HTTPException(status_code=404, detail=f"Cache '{name}' not found")
            cache = document_service.active_caches[name]
            docs = await document_service.db.get_documents(auth, filters=cache.filters)
            docs_to_add = [doc for doc in docs if doc.id not in cache.docs]
            return cache.add_docs(docs_to_add)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/add_docs")
async def add_docs_to_cache(
    name: str, docs: List[str], auth: AuthContext = Depends(verify_token)
) -> Dict[str, bool]:
    """Add specific documents to the cache."""
    try:
        async with telemetry.track_operation(
            operation_type="add_docs_to_cache",
            user_id=auth.entity_id,
            metadata={"name": name, "docs": docs},
        ):
            cache = document_service.active_caches[name]
            docs_to_add = [
                await document_service.db.get_document(doc_id, auth)
                for doc_id in docs
                if doc_id not in cache.docs
            ]
            return cache.add_docs(docs_to_add)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/query")
async def query_cache(
    name: str,
    query: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    auth: AuthContext = Depends(verify_token),
) -> CompletionResponse:
    """Query the cache with a prompt."""
    try:
        # Check cache query limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "cache_query", 1)
            
        async with telemetry.track_operation(
            operation_type="query_cache",
            user_id=auth.entity_id,
            metadata={
                "name": name,
                "query": query,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        ):
            cache = document_service.active_caches[name]
            print(f"Cache state: {cache.state.n_tokens}", file=sys.stderr)
            return cache.query(query)  # , max_tokens, temperature)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/graph/create", response_model=Graph)
async def create_graph(
    request: CreateGraphRequest,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """
    Create a graph from documents.

    This endpoint extracts entities and relationships from documents
    matching the specified filters or document IDs and creates a graph.

    Args:
        request: CreateGraphRequest containing:
            - name: Name of the graph to create
            - filters: Optional metadata filters to determine which documents to include
            - documents: Optional list of specific document IDs to include
        auth: Authentication context

    Returns:
        Graph: The created graph object
    """
    try:
        # Check graph creation limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "graph", 1)
            
        async with telemetry.track_operation(
            operation_type="create_graph",
            user_id=auth.entity_id,
            metadata={
                "name": request.name,
                "filters": request.filters,
                "documents": request.documents,
            },
        ):
            return await document_service.create_graph(
                name=request.name,
                auth=auth,
                filters=request.filters,
                documents=request.documents,
            )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/graph/{name}", response_model=Graph)
async def get_graph(
    name: str,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """
    Get a graph by name.

    This endpoint retrieves a graph by its name if the user has access to it.

    Args:
        name: Name of the graph to retrieve
        auth: Authentication context

    Returns:
        Graph: The requested graph object
    """
    try:
        async with telemetry.track_operation(
            operation_type="get_graph",
            user_id=auth.entity_id,
            metadata={"name": name},
        ):
            graph = await document_service.db.get_graph(name, auth)
            if not graph:
                raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
            return graph
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graphs", response_model=List[Graph])
async def list_graphs(
    auth: AuthContext = Depends(verify_token),
) -> List[Graph]:
    """
    List all graphs the user has access to.

    This endpoint retrieves all graphs the user has access to.

    Args:
        auth: Authentication context

    Returns:
        List[Graph]: List of graph objects
    """
    try:
        async with telemetry.track_operation(
            operation_type="list_graphs",
            user_id=auth.entity_id,
        ):
            return await document_service.db.list_graphs(auth)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/local/generate_uri", include_in_schema=True)
async def generate_local_uri(
    name: str = Form("admin"),
    expiry_days: int = Form(30),
) -> Dict[str, str]:
    """Generate a local URI for development. This endpoint is unprotected."""
    try:
        # Clean name
        name = name.replace(" ", "_").lower()

        # Create payload
        payload = {
            "type": "developer",
            "entity_id": name,
            "permissions": ["read", "write", "admin"],
            "exp": datetime.now(UTC) + timedelta(days=expiry_days),
        }

        # Generate token
        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

        # Read config for host/port
        with open("databridge.toml", "rb") as f:
            config = tomli.load(f)
        base_url = f"{config['api']['host']}:{config['api']['port']}".replace(
            "localhost", "127.0.0.1"
        )

        # Generate URI
        uri = f"databridge://{name}:{token}@{base_url}"
        return {"uri": uri}
    except Exception as e:
        logger.error(f"Error generating local URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cloud/generate_uri", include_in_schema=True)
async def generate_cloud_uri(
    request: GenerateUriRequest,
    authorization: str = Header(None),
) -> Dict[str, str]:
    """Generate a URI for cloud hosted applications."""
    try:
        app_id = request.app_id
        name = request.name
        user_id = request.user_id
        expiry_days = request.expiry_days

        logger.debug(f"Generating cloud URI for app_id={app_id}, name={name}, user_id={user_id}")

        # Verify authorization header before proceeding
        if not authorization:
            logger.warning("Missing authorization header")
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify the token is valid
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]  # Remove "Bearer "

        try:
            # Decode the token to ensure it's valid
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

            # Only allow users to create apps for themselves (or admin)
            token_user_id = payload.get("user_id")
            logger.debug(f"Token user ID: {token_user_id}")
            logger.debug(f"User ID: {user_id}")
            if not (token_user_id == user_id or "admin" in payload.get("permissions", [])):
                raise HTTPException(
                    status_code=403,
                    detail="You can only create apps for your own account unless you have admin permissions"
                )
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=str(e))

        # Import UserService here to avoid circular imports
        from core.services.user_service import UserService
        user_service = UserService()

        # Initialize user service if needed
        await user_service.initialize()

        # Clean name
        name = name.replace(" ", "_").lower()

        # Check if the user is within app limit and generate URI
        uri = await user_service.generate_cloud_uri(user_id, app_id, name, expiry_days)

        if not uri:
            logger.debug("Application limit reached for this account tier with user_id: %s", user_id)
            raise HTTPException(
                status_code=403,
                detail="Application limit reached for this account tier"
            )

        return {"uri": uri, "app_id": app_id}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating cloud URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/upgrade", include_in_schema=True)
async def upgrade_user_tier(
    user_id: str,
    tier: str,
    custom_limits: Optional[Dict[str, Any]] = None,
    authorization: str = Header(None),
) -> Dict[str, Any]:
    """Upgrade a user to a higher tier."""
    try:
        # Verify admin authorization
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
            
        token = authorization[7:]  # Remove "Bearer "
        
        try:
            # Decode token
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            
            # Only allow admins to upgrade users
            if "admin" not in payload.get("permissions", []):
                raise HTTPException(
                    status_code=403,
                    detail="Admin permission required"
                )
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=str(e))
        
        # Validate tier
        from core.models.tiers import AccountTier
        try:
            account_tier = AccountTier(tier)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}")
        
        # Upgrade user
        from core.services.user_service import UserService
        user_service = UserService()
        
        # Initialize user service
        await user_service.initialize()
        
        # Update user tier
        success = await user_service.update_user_tier(user_id, tier, custom_limits)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="User not found or upgrade failed"
            )
        
        return {
            "user_id": user_id,
            "tier": tier,
            "message": f"User upgraded to {tier} tier"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error upgrading user tier: {e}")
        raise HTTPException(status_code=500, detail=str(e))
