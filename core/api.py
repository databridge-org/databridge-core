import json
from datetime import datetime, UTC, timedelta
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Depends, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import jwt
import logging
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import tomli
from core.openai_api import router as openai_router
from core.shared import verify_token, document_service, settings
from core.services.telemetry import TelemetryService
from core.models.documents import Document, DocumentResult, ChunkResult
from core.models.request import (
    IngestTextRequest,
    RetrieveRequest,
    CompletionQueryRequest,
)
from core.models.auth import AuthContext
from core.completion.base_completion import CompletionResponse

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

# Include the OpenAI router
app.include_router(openai_router)


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
            "parser": settings.PARSER_PROVIDER,
        },
    }


@app.post("/ingest/text", response_model=Document)
async def ingest_text(
    request: IngestTextRequest, auth: AuthContext = Depends(verify_token)
) -> Document:
    """Ingest a text document."""
    try:
        async with telemetry.track_operation(
            operation_type="ingest_text",
            user_id=auth.entity_id,
            tokens_used=len(request.content.split()),  # Approximate token count
            metadata=request.metadata if request.metadata else None,
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
    try:
        async with telemetry.track_operation(
            operation_type="retrieve_chunks",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "use_reranking": request.use_reranking,
            },
        ):
            return await document_service.retrieve_chunks(
                request.query,
                auth,
                request.filters,
                request.k,
                request.min_score,
                request.use_reranking,
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
            },
        ):
            return await document_service.retrieve_docs(
                request.query,
                auth,
                request.filters,
                request.k,
                request.min_score,
                request.use_reranking,
            )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest, auth: AuthContext = Depends(verify_token)
):
    """Generate completion using relevant chunks as context."""
    try:
        async with telemetry.track_operation(
            operation_type="query",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "use_reranking": request.use_reranking,
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
