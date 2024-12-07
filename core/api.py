import json
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Union
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Depends,
    Header,
    UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
import jwt
import logging
from core.models.request import IngestTextRequest, QueryRequest
from core.models.documents import (
    Document,
    DocumentResult,
    ChunkResult,
    EntityType
)
from core.models.auth import AuthContext
from core.services.document_service import DocumentService
from core.config import get_settings
from core.database.mongo_database import MongoDatabase
from core.vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from core.storage.s3_storage import S3Storage
from core.parser.unstructured_parser import UnstructuredAPIParser
from core.embedding_model.openai_embedding_model import OpenAIEmbeddingModel


# Initialize FastAPI app
app = FastAPI(title="DataBridge API")
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize service
settings = get_settings()

# Initialize components
database = MongoDatabase(
    **settings.get_mongodb_settings()
)

vector_store = MongoDBAtlasVectorStore(
    settings.MONGODB_URI,
    settings.DATABRIDGE_DB,
    settings.CHUNKS_COLLECTION,
    settings.VECTOR_INDEX_NAME
)

storage = S3Storage(
    **settings.get_storage_settings()
)

parser = UnstructuredAPIParser(
    api_key=settings.UNSTRUCTURED_API_KEY,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)

embedding_model = OpenAIEmbeddingModel(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.EMBEDDING_MODEL
)

# Initialize document service
document_service = DocumentService(
    database=database,
    vector_store=vector_store,
    storage=storage,
    parser=parser,
    embedding_model=embedding_model
)


async def verify_token(authorization: str = Header(None)) -> AuthContext:
    """Verify JWT Bearer token."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header"
            )

        token = authorization[7:]  # Remove "Bearer "
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
            raise HTTPException(status_code=401, detail="Token expired")

        return AuthContext(
            entity_type=EntityType(payload["type"]),
            entity_id=payload["entity_id"],
            app_id=payload.get("app_id"),
            permissions=set(payload.get("permissions", ["read"]))
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/ingest/text", response_model=Document)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token)
) -> Document:
    """Ingest a text document."""
    try:
        return await document_service.ingest_text(request, auth)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/ingest/file", response_model=Document)
async def ingest_file(
    file: UploadFile,
    metadata: str = Form("{}"),  # JSON string of metadata
    auth: AuthContext = Depends(verify_token)
) -> Document:
    """Ingest a file document."""
    try:
        metadata_dict = json.loads(metadata)
        doc = await document_service.ingest_file(file, metadata_dict, auth)
        return doc  # TODO: Might be lighter on network to just send the document ID.
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid metadata JSON")


@app.post("/query", response_model=Union[List[ChunkResult], List[DocumentResult]])
async def query_documents(
    request: QueryRequest,
    auth: AuthContext = Depends(verify_token)
):
    """Query documents with specified return type."""
    return await document_service.query(request, auth)


@app.get("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None
):
    """List accessible documents."""
    return await document_service.db.get_documents(auth, skip, limit, filters)


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    auth: AuthContext = Depends(verify_token)
):
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
