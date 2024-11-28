import uuid
from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Annotated, Union
from pydantic import BaseModel, Field
import jwt
import os
from datetime import datetime, UTC
import logging

from pymongo import MongoClient
from .vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from .vector_store.contextual_vector_store import ContextualVectorStore
from .embedding_model.openai_embedding_model import OpenAIEmbeddingModel
from .embedding_model.voyageai_embedding_model import VoyageAIEmbeddingModel
from .parser.unstructured_parser import UnstructuredAPIParser
from .planner.simple_planner import SimpleRAGPlanner
from .document import DocumentChunk, Permission, Source, SystemMetadata, AuthContext, AuthType
from .utils.aws_utils import get_s3_client, upload_from_encoded_string, create_presigned_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DataBridge API",
    description="REST API for DataBridge document ingestion and querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataBridgeException(HTTPException):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)


class AuthenticationError(DataBridgeException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(detail=detail, status_code=status.HTTP_401_UNAUTHORIZED)


class ServiceConfig:
    """Service-wide configuration and component management"""
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET_KEY environment variable not set")

        # Required environment variables
        required_vars = {
            "MONGODB_URI": "MongoDB connection string",
            "OPENAI_API_KEY": "OpenAI API key",
            "UNSTRUCTURED_API_KEY": "Unstructured API key"
        }

        missing = [f"{var} ({desc})" for var, desc in required_vars.items() if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Initialize core components
        self._init_components()

    def _init_components(self):
        """Initialize service components"""
        try:
            self.database = MongoClient(os.getenv("MONGODB_URI")).get_database(os.getenv("DB_NAME", "DataBridgeTest")).get_collection(os.getenv("COLLECTION_NAME", "test"))
            
            # Create enhanced vector store
            self.vector_store = ContextualVectorStore(
                mongo_store=MongoDBAtlasVectorStore(
                    connection_string=os.getenv("MONGODB_URI"),
                    database_name=os.getenv("DB_NAME", "DataBridgeTest"),
                    collection_name=os.getenv("COLLECTION_NAME", "test")
                ),
                semantic_weight=0.8,  # Configurable weights
                bm25_weight=0.2
            )

            # self.embedding_model = OpenAIEmbeddingModel(
            #     api_key=os.getenv("OPENAI_API_KEY"),
            #     model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            # )
            self.embedding_model = VoyageAIEmbeddingModel()

            self.parser = UnstructuredAPIParser(
                api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
            )

            self.planner = SimpleRAGPlanner(
                default_k=int(os.getenv("DEFAULT_K", "4"))
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize components: {str(e)}")

    async def verify_token(self, token: str, owner_id: str) -> AuthContext:
        """Verify JWT token and return auth context"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
                raise AuthenticationError("Token has expired")

            # Check if this is a developer token
            if "." in owner_id:  # dev_id.app_id format
                dev_id, app_id = owner_id.split(".")
                return AuthContext(
                    type=AuthType.DEVELOPER,
                    dev_id=dev_id,
                    app_id=app_id
                )
            else:  # User token
                return AuthContext(
                    type=AuthType.USER,
                    eu_id=owner_id
                )

        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {str(e)}")


# Initialize service
service = ServiceConfig()


# Request/Response Models
class Document(BaseModel):
    id: str
    name: str
    type: str
    source: str
    uploaded_at: str
    size: str
    redaction_level: str
    stats: Dict[str, Union[int, str]] = Field(
        default_factory=lambda: {
            "ai_queries": 0,
            "time_saved": "0h",
            "last_accessed": ""
        }
    )
    accessed_by: List[Dict[str, str]] = Field(default_factory=list)
    sensitive_content: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    presigned_url: Optional[str] = None

    @classmethod
    def from_mongo(cls, data: Dict[str, Any]) -> "Document":
        """Create from MongoDB document"""
        # Convert MongoDB document to Document model
        s3_key = data.get("system_metadata", {}).get("s3_key")
        s3_bucket = data.get("system_metadata", {}).get("s3_bucket")
        presigned_url = create_presigned_url(get_s3_client(), s3_bucket, s3_key)
        return cls(
            id=str(data.get("_id")),
            name=data.get("system_metadata", {}).get("filename") or "Untitled",
            type="document", # Default type for now
            source=data.get("source"),
            uploaded_at=str(data.get("_id").generation_time), # MongoDB ObjectId contains timestamp
            size="N/A", # Size not stored currently
            redaction_level="none", # Default redaction level
            stats={
                "ai_queries": 0,
                "time_saved": "0h", 
                "last_accessed": ""
            },
            accessed_by=[],
            metadata=data.get("metadata", {}),
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            presigned_url=presigned_url
        )


class IngestRequest(BaseModel):
    content: str = Field(..., description="Document content (text or base64)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    eu_id: Optional[str] = Field(None, description="End user ID when developer ingests for user")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string")
    k: Optional[int] = Field(default=4, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, 
                                            description="Optional metadata filters")


class IngestResponse(BaseModel):
    document_id: str = Field(..., description="Ingested document ID")
    message: str = Field(default="Document ingested successfully")


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total number of results")


# Authentication dependency
async def verify_auth(
    owner_id: Annotated[str, Header(alias="X-Owner-ID")],
    auth_token: Annotated[str, Header(alias="X-Auth-Token")]
) -> str:
    """Verify authentication headers"""
    return await service.verify_token(auth_token, owner_id)


# Error handler middleware
@app.middleware("http")
async def error_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except DataBridgeException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.exception("Unexpected error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"}
        )
    

#TODO: This is not complete - only returns s3 urls, we need to find a way to make itwork over regular content also or make ti work for preseigned urls.
@app.get("/documents", response_model=List[Document])
async def get_documents(auth: AuthContext = Depends(verify_auth)) -> List[Document]:
    """Get all document files. Content ingested as string will have an empty presigned url field."""
    match_stage = {
        "$match": {
            "$or": [
                {"system_metadata.dev_id": auth.dev_id},  # Dev's own docs
                {"permissions": {"$in": [auth.app_id]}}  # Docs app has access to
            ]
        } if auth.type == AuthType.DEVELOPER else {"system_metadata.eu_id": auth.eu_id}
    }

    pipeline = [
        match_stage,
        {
            "$group": {
                "_id": "$system_metadata.s3_key",
                "doc": {"$first": "$$ROOT"}
            }
        },
        {
            "$replaceRoot": {"newRoot": "$doc"}
        }
    ]

    documents = list(service.database.aggregate(pipeline))
    return [Document.from_mongo(doc) for doc in documents]


# TODO: Move to the way brandsync stored embeddings and documents separately - 
# all the metadata and all info is stored in one collection, and embeddings stored 
# in another store. (seperation of concerns)
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    auth: AuthContext = Depends(verify_auth)
) -> IngestResponse:
    """Ingest a document into DataBridge."""
    logger.info(f"Ingesting document for {auth.type}")

    # Generate document ID for all chunks.
    doc_id = str(uuid.uuid4())
    s3_client = get_s3_client()
    s3_bucket, s3_key = upload_from_encoded_string(s3_client, request.content, doc_id)

    # Set up system metadata.
    system_metadata = SystemMetadata(doc_id=doc_id, s3_bucket=s3_bucket, s3_key=s3_key)
    if request.metadata.get("filename"):
        system_metadata.filename = request.metadata["filename"]
    if auth.type == AuthType.DEVELOPER:
        system_metadata.dev_id = auth.dev_id
        system_metadata.app_id = auth.app_id
        if request.eu_id:
            system_metadata.eu_id = request.eu_id
    else:
        system_metadata.eu_id = auth.eu_id

    # Parse into chunks.
    chunk_texts = service.parser.parse(request.content, request.metadata)
    embeddings = await service.embedding_model.embed_for_ingestion(chunk_texts)

    # Create chunks.
    chunks = []
    for text, embedding in zip(chunk_texts, embeddings):
        # Set source and permissions based on context.
        if auth.type == AuthType.DEVELOPER:
            source = Source.APP
            permissions = {auth.app_id: {Permission.READ, Permission.WRITE, Permission.DELETE}} if request.eu_id else {}
        else:
            source = Source.SELF_UPLOADED
            permissions = {}

        chunk = DocumentChunk(
            content=text,
            embedding=embedding,
            metadata=request.metadata,
            system_metadata=system_metadata,
            source=source,
            permissions=permissions
        )
        chunks.append(chunk)

    # Store in vector store.
    if not service.vector_store.store_embeddings(chunks):
        raise DataBridgeException(
            "Failed to store embeddings",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    return IngestResponse(document_id=doc_id)


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    auth: AuthContext = Depends(verify_auth)
) -> QueryResponse:
    """Enhanced query endpoint using contextual embeddings, BM25, and reranking."""
    logger.info(f"Processing query for owner {auth.type}")
    
    # Get query embedding
    query_embedding = await service.embedding_model.embed_for_query(request.query)
    
    # Query using enhanced retrieval
    chunks = await service.vector_store.query_similar(
        query_embedding=query_embedding,
        query_text=request.query,
        k=request.k or 4,
        auth=auth,
        filters=request.filters,
        rerank=True  # Enable reranking by default
    )
    
    results = [
        {
            "content": chunk["content"],
            "doc_id": chunk["metadata"]["doc_id"],
            "score": chunk["score"],
            "metadata": chunk["metadata"]
        }
        for chunk in chunks
    ]

    return QueryResponse(
        results=results,
        total_results=len(results)
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check service health"""
    try:
        # Verify MongoDB connection
        service.vector_store.mongo_store.collection.find_one({})
        return {"status": "healthy"}
    except Exception as e:
        raise DataBridgeException(
            f"Service unhealthy: {str(e)}", 
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Verify all connections on startup"""
    logger.info("Starting DataBridge service")
    await health_check()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DataBridge service")
