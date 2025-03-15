from typing import List, Optional, Tuple, Union
import logging
import torch
import numpy as np
import psycopg
from pgvector.psycopg import Bit, register_vector
from core.models.chunk import DocumentChunk
from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class MultiVectorStore(BaseVectorStore):
    """PostgreSQL implementation for storing and querying multi-vector embeddings using psycopg."""

    def __init__(
        self,
        uri: str,
    ):
        """Initialize PostgreSQL connection for multi-vector storage.

        Args:
            uri: PostgreSQL connection URI
        """
        # Convert SQLAlchemy URI to psycopg format if needed
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://")
        self.uri = uri
        # self.conn = psycopg.connect(self.uri, autocommit=True)
        self.conn = None
        self.initialize()

    def initialize(self):
        """Initialize database tables and max_sim function."""
        try:
            # Connect to database
            self.conn = psycopg.connect(self.uri, autocommit=True)

            # Register vector extension
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self.conn)

            # First check if the table exists and if it has the required columns
            check_table = self.conn.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'multi_vector_embeddings'
                );
            """
            ).fetchone()[0]

            if check_table:
                # Check if document_id column exists
                has_document_id = self.conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'multi_vector_embeddings' AND column_name = 'document_id'
                    );
                """
                ).fetchone()[0]

                # If the table exists but doesn't have document_id, we need to add the required columns
                if not has_document_id:
                    logger.info("Updating multi_vector_embeddings table with required columns")
                    self.conn.execute(
                        """
                        ALTER TABLE multi_vector_embeddings 
                        ADD COLUMN document_id TEXT,
                        ADD COLUMN chunk_number INTEGER,
                        ADD COLUMN content TEXT,
                        ADD COLUMN chunk_metadata TEXT
                    """
                    )
                    self.conn.execute(
                        """
                        ALTER TABLE multi_vector_embeddings 
                        ALTER COLUMN document_id SET NOT NULL
                    """
                    )

                    # Add a commit to ensure changes are applied
                    self.conn.commit()
            else:
                # Create table if it doesn't exist with all required columns
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS multi_vector_embeddings (
                        id BIGSERIAL PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_number INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        chunk_metadata TEXT,
                        embeddings BIT(128)[]
                    )
                """
                )

            # Add a commit to ensure table creation is complete
            self.conn.commit()

            try:
                # Create index on document_id
                self.conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_multi_vector_document_id 
                    ON multi_vector_embeddings (document_id)
                """
                )
            except Exception as e:
                # Log index creation failure but continue
                logger.warning(f"Failed to create index: {str(e)}")

            try:
                # First, try to drop the existing function if it exists
                self.conn.execute(
                    """
                    DROP FUNCTION IF EXISTS max_sim(bit[], bit[])
                """
                )
                logger.info("Dropped existing max_sim function")

                # Create max_sim function
                self.conn.execute(
                    """
                    CREATE OR REPLACE FUNCTION max_sim(document bit[], query bit[]) RETURNS double precision AS $$
                        WITH queries AS (
                            SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query) AS foo
                        ),
                        documents AS (
                            SELECT unnest(document) AS document
                        ),
                        similarities AS (
                            SELECT 
                                query_number, 
                                1.0 - (bit_count(document # query)::float / greatest(bit_length(query), 1)::float) AS similarity
                            FROM queries CROSS JOIN documents
                        ),
                        max_similarities AS (
                            SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
                        )
                        SELECT SUM(max_similarity) FROM max_similarities
                    $$ LANGUAGE SQL
                """
                )
                logger.info("Created max_sim function successfully")
            except Exception as e:
                logger.error(f"Error creating max_sim function: {str(e)}")
                # Continue even if function creation fails - it might already exist and be usable

            logger.info("MultiVectorStore initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MultiVectorStore: {str(e)}")
            return False

    def _binary_quantize(self, embeddings: Union[np.ndarray, torch.Tensor, List]) -> List[Bit]:
        """Convert embeddings to binary format for PostgreSQL BIT[] arrays."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(embeddings, list) and not isinstance(embeddings[0], np.ndarray):
            embeddings = np.array(embeddings)
        # try:
        return [Bit(embedding > 0) for embedding in embeddings]
        # except Exception as e:
        #     logger.error(f"Error quantizing embeddings: {str(e)}")
        #     raise e

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their multi-vector embeddings."""
        # try:
        if not chunks:
            return True, []

        stored_ids = []

        for chunk in chunks:
            # Ensure embeddings exist
            if not hasattr(chunk, "embedding") or chunk.embedding is None:
                logger.error(
                    f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}"
                )
                continue

            # For multi-vector embeddings, we expect a list of vectors
            embeddings = chunk.embedding

            # Create binary representation for each vector
            binary_embeddings = self._binary_quantize(embeddings)

            # Insert into database
            self.conn.execute(
                """
                INSERT INTO multi_vector_embeddings 
                (document_id, chunk_number, content, chunk_metadata, embeddings) 
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    chunk.document_id,
                    chunk.chunk_number,
                    chunk.content,
                    str(chunk.metadata),
                    binary_embeddings,
                ),
            )

            stored_ids.append(f"{chunk.document_id}-{chunk.chunk_number}")

        logger.info(f"{len(stored_ids)} vector embeddings added successfully!")
        return len(stored_ids) > 0, stored_ids

        # except Exception as e:
        #     logger.error(f"Error storing multi-vector embeddings: {str(e)}")
        #     raise e
        #     return False, []

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using the max_sim function for multi-vectors."""
        # try:
        # Convert query embeddings to binary format
        binary_query_embeddings = self._binary_quantize(query_embedding)

        # Build query
        query = """
            SELECT id, document_id, chunk_number, content, chunk_metadata, 
                    max_sim(embeddings, %s) AS similarity
            FROM multi_vector_embeddings
        """

        params = [binary_query_embeddings]

        # Add document filter if needed
        if doc_ids:
            doc_ids_str = "', '".join(doc_ids)
            query += f" WHERE document_id IN ('{doc_ids_str}')"

        # Add ordering and limit
        query += " ORDER BY similarity DESC LIMIT %s"
        params.append(k)

        # Execute query
        result = self.conn.execute(query, params).fetchall()

        # Convert to DocumentChunks
        chunks = []
        for row in result:
            try:
                metadata = eval(row[4]) if row[4] else {}
            except (ValueError, SyntaxError):
                metadata = {}

            chunk = DocumentChunk(
                document_id=row[1],
                chunk_number=row[2],
                content=row[3],
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=float(row[5]),  # Use the similarity score from max_sim
            )
            chunks.append(chunk)

        return chunks

        # except Exception as e:
        #     logger.error(f"Error querying similar chunks: {str(e)}")
        #     raise e
        #     return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
    ) -> List[DocumentChunk]:
        """
        Retrieve specific chunks by document ID and chunk number in a single database query.
        
        Args:
            chunk_identifiers: List of (document_id, chunk_number) tuples
            
        Returns:
            List of DocumentChunk objects
        """
        # try:
        if not chunk_identifiers:
            return []
            
        # Construct the WHERE clause with OR conditions
        conditions = []
        for doc_id, chunk_num in chunk_identifiers:
            conditions.append(f"(document_id = '{doc_id}' AND chunk_number = {chunk_num})")
        
        where_clause = " OR ".join(conditions)
        
        # Build and execute query
        query = f"""
            SELECT document_id, chunk_number, content, chunk_metadata
            FROM multi_vector_embeddings
            WHERE {where_clause}
        """
        
        logger.info(f"Batch retrieving {len(chunk_identifiers)} chunks from multi-vector store")
        
        result = self.conn.execute(query).fetchall()
        
        # Convert to DocumentChunks
        chunks = []
        for row in result:
            try:
                metadata = eval(row[3]) if row[3] else {}
            except (ValueError, SyntaxError):
                metadata = {}
                
            chunk = DocumentChunk(
                document_id=row[0],
                chunk_number=row[1],
                content=row[2],
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=0.0,  # No relevance score for direct retrieval
            )
            chunks.append(chunk)
            
        logger.info(f"Found {len(chunks)} chunks in batch retrieval from multi-vector store")
        return chunks
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
