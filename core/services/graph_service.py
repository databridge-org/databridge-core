import httpx
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel

from core.models.completion import ChunkSource, CompletionResponse, CompletionRequest
from core.models.graph import Graph, Entity, Relationship
from core.models.auth import AuthContext
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.completion.base_completion import BaseCompletionModel
from core.database.base_database import BaseDatabase
from core.models.documents import Document, ChunkResult
from core.config import get_settings
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EntityExtraction(BaseModel):
    """Model for entity extraction results"""
    label: str
    type: str
    properties: Dict[str, Any] = {}


class RelationshipExtraction(BaseModel):
    """Model for relationship extraction results"""
    source: str
    target: str
    relationship: str


class ExtractionResult(BaseModel):
    """Model for structured extraction from LLM"""
    entities: List[EntityExtraction] = []
    relationships: List[RelationshipExtraction] = []


class GraphService:
    """Service for managing knowledge graphs and graph-based operations"""

    def __init__(
        self,
        db: BaseDatabase,
        embedding_model: BaseEmbeddingModel,
        completion_model: BaseCompletionModel,
    ):
        self.db = db
        self.embedding_model = embedding_model
        self.completion_model = completion_model

    async def create_graph(
        self,
        name: str,
        auth: AuthContext,
        document_service,  # Passed in to avoid circular import
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
    ) -> Graph:
        """Create a graph from documents.

        This function processes documents matching filters or specific document IDs,
        extracts entities and relationships from document chunks, and saves them as a graph.

        Args:
            name: Name of the graph to create
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents and chunks
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include

        Returns:
            Graph: The created graph
        """
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Find documents to process based on filters and/or specific document IDs
        document_ids = set(documents or [])

        # If filters were provided, get matching documents
        if filters:
            filtered_docs = await self.db.get_documents(auth, filters=filters)
            document_ids.update(doc.external_id for doc in filtered_docs)

        if not document_ids:
            raise ValueError("No documents found matching criteria")

        # Batch retrieve documents for authorization check
        document_objects = await document_service.batch_retrieve_documents(list(document_ids), auth)
        if not document_objects:
            raise ValueError("No authorized documents found matching criteria")

        # Create a new graph with authorization info
        graph = Graph(
            name=name,
            document_ids=[doc.external_id for doc in document_objects],
            filters=filters,
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
            },
        )

        # Extract entities and relationships
        entities, relationships = await self._process_documents_for_entities(
            document_objects, auth, document_service
        )
        
        # Add entities and relationships to the graph
        graph.entities = list(entities.values())
        graph.relationships = relationships

        # Store the graph in the database
        if not await self.db.store_graph(graph):
            raise Exception("Failed to store graph")

        return graph

    async def _process_documents_for_entities(
        self, 
        documents: List[Document], 
        auth: AuthContext,
        document_service
    ) -> Tuple[Dict[str, Entity], List[Relationship]]:
        """Process documents to extract entities and relationships.

        Args:
            documents: List of documents to process
            auth: Authentication context
            document_service: DocumentService instance for retrieving chunks

        Returns:
            Tuple of (entities_dict, relationships_list)
        """
        # Dictionary to collect entities by label (to avoid duplicates)
        entities = {}
        # List to collect all relationships
        relationships = []

        # Collect all chunk sources from documents.
        chunk_sources = [
            ChunkSource(document_id=doc.external_id, chunk_number=i)
            for doc in documents
            for i, _ in enumerate(doc.chunk_ids)
        ]

        # Batch retrieve chunks
        chunks = await document_service.batch_retrieve_chunks(chunk_sources, auth)
        logger.info(f"Retrieved {len(chunks)} chunks for processing")

        # Process each chunk individually
        for chunk in chunks:
            try:
                # Extract entities and relationships from the chunk
                chunk_entities, chunk_relationships = await self.extract_entities_from_text(
                    chunk.content,
                    chunk.document_id,
                    chunk.chunk_number
                )
                
                # Add entities to the collection, avoiding duplicates
                for entity in chunk_entities:
                    if entity.label not in entities:
                        # For new entities, initialize chunk_sources with the current chunk
                        entities[entity.label] = entity
                    else:
                        # If entity already exists, add this chunk source if not already present
                        existing_entity = entities[entity.label]
                        
                        # Add to chunk_sources dictionary
                        if chunk.document_id not in existing_entity.chunk_sources:
                            existing_entity.chunk_sources[chunk.document_id] = [chunk.chunk_number]
                        elif chunk.chunk_number not in existing_entity.chunk_sources[chunk.document_id]:
                            existing_entity.chunk_sources[chunk.document_id].append(chunk.chunk_number)

                # Add the current chunk source to each relationship
                for relationship in chunk_relationships:
                    # Add to chunk_sources dictionary
                    if chunk.document_id not in relationship.chunk_sources:
                        relationship.chunk_sources[chunk.document_id] = [chunk.chunk_number]
                    elif chunk.chunk_number not in relationship.chunk_sources[chunk.document_id]:
                        relationship.chunk_sources[chunk.document_id].append(chunk.chunk_number)
                
                # Add relationships to the collection
                relationships.extend(chunk_relationships)
                
            except ValueError as e:
                # Handle specific extraction errors we've wrapped
                logger.warning(f"Skipping chunk {chunk.chunk_number} in document {chunk.document_id}: {e}")
                continue
            except Exception as e:
                # For other errors, log and re-raise to abort graph creation
                logger.error(f"Fatal error processing chunk {chunk.chunk_number} in document {chunk.document_id}: {e}")
                raise
                
        return entities, relationships

    async def extract_entities_from_text(
        self, content: str, doc_id: str, chunk_number: int
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text content using the LLM.

        Args:
            content: Text content to process
            doc_id: Document ID
            chunk_number: Chunk number within the document

        Returns:
            Tuple of (entities, relationships)
        """
        settings = get_settings()
        
        # Limit text length to avoid token limits
        content_limited = content[:min(len(content), 5000)]
        
        # System message with clear instructions
        system_message = {
            "role": "system",
            "content": (
                "You are an entity extraction assistant. Extract entities and their relationships from text precisely and thoroughly. "
                "For relationships, use a simple format with source, target, and relationship fields."
            )
        }
        
        # User message with text to analyze
        user_message = {
            "role": "user",
            "content": (
                "Extract named entities and their relationships from the following text. "
                "For entities, include entity types (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.) and any relevant properties. "
                "For relationships, simply specify the source entity, target entity, and the relationship between them. "
                "Return your response as valid JSON:\n\n" + content_limited
            )
        }

        if settings.GRAPH_PROVIDER == "openai":
            # Use OpenAI with structured output using parse for Pydantic model
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # The parse method will ensure we get a structured result matching our model
            response = await client.chat.completions.parse(
                model=settings.GRAPH_MODEL,
                messages=[system_message, user_message],
                response_format=ExtractionResult,
            )
            extraction_result = response.choices[0].message.parsed
            
        elif settings.GRAPH_PROVIDER == "ollama":
            # For Ollama, use structured output format
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Create the schema for structured output
                format_schema = ExtractionResult.model_json_schema()
                
                response = await client.post(
                    f"{settings.EMBEDDING_OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": settings.GRAPH_MODEL,
                        "messages": [system_message, user_message],
                        "stream": False,
                        "format": format_schema
                    },
                )
                response.raise_for_status()
                result = response.json()
                
                # Log the raw response for debugging
                logger.info(f"Raw Ollama response for entity extraction: {result['message']['content']}")
                
                # Parse the JSON response - Pydantic will handle validation
                extraction_result = ExtractionResult.model_validate_json(result["message"]["content"])
        else:
            logger.error(f"Unsupported graph provider: {settings.GRAPH_PROVIDER}")
            return [], []

        # Process extraction results
        entities, relationships = self._process_extraction_results(extraction_result, doc_id, chunk_number)
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from document {doc_id}, chunk {chunk_number}")
        return entities, relationships

    def _process_extraction_results(
        self, 
        extraction_result: ExtractionResult, 
        doc_id: str, 
        chunk_number: int
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Process extraction results into entity and relationship objects."""
        # Initialize chunk_sources with the current chunk - reused across entities
        chunk_sources = {doc_id: [chunk_number]}
        
        # Convert extracted data to entity objects using list comprehension
        entities = [
            Entity(
                label=entity.label,
                type=entity.type,
                properties=entity.properties,
                chunk_sources=chunk_sources.copy(),  # Need to copy to avoid shared reference
                document_ids=[doc_id]
            )
            for entity in extraction_result.entities
        ]

        # Create a mapping of entity labels to IDs
        entity_mapping = {entity.label: entity.id for entity in entities}

        # Convert to relationship objects using list comprehension with filtering
        relationships = [
            Relationship(
                source_id=entity_mapping[rel.source],
                target_id=entity_mapping[rel.target],
                type=rel.relationship,
                chunk_sources=chunk_sources.copy(),  # Need to copy to avoid shared reference
                document_ids=[doc_id]
            )
            for rel in extraction_result.relationships
            if rel.source in entity_mapping and rel.target in entity_mapping
        ]

        return entities, relationships

    async def query_with_graph(
        self,
        query: str,
        graph_name: str,
        auth: AuthContext,
        document_service,  # Passed to avoid circular import
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
    ) -> CompletionResponse:
        """Generate completion using knowledge graph-enhanced retrieval.
        
        This method enhances retrieval by:
        1. Extracting entities from the query
        2. Finding similar entities in the graph
        3. Traversing the graph to find related entities
        4. Retrieving chunks containing these entities
        5. Combining with traditional vector search results
        6. Generating a completion with enhanced context
        
        Args:
            query: The query text
            graph_name: Name of the graph to use
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents
            filters: Optional metadata filters
            k: Number of chunks to retrieve
            min_score: Minimum similarity score
            max_tokens: Maximum tokens for completion
            temperature: Temperature for completion
            use_reranking: Whether to use reranking
            use_colpali: Whether to use colpali embedding
            hop_depth: Number of relationship hops to traverse (1-3)
            include_paths: Whether to include relationship paths in response
        """
        logger.info(f"Querying with graph: {graph_name}, hop depth: {hop_depth}")
        
        # Get the knowledge graph
        graph = await self.db.get_graph(graph_name, auth)
        if not graph:
            logger.warning(f"Graph '{graph_name}' not found or not accessible")
            # Fall back to standard retrieval if graph not found
            return await document_service.query(
                query=query,
                auth=auth,
                filters=filters,
                k=k,
                min_score=min_score,
                max_tokens=max_tokens,
                temperature=temperature,
                use_reranking=use_reranking,
                use_colpali=use_colpali,
                graph_name=None,
            )
        
        # Parallel approach
        # 1. Standard vector search
        vector_chunks = await document_service.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali
        )
        logger.info(f"Vector search retrieved {len(vector_chunks)} chunks")
        
        # 2. Graph-based retrieval
        # First extract entities from the query
        query_entities = await self._extract_entities_from_query(query)
        logger.info(f"Extracted {len(query_entities)} entities from query: {', '.join(e.label for e in query_entities)}")
        
        # If no entities extracted, fallback to embedding similarity
        if not query_entities:
            # Find similar entities using embedding similarity
            top_entities = await self._find_similar_entities(query, graph.entities, k)
        else:
            # Use extracted entities directly
            entity_map = {entity.label.lower(): entity for entity in graph.entities}
            matched_entities = []
            
            # Match extracted entities with graph entities
            for query_entity in query_entities:
                if query_entity.label.lower() in entity_map:
                    matched_entities.append(entity_map[query_entity.label.lower()])
                    
            # If no matches, fallback to embedding similarity
            if matched_entities:
                top_entities = [(entity, 1.0) for entity in matched_entities]  # Score 1.0 for direct matches
            else:
                top_entities = await self._find_similar_entities(query, graph.entities, k)
        
        logger.info(f"Found {len(top_entities)} relevant entities in graph")
        
        # Traverse the graph to find related entities
        expanded_entities = self._expand_entities(graph, [e[0] for e in top_entities], hop_depth)
        logger.info(f"Expanded to {len(expanded_entities)} entities after traversal")
        
        # Get specific chunks containing these entities
        graph_chunks = await self._retrieve_entity_chunks(expanded_entities, auth, filters, document_service)
        logger.info(f"Retrieved {len(graph_chunks)} chunks containing relevant entities")
        
        # Calculate paths if requested
        paths = []
        if include_paths:
            paths = self._find_relationship_paths(graph, [e[0] for e in top_entities], hop_depth)
            logger.info(f"Found {len(paths)} relationship paths")
        
        # Combine vector and graph results
        combined_chunks = self._combine_chunk_results(vector_chunks, graph_chunks, k)
        
        # Generate completion with enhanced context
        completion_response = await self._generate_completion(
            query,
            combined_chunks,
            document_service,
            max_tokens,
            temperature,
            include_paths,
            paths,
            auth,
            graph_name
        )
        
        return completion_response
    
    async def _extract_entities_from_query(self, query: str) -> List[Entity]:
        """Extract entities from the query text using the LLM."""
        try:
            # Extract entities from the query using the same extraction function
            # but with a simplified prompt specific for queries
            entities, _ = await self.extract_entities_from_text(
                content=query,
                doc_id="query",  # Use "query" as doc_id 
                chunk_number=0   # Use 0 as chunk_number
            )
            return entities
        except Exception as e:
            # If extraction fails, log and return empty list to fall back to embedding similarity
            logger.warning(f"Failed to extract entities from query: {e}")
            return []
    
    async def _find_similar_entities(
        self, query: str, entities: List[Entity], k: int
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to the query based on embedding similarity."""
        if not entities:
            return []
            
        # Get embedding for query
        query_embedding = await self.embedding_model.embed_for_query(query)
        
        # Create entity text representations and get embeddings for all entities
        entity_texts = [
            f"{entity.label} {entity.type} " + " ".join(
                f"{key}: {value}" for key, value in entity.properties.items()
            )
            for entity in entities
        ]
        
        # Get embeddings for all entity texts
        entity_embeddings = await self._batch_get_embeddings(entity_texts)
        
        # Calculate similarities and pair with entities
        entity_similarities = [
            (entity, self._calculate_cosine_similarity(query_embedding, embedding))
            for entity, embedding in zip(entities, entity_embeddings)
        ]
        
        # Sort by similarity and take top k
        entity_similarities.sort(key=lambda x: x[1], reverse=True)
        return entity_similarities[:min(k, len(entity_similarities))]
        
    async def _batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts efficiently."""
        # This could be implemented with proper batch embedding if the embedding model supports it
        # For now, we'll just map over the texts and get embeddings one by one
        return [await self.embedding_model.embed_for_query(text) for text in texts]
    
    def _expand_entities(self, graph: Graph, seed_entities: List[Entity], hop_depth: int) -> List[Entity]:
        """Expand entities by traversing relationships."""
        if hop_depth <= 1:
            return seed_entities
            
        # Create a set of entity IDs we've seen
        seen_entity_ids = {entity.id for entity in seed_entities}
        all_entities = list(seed_entities)
        
        # Create a map for fast entity lookup
        entity_map = {entity.id: entity for entity in graph.entities}
        
        # For each hop
        for _ in range(hop_depth - 1):
            new_entities = []
            
            # For each entity we've found so far
            for entity in all_entities:
                # Find connected entities through relationships
                connected_ids = self._get_connected_entity_ids(graph.relationships, entity.id, seen_entity_ids)
                
                # Add new connected entities
                for entity_id in connected_ids:
                    if target_entity := entity_map.get(entity_id):
                        new_entities.append(target_entity)
                        seen_entity_ids.add(entity_id)
            
            # Add new entities to our list
            all_entities.extend(new_entities)
            
            # Stop if no new entities found
            if not new_entities:
                break
                
        return all_entities
    
    def _get_connected_entity_ids(
        self, relationships: List[Relationship], entity_id: str, seen_ids: Set[str]
    ) -> Set[str]:
        """Get IDs of entities connected to the given entity that haven't been seen yet."""
        connected_ids = set()
        
        for relationship in relationships:
            # Check outgoing relationships
            if relationship.source_id == entity_id and relationship.target_id not in seen_ids:
                connected_ids.add(relationship.target_id)
                
            # Check incoming relationships
            elif relationship.target_id == entity_id and relationship.source_id not in seen_ids:
                connected_ids.add(relationship.source_id)
                
        return connected_ids
    
    async def _retrieve_entity_chunks(
        self, 
        entities: List[Entity], 
        auth: AuthContext,
        filters: Optional[Dict[str, Any]],
        document_service
    ) -> List[ChunkResult]:
        """Retrieve chunks containing the specified entities."""
        if not entities:
            return []
            
        # Collect all chunk sources from entities using set comprehension
        entity_chunk_sources = {
            (doc_id, chunk_num)
            for entity in entities
            for doc_id, chunk_numbers in entity.chunk_sources.items()
            for chunk_num in chunk_numbers
        }
        
        # Get unique document IDs for authorization check
        doc_ids = {doc_id for doc_id, _ in entity_chunk_sources}
        
        # Check document authorization
        documents = await document_service.batch_retrieve_documents(list(doc_ids), auth)
        
        # Apply filters if needed
        authorized_doc_ids = {
            doc.external_id for doc in documents
            if not filters or all(doc.metadata.get(k) == v for k, v in filters.items())
        }
            
        # Filter chunk sources to only those from authorized documents
        chunk_sources = [
            ChunkSource(document_id=doc_id, chunk_number=chunk_num)
            for doc_id, chunk_num in entity_chunk_sources
            if doc_id in authorized_doc_ids
        ]
        
        # Retrieve and return chunks if we have any valid sources
        return await document_service.batch_retrieve_chunks(chunk_sources, auth) if chunk_sources else []
    
    def _combine_chunk_results(
        self, vector_chunks: List[ChunkResult], graph_chunks: List[ChunkResult], k: int
    ) -> List[ChunkResult]:
        """Combine and deduplicate chunk results from vector search and graph search."""
        # Create dictionary with vector chunks first
        all_chunks = {f"{chunk.document_id}_{chunk.chunk_number}": chunk for chunk in vector_chunks}
        
        # Process and add graph chunks with a boost
        for chunk in graph_chunks:
            chunk_key = f"{chunk.document_id}_{chunk.chunk_number}"
            
            # Set default score if missing and apply boost (5%)
            chunk.score = min(1.0, (getattr(chunk, 'score', 0.7) or 0.7) * 1.05)
            
            # Keep the higher-scored version
            if chunk_key not in all_chunks or chunk.score > all_chunks[chunk_key].score:
                all_chunks[chunk_key] = chunk
        
        # Convert to list, sort by score, and return top k
        return sorted(
            all_chunks.values(), 
            key=lambda x: getattr(x, 'score', 0), 
            reverse=True
        )[:k]
    
    def _find_relationship_paths(
        self, graph: Graph, seed_entities: List[Entity], hop_depth: int
    ) -> List[List[str]]:
        """Find meaningful paths in the graph starting from seed entities."""
        paths = []
        entity_map = {entity.id: entity for entity in graph.entities}
        
        # For each seed entity
        for start_entity in seed_entities:
            # Start BFS from this entity
            queue = [(start_entity.id, [start_entity.label])]
            visited = set([start_entity.id])
            
            while queue:
                entity_id, path = queue.pop(0)
                
                # If path is already at max length, record it but don't expand
                if len(path) >= hop_depth * 2:  # *2 because path includes relationship types
                    paths.append(path)
                    continue
                    
                # Find connected relationships
                for relationship in graph.relationships:
                    # Process both outgoing and incoming relationships
                    if relationship.source_id == entity_id:
                        target_id = relationship.target_id
                        if target_id in visited:
                            continue
                            
                        target_entity = entity_map.get(target_id)
                        if not target_entity:
                            continue
                            
                        # Check for common chunks
                        common_chunks = self._find_common_chunks(
                            entity_map[entity_id], 
                            target_entity, 
                            relationship
                        )
                        
                        # Only include relationships where entities co-occur
                        if common_chunks:
                            visited.add(target_id)
                            # Create path with relationship info
                            rel_context = f"({relationship.type}, {len(common_chunks)} shared chunks)"
                            new_path = path + [rel_context, target_entity.label]
                            queue.append((target_id, new_path))
                            paths.append(new_path)
                            
                    elif relationship.target_id == entity_id:
                        source_id = relationship.source_id
                        if source_id in visited:
                            continue
                            
                        source_entity = entity_map.get(source_id)
                        if not source_entity:
                            continue
                            
                        # Check for common chunks
                        common_chunks = self._find_common_chunks(
                            entity_map[entity_id], 
                            source_entity, 
                            relationship
                        )
                        
                        # Only include relationships where entities co-occur
                        if common_chunks:
                            visited.add(source_id)
                            # Create path with relationship info (note reverse direction)
                            rel_context = f"(is {relationship.type} of, {len(common_chunks)} shared chunks)"
                            new_path = path + [rel_context, source_entity.label]
                            queue.append((source_id, new_path))
                            paths.append(new_path)
        
        return paths
    
    def _find_common_chunks(
        self, entity1: Entity, entity2: Entity, relationship: Relationship
    ) -> Set[Tuple[str, int]]:
        """Find chunks that contain both entities and their relationship."""
        # Get chunk locations for each element
        entity1_chunks = set()
        for doc_id, chunk_numbers in entity1.chunk_sources.items():
            for chunk_num in chunk_numbers:
                entity1_chunks.add((doc_id, chunk_num))
                
        entity2_chunks = set()
        for doc_id, chunk_numbers in entity2.chunk_sources.items():
            for chunk_num in chunk_numbers:
                entity2_chunks.add((doc_id, chunk_num))
                
        rel_chunks = set()
        for doc_id, chunk_numbers in relationship.chunk_sources.items():
            for chunk_num in chunk_numbers:
                rel_chunks.add((doc_id, chunk_num))
        
        # Return intersection
        return entity1_chunks.intersection(entity2_chunks).intersection(rel_chunks)
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Convert to numpy arrays and calculate in one go
        vec1_np, vec2_np = np.array(vec1), np.array(vec2)
        
        # Get magnitudes
        magnitude1, magnitude2 = np.linalg.norm(vec1_np), np.linalg.norm(vec2_np)
        
        # Avoid division by zero and calculate similarity
        return 0 if magnitude1 == 0 or magnitude2 == 0 else np.dot(vec1_np, vec2_np) / (magnitude1 * magnitude2)
    
    async def _generate_completion(
        self,
        query: str,
        chunks: List[ChunkResult],
        document_service,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_paths: bool = False,
        paths: Optional[List[List[str]]] = None,
        auth: Optional[AuthContext] = None,
        graph_name: Optional[str] = None,
    ) -> CompletionResponse:
        """Generate completion using the retrieved chunks and optional path information."""
        if not chunks:
            chunks = []  # Ensure chunks is a list even if empty
            
        # Create document results for context augmentation
        documents = await document_service._create_document_results(
            auth, chunks
        )
        
        # Create augmented chunk contents
        chunk_contents = [
            chunk.augmented_content(documents[chunk.document_id]) 
            for chunk in chunks 
            if chunk.document_id in documents
        ]
        
        # Include graph context in prompt if paths are requested
        if include_paths and paths:
            # Create a readable representation of the paths
            paths_text = "Knowledge Graph Context:\n"
            # Limit to 5 paths to avoid token limits
            for path in paths[:5]:
                paths_text += " -> ".join(path) + "\n"
                
            # Add to the first chunk or create a new first chunk if none
            if chunk_contents:
                chunk_contents[0] = paths_text + "\n\n" + chunk_contents[0]
            else:
                chunk_contents = [paths_text]
                
        # Generate completion
        request = CompletionRequest(
            query=query,
            context_chunks=chunk_contents,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Get completion from model
        response = await document_service.completion_model.complete(request)
        
        # Add sources information
        response.sources = [
            ChunkSource(
                document_id=chunk.document_id,
                chunk_number=chunk.chunk_number,
                score=getattr(chunk, 'score', 0)
            )
            for chunk in chunks
        ]
        
        # Include graph metadata if paths were requested
        if include_paths:
            if not hasattr(response, 'metadata') or response.metadata is None:
                response.metadata = {}
            
            # Extract unique entities from paths (items that don't start with "(")
            unique_entities = set()
            if paths:
                for path in paths[:5]:
                    for item in path:
                        if not item.startswith("("):
                            unique_entities.add(item)
            
            # Add graph-specific metadata
            response.metadata["graph"] = {
                "name": graph_name,
                "relevant_entities": list(unique_entities),
                "paths": [" -> ".join(path) for path in paths[:5]] if paths else [],
            }

        return response
