from typing import List, Dict, Any, Optional
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import cohere
from core.document import AuthType, DocumentChunk, AuthContext

class ContextualVectorStore:
    """Enhanced vector store combining MongoDB Atlas, Elasticsearch BM25, and reranking."""
    
    def __init__(
        self,
        mongo_store,  # Existing MongoDBAtlasVectorStore
        es_host: str = "http://localhost:9200",
        es_index: str = "contextual_bm25_index",
        semantic_weight: float = 0.8,
        bm25_weight: float = 0.2
    ):
        self.mongo_store = mongo_store
        self.es_client = Elasticsearch(es_host)
        self.es_index = es_index
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.cohere_client = cohere.Client()
        self._setup_es_index()

    def _setup_es_index(self):
        """Create Elasticsearch index with appropriate mappings."""
        if not self.es_client.indices.exists(index=self.es_index):
            index_settings = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "english"}}},
                    "similarity": {"default": {"type": "BM25"}},
                    "index.queries.cache.enabled": False
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "english"},
                        "contextualized_content": {"type": "text", "analyzer": "english"},
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "metadata": {"type": "object"},
                        "system_metadata": {"type": "object"},
                        "permissions": {"type": "object"}
                    }
                }
            }
            self.es_client.indices.create(index=self.es_index, body=index_settings)

    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store embeddings in both MongoDB and Elasticsearch."""
        try:
            # Store in MongoDB
            mongo_success = self.mongo_store.store_embeddings(chunks)
            if not mongo_success:
                return False

            # Prepare documents for Elasticsearch
            es_docs = []
            for chunk in chunks:
                es_doc = {
                    "_index": self.es_index,
                    "_source": {
                        "content": chunk.content,
                        "contextualized_content": chunk.metadata.get("contextualized_content", ""),
                        "doc_id": chunk.system_metadata.doc_id,
                        "chunk_id": chunk.metadata.get("chunk_id"),
                        "metadata": chunk.metadata,
                        "system_metadata": chunk.system_metadata.dict(),
                        "permissions": chunk.permissions
                    }
                }
                es_docs.append(es_doc)

            # Store in Elasticsearch
            if es_docs:
                success, _ = bulk(self.es_client, es_docs)
                self.es_client.indices.refresh(index=self.es_index)
                return success > 0

            return True

        except Exception as e:
            print(f"Error storing embeddings: {str(e)}")
            return False

    def _build_es_query(self, query: str, auth: AuthContext) -> dict:
        """Build Elasticsearch query with auth filtering."""
        if auth.type == AuthType.DEVELOPER:
            auth_filter = {
                "bool": {
                    "should": [
                        {"term": {"system_metadata.dev_id": auth.dev_id}},
                        {"term": {f"permissions.{auth.app_id}": "read"}}
                    ]
                }
            }
        else:
            auth_filter = {"term": {"system_metadata.eu_id": auth.eu_id}}

        return {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "contextualized_content"],
                            "type": "most_fields"
                        }
                    },
                    "filter": auth_filter
                }
            }
        }

    def _merge_results(self, semantic_results: List[dict], bm25_results: List[dict], k: int) -> List[dict]:
        """Merge semantic and BM25 results using reciprocal rank fusion."""
        # Create a map of chunk IDs to their scores
        chunk_scores = {}
        
        # Process semantic search results
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = (result["metadata"]["doc_id"], result["metadata"].get("chunk_id"))
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + (self.semantic_weight / rank)
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = (result["_source"]["doc_id"], result["_source"].get("chunk_id"))
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + (self.bm25_weight / rank)

        # Sort chunks by combined score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        seen_chunks = set()
        for chunk_id, score in sorted_chunks:
            if len(results) >= k:
                break
                
            # Find the full document data from either semantic or BM25 results
            chunk_data = None
            for r in semantic_results:
                if (r["metadata"]["doc_id"], r["metadata"].get("chunk_id")) == chunk_id:
                    chunk_data = r
                    break
                    
            if not chunk_data:
                for r in bm25_results:
                    if (r["_source"]["doc_id"], r["_source"].get("chunk_id")) == chunk_id:
                        chunk_data = {
                            "metadata": r["_source"]["metadata"],
                            "content": r["_source"]["content"],
                            "score": score
                        }
                        break

            if chunk_data and chunk_id not in seen_chunks:
                results.append(chunk_data)
                seen_chunks.add(chunk_id)

        return results

    async def query_similar(
        self,
        query_embedding: List[float],
        query_text: str,
        k: int,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[DocumentChunk]:
        """
        Enhanced query combining semantic search, BM25, and optional reranking.
        Returns merged and optionally reranked results.
        """
        try:
            # Get more results than needed for better reranking
            expanded_k = k * 3 if rerank else k
            
            # Get semantic search results from MongoDB
            semantic_results = self.mongo_store.query_similar(
                query_embedding, 
                expanded_k, 
                auth, 
                filters
            )
            
            # Get BM25 results from Elasticsearch
            es_query = self._build_es_query(query_text, auth)
            es_results = self.es_client.search(
                index=self.es_index,
                body=es_query,
                size=expanded_k
            )
            bm25_results = es_results["hits"]["hits"]
            
            # Merge results using reciprocal rank fusion
            merged_results = self._merge_results(
                semantic_results,
                bm25_results,
                expanded_k
            )
            
            if rerank and len(merged_results) > k:
                # Prepare documents for reranking
                docs = [r["content"] for r in merged_results]
                
                # Rerank using Cohere
                rerank_results = self.cohere_client.rerank(
                    model="rerank-english-v2.0",
                    query=query_text,
                    documents=docs,
                    top_n=k
                )
                
                # Get final results in reranked order
                final_results = []
                for r in rerank_results:
                    doc = merged_results[r.index]
                    doc["score"] = r.relevance_score
                    final_results.append(doc)
                    
                return final_results[:k]
            
            return merged_results[:k]

        except Exception as e:
            print(f"Error querying similar documents: {str(e)}")
            return []