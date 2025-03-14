#!/usr/bin/env python3
"""
DataBridge interactive CLI.
Assumes a DataBridge server is running.

Usage:
    Without authentication (connects to localhost):
        python shell.py
    
    With authentication:
        python shell.py <uri>
        Example: python shell.py "databridge://user:token@localhost:8000"

This provides the exact same interface as the Python SDK:
    db.ingest_text("content", metadata={...})
    db.ingest_file("path/to/file")
    db.query("what are the key findings?")
    etc...
"""

import sys
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import requests

# Add local SDK to path before other imports
_SDK_PATH = str(Path(__file__).parent / "sdks" / "python")
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from databridge import DataBridge  # noqa: E402
from databridge.models import Document  # noqa: E402


class DB:
    def __init__(self, uri: str = None):
        """Initialize DataBridge with optional URI"""
        self._client = DataBridge(uri, is_local=True, timeout=1000)
        self.base_url = "http://localhost:8000"  # For health check only

    def check_health(self, max_retries=30, retry_interval=1) -> bool:
        """Check if DataBridge server is healthy with retries"""
        health_url = f"{self.base_url}/health"

        for attempt in range(max_retries):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            if attempt < max_retries - 1:
                print(
                    f"Waiting for DataBridge server to be ready... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_interval)

        return False

    def ingest_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        use_colpali: bool = True,
        as_object: bool = False,
    ) -> Union[dict, 'Document']:
        """
        Ingest text content into DataBridge.

        Args:
            content: Text content to ingest
            metadata: Optional metadata dictionary
            rules: Optional list of rule objects. Examples:
                  [{"type": "metadata_extraction", "schema": {"name": "string"}},
                   {"type": "natural_language", "prompt": "Remove PII"}]
            use_colpali: Whether to use ColPali-style embedding model to ingest the text
            as_object: If True, returns the Document object with update methods, otherwise returns a dict
            
        Returns:
            Document metadata (dict or Document object)
            
        Example:
            ```python
            # Create a document and immediately update it with new content
            doc = db.ingest_text("Initial content", as_object=True)
            doc.update_with_text("Additional content")
            ```
        """
        doc = self._client.ingest_text(
            content, metadata=metadata or {}, rules=rules, use_colpali=use_colpali
        )
        return doc if as_object else doc.model_dump()

    def ingest_file(
        self,
        file: str,
        filename: str = None,
        metadata: dict = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        use_colpali: bool = True,
        as_object: bool = False,
    ) -> Union[dict, 'Document']:
        """
        Ingest a file into DataBridge.

        Args:
            file: Path to file to ingest
            filename: Optional filename (defaults to basename of file path)
            metadata: Optional metadata dictionary
            rules: Optional list of rule objects. Examples:
                  [{"type": "metadata_extraction", "schema": {"title": "string"}},
                   {"type": "natural_language", "prompt": "Summarize"}]
            use_colpali: Whether to use ColPali-style embedding model to ingest the file
            as_object: If True, returns the Document object with update methods, otherwise returns a dict
            
        Returns:
            Document metadata (dict or Document object)
            
        Example:
            ```python
            # Create a document from a file and immediately update it with text
            doc = db.ingest_file("document.pdf", as_object=True)
            doc.update_with_text("Additional notes about this document")
            ```
        """
        file_path = Path(file)
        filename = filename or file_path.name
        doc = self._client.ingest_file(
            file=file_path,
            filename=filename,
            metadata=metadata or {},
            rules=rules,
            use_colpali=use_colpali,
        )
        return doc if as_object else doc.model_dump()

    def retrieve_chunks(
        self,
        query: str,
        filters: dict = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> list:
        """
        Search for relevant chunks

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model for retrieval
        """
        results = self._client.retrieve_chunks(
            query, filters=filters or {}, k=k, min_score=min_score, use_colpali=use_colpali
        )
        return [r.model_dump() for r in results]

    def retrieve_docs(
        self,
        query: str,
        filters: dict = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> list:
        """
        Retrieve relevant documents

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model for retrieval
        """
        results = self._client.retrieve_docs(
            query, filters=filters or {}, k=k, min_score=min_score, use_colpali=use_colpali
        )
        return [r.model_dump() for r in results]

    def query(
        self,
        query: str,
        filters: dict = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: int = None,
        temperature: float = None,
        use_colpali: bool = True,
        graph_name: str = None,
        hop_depth: int = 1,
        include_paths: bool = False,
    ) -> dict:
        """
        Generate completion using relevant chunks as context

        Args:
            query: Query text
            filters: Optional metadata filters
            k: Number of chunks to use as context (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            max_tokens: Maximum tokens in completion
            temperature: Model temperature
            use_colpali: Whether to use ColPali-style embedding model for retrieval
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            
        Examples:
            Standard query:
            >>> db.query("What are the key findings?", filters={"category": "research"})
            
            Knowledge graph enhanced query:
            >>> db.query("How does product X relate to customer segment Y?", 
                         graph_name="market_graph", hop_depth=2, include_paths=True)
                         
            # If include_paths=True, you can inspect the graph paths
            >>> response = db.query("...", graph_name="sales_graph", include_paths=True)
            >>> if "graph" in response.get("metadata", {}):
            >>>     for path in response["metadata"]["graph"]["paths"]:
            >>>         print(" -> ".join(path))
        """
        response = self._client.query(
            query,
            filters=filters or {},
            k=k,
            min_score=min_score,
            max_tokens=max_tokens,
            temperature=temperature,
            use_colpali=use_colpali,
            graph_name=graph_name,
            hop_depth=hop_depth,
            include_paths=include_paths,
        )
        return response.model_dump()

    def list_documents(self, skip: int = 0, limit: int = 100, filters: dict = None, as_objects: bool = False) -> list:
        """
        List accessible documents
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional metadata filters
            as_objects: If True, returns Document objects with update methods, otherwise returns dicts
            
        Returns:
            List of documents (as dicts or Document objects)
            
        Example:
            ```python
            # Get a list of documents that can be updated
            docs = db.list_documents(as_objects=True)
            for doc in docs:
                doc.update_metadata({"status": "reviewed"})
            ```
        """
        docs = self._client.list_documents(skip=skip, limit=limit, filters=filters or {})
        return docs if as_objects else [doc.model_dump() for doc in docs]

    def get_document(self, document_id: str, as_object: bool = False) -> Union[dict, 'Document']:
        """
        Get document metadata by ID
        
        Args:
            document_id: ID of the document
            as_object: If True, returns the Document object with update methods, otherwise returns a dict
            
        Returns:
            Document metadata (dict or Document object)
        """
        doc = self._client.get_document(document_id)
        return doc if as_object else doc.model_dump()
        
    def get_document_by_filename(self, filename: str, as_object: bool = False) -> Union[dict, 'Document']:
        """
        Get document metadata by filename
        
        Args:
            filename: Filename of the document
            as_object: If True, returns the Document object with update methods, otherwise returns a dict
            
        Returns:
            Document metadata (dict or Document object)
            
        Example:
            ```python
            # Get a document by its filename
            doc = db.get_document_by_filename("report.pdf")
            print(f"Document ID: {doc['external_id']}")
            ```
        """
        doc = self._client.get_document_by_filename(filename)
        return doc if as_object else doc.model_dump()
        
    def update_document_with_text(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: bool = None,
    ) -> dict:
        """
        Update a document with new text content using the specified strategy.
        
        Args:
            document_id: ID of the document to update
            content: The new content to add
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding
            
        Returns:
            Updated document metadata
        """
        doc = self._client.update_document_with_text(
            document_id=document_id,
            content=content,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )
        return doc.model_dump()
        
    def update_document_with_file(
        self,
        document_id: str,
        file: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: bool = None,
    ) -> dict:
        """
        Update a document with content from a file using the specified strategy.
        
        Args:
            document_id: ID of the document to update
            file: Path to file to add
            filename: Name of the file (optional, defaults to basename of file path)
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding
            
        Returns:
            Updated document metadata
        """
        file_path = Path(file)
        filename = filename or file_path.name
        
        doc = self._client.update_document_with_file(
            document_id=document_id,
            file=file_path,
            filename=filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )
        return doc.model_dump()
        
    def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any],
    ) -> dict:
        """
        Update only the metadata of a document.
        
        Args:
            document_id: ID of the document to update
            metadata: New metadata to set
            
        Returns:
            Document: Updated document metadata
        """
        doc = self._client.update_document_metadata(
            document_id=document_id,
            metadata=metadata,
        )
        return doc.model_dump()
        
    def update_document_by_filename_with_text(
        self,
        filename: str,
        content: str,
        new_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: bool = None,
    ) -> dict:
        """
        Update a document identified by filename with new text content.
        
        Args:
            filename: Filename of the document to update
            content: The new content to add
            new_filename: Optional new filename for the document
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding
            
        Returns:
            Updated document metadata
        """
        doc = self._client.update_document_by_filename_with_text(
            filename=filename,
            content=content,
            new_filename=new_filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )
        return doc.model_dump()
        
    def update_document_by_filename_with_file(
        self,
        filename: str,
        file: str,
        new_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: bool = None,
    ) -> dict:
        """
        Update a document identified by filename with content from a file.
        
        Args:
            filename: Filename of the document to update
            file: Path to file to add
            new_filename: Optional new filename for the document
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding
            
        Returns:
            Updated document metadata
        """
        file_path = Path(file)
        new_filename = new_filename or file_path.name
        
        doc = self._client.update_document_by_filename_with_file(
            filename=filename,
            file=file_path,
            new_filename=new_filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )
        return doc.model_dump()
        
    def update_document_by_filename_metadata(
        self,
        filename: str,
        metadata: Dict[str, Any],
        new_filename: Optional[str] = None,
    ) -> dict:
        """
        Update a document's metadata using filename to identify the document.
        
        Args:
            filename: Filename of the document to update
            metadata: New metadata to set
            new_filename: Optional new filename to assign to the document
            
        Returns:
            Document: Updated document metadata
        """
        doc = self._client.update_document_by_filename_metadata(
            filename=filename,
            metadata=metadata,
            new_filename=new_filename,
        )
        return doc.model_dump()
    
    def batch_get_documents(self, document_ids: List[str], as_objects: bool = False) -> List[Union[dict, 'Document']]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.
        
        Args:
            document_ids: List of document IDs to retrieve
            as_objects: If True, returns Document objects with update methods, otherwise returns dicts
            
        Returns:
            List of document metadata (as dicts or Document objects)
            
        Example:
            ```python
            # Get multiple documents that can be updated
            docs = db.batch_get_documents(["doc_123", "doc_456"], as_objects=True)
            for doc in docs:
                doc.update_metadata({"batch_processed": True})
            ```
        """
        docs = self._client.batch_get_documents(document_ids)
        return docs if as_objects else [doc.model_dump() for doc in docs]
        
    def batch_get_chunks(self, sources: List[dict]) -> List[dict]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation.
        
        Args:
            sources: List of dictionaries with document_id and chunk_number fields
            
        Returns:
            List of chunk results
            
        Example:
            sources = [
                {"document_id": "doc_123", "chunk_number": 0},
                {"document_id": "doc_456", "chunk_number": 2}
            ]
        """
        chunks = self._client.batch_get_chunks(sources)
        return [chunk.model_dump() for chunk in chunks]

    def create_cache(
        self,
        name: str,
        model: str,
        gguf_file: str,
        filters: dict = None,
        docs: list = None,
    ) -> dict:
        """Create a new cache with specified configuration"""
        response = self._client.create_cache(
            name=name,
            model=model,
            gguf_file=gguf_file,
            filters=filters or {},
            docs=docs,
        )
        return response

    def get_cache(self, name: str) -> "Cache":
        """Get a cache by name"""
        return self._client.get_cache(name)

    def create_graph(
        self,
        name: str,
        filters: Dict[str, Any] = None,
        documents: List[str] = None,
    ) -> dict:
        """
        Create a graph from documents.

        This function processes documents matching filters or specific document IDs,
        extracts entities and relationships, and saves them as a graph.

        Args:
            name: Name of the graph to create
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include

        Returns:
            dict: Information about the created graph

        Examples:
            Create a graph from documents with category="research":
            >>> db.create_graph("research_graph", filters={"category": "research"})

            Create a graph from specific documents:
            >>> db.create_graph("custom_graph", documents=["doc1", "doc2", "doc3"])
        """
        graph = self._client.create_graph(
            name=name,
            filters=filters,
            documents=documents,
        )
        # Return the graph directly since it's already a dictionary
        return graph

    def close(self):
        """Close the client connection"""
        self._client.close()


class Cache:
    def __init__(self, db: DB, name: str):
        self._db = db
        self._name = name
        self._client_cache = db._client.get_cache(name)

    def update(self) -> bool:
        """Update the cache"""
        return self._client_cache.update()

    def add_docs(self, docs: list) -> bool:
        """Add documents to the cache"""
        return self._client_cache.add_docs(docs)

    def query(self, query: str, max_tokens: int = None, temperature: float = None) -> dict:
        """Query the cache"""
        response = self._client_cache.query(
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.model_dump()


if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else None
    db = DB(uri)

    # Check server health
    if not db.check_health():
        print("Error: Could not connect to DataBridge server")
        sys.exit(1)

    print("\nConnected to DataBridge")

    # Start an interactive Python shell with 'db' already imported
    import code
    import readline  # Enable arrow key history
    import rlcompleter  # noqa: F401 # Enable tab completion

    readline.parse_and_bind("tab: complete")

    # Create the interactive shell
    shell = code.InteractiveConsole(locals())

    # Print welcome message
    print("\nDataBridge CLI ready to use. The 'db' object is available with all SDK methods.")
    print("Examples:")
    print("  db.ingest_text('hello world')")
    print("  db.query('what are the key findings?')")
    print("  db.batch_get_documents(['doc_id1', 'doc_id2'])")
    print("  db.batch_get_chunks([{'document_id': 'doc_123', 'chunk_number': 0}])")
    print("\nUpdate by Document ID:")
    print("  db.get_document('doc_123')")
    print("  db.update_document_with_text('doc_123', 'This is new content to append', filename='updated_doc.txt')")
    print("  db.update_document_with_file('doc_123', 'path/to/file.pdf', metadata={'status': 'updated'})")
    print("  db.update_document_metadata('doc_123', {'reviewed': True, 'reviewer': 'John'})")
    print("\nUpdate by Filename:")
    print("  db.get_document_by_filename('report.pdf')")
    print("  db.update_document_by_filename_with_text('report.pdf', 'New content', new_filename='updated_report.pdf')")
    print("  db.update_document_by_filename_with_file('report.pdf', 'path/to/new_data.pdf')")
    print("  db.update_document_by_filename_metadata('report.pdf', {'reviewed': True}, new_filename='reviewed_report.pdf')")
    print("\nQuerying:")
    print("  result = db.query('how to use this API?'); print(result['sources'])")
    print("Example: db.ingest_text('hello world')")
    print("Example: db.create_graph('knowledge_graph', filters={'category': 'research'})")
    print("Example: db.query('How does X relate to Y?', graph_name='knowledge_graph', include_paths=True)")
    print("Type help(db) for documentation.")

    # Start the shell
    shell.interact(banner="")
