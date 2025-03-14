-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    external_id VARCHAR PRIMARY KEY,
    owner JSONB,
    content_type VARCHAR,
    filename VARCHAR,
    doc_metadata JSONB DEFAULT '{}',
    storage_info JSONB DEFAULT '{}',
    system_metadata JSONB DEFAULT '{}',
    additional_metadata JSONB DEFAULT '{}',
    access_control JSONB DEFAULT '{}',
    chunk_ids JSONB DEFAULT '[]'
);

-- Create indexes for documents table
CREATE INDEX IF NOT EXISTS idx_owner_id ON documents USING gin(owner);
CREATE INDEX IF NOT EXISTS idx_access_control ON documents USING gin(access_control);
CREATE INDEX IF NOT EXISTS idx_system_metadata ON documents USING gin(system_metadata);

-- Create vector_embeddings table
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255),
    chunk_number INTEGER,
    content TEXT,
    chunk_metadata TEXT,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create caches table
CREATE TABLE IF NOT EXISTS caches (
    name TEXT PRIMARY KEY,
    metadata JSON NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create vector index
CREATE INDEX IF NOT EXISTS vector_idx
ON vector_embeddings USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Initialize multi-vector embeddings table for new multi-vector functionality.
CREATE TABLE IF NOT EXISTS multi_vector_embeddings (
    id BIGSERIAL PRIMARY KEY,
    embeddings BIT(128)[]
);

-- Create function for multi-vector similarity computation
CREATE OR REPLACE FUNCTION max_sim(document bit[], query bit[]) RETURNS double precision AS $$
    WITH queries AS (
        SELECT row_number() OVER () AS query_number, *
        FROM (SELECT unnest(query) AS query) AS foo
    ),
    documents AS (
        SELECT unnest(document) AS document
    ),
    similarities AS (
        SELECT query_number, 1 - ((document <~> query) / bit_length(query)) AS similarity
        FROM queries CROSS JOIN documents
    ),
    max_similarities AS (
        SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
    )
    SELECT SUM(max_similarity) FROM max_similarities;
$$ LANGUAGE SQL;

-- Create graphs table for knowledge graph functionality
CREATE TABLE IF NOT EXISTS graphs (
    id VARCHAR PRIMARY KEY,
    name VARCHAR UNIQUE,
    entities JSONB DEFAULT '[]',
    relationships JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    document_ids JSONB DEFAULT '[]',
    filters JSONB DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    owner JSONB DEFAULT '{}',
    access_control JSONB DEFAULT '{"readers": [], "writers": [], "admins": []}'
);

-- Create index for graph name for faster lookups
CREATE INDEX IF NOT EXISTS idx_graph_name ON graphs(name);
