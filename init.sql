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
