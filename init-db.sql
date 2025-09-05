-- Initialize the database with pgvector extension and conversation tables

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for conversation summaries
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    participants TEXT,
    topics TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding vector(1536) -- OpenAI embeddings are 1536 dimensions
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS conversation_summaries_embedding_idx 
ON conversation_summaries 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create index for thread_id lookups
CREATE INDEX IF NOT EXISTS conversation_summaries_thread_id_idx 
ON conversation_summaries (thread_id);

-- Create index for created_at for time-based queries
CREATE INDEX IF NOT EXISTS conversation_summaries_created_at_idx 
ON conversation_summaries (created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_conversation_summaries_updated_at 
    BEFORE UPDATE ON conversation_summaries 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
