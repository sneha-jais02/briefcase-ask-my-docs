CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('platform_admin', 'admin', 'editor', 'viewer')),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS users_workspace_idx ON users (workspace_id, role);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    source_url TEXT NOT NULL,
    title TEXT NOT NULL,
    jurisdiction TEXT NOT NULL DEFAULT '',
    court TEXT NOT NULL DEFAULT '',
    case_number TEXT NOT NULL DEFAULT '',
    document_date DATE,
    document_hash TEXT NOT NULL,
    version INTEGER NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (workspace_id, source_url, version)
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);

CREATE INDEX IF NOT EXISTS documents_workspace_active_idx ON documents (workspace_id, active, jurisdiction);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    token_count INTEGER NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS chunk_terms (
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    term_frequency INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, term)
);

CREATE INDEX IF NOT EXISTS chunk_terms_term_idx ON chunk_terms (term, chunk_id);

CREATE TABLE IF NOT EXISTS term_stats (
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    document_frequency BIGINT NOT NULL,
    PRIMARY KEY (workspace_id, term)
);

CREATE TABLE IF NOT EXISTS corpus_stats (
    workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
    active_chunks BIGINT NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    user_id UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id);
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'sessions_workspace_user_key'
          AND conrelid = 'sessions'::regclass
    ) THEN
        ALTER TABLE sessions ADD CONSTRAINT sessions_workspace_user_key UNIQUE (id, workspace_id, user_id);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS messages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id UUID NOT NULL,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    user_id UUID NOT NULL REFERENCES users(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE messages ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'messages_session_workspace_user_fk'
          AND conrelid = 'messages'::regclass
    ) THEN
        ALTER TABLE messages ADD CONSTRAINT messages_session_workspace_user_fk
            FOREIGN KEY (session_id, workspace_id, user_id)
            REFERENCES sessions (id, workspace_id, user_id) ON DELETE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, id);
