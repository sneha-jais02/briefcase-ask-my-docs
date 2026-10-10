import json
import re
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import Connection, connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app import auth
from app.config import get_settings

TOKEN = re.compile(r"\w+", re.UNICODE)
_pool: ConnectionPool | None = None


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


@contextmanager
def connection() -> Iterator[Connection]:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized.")
    with _pool.connection() as conn:
        yield conn


def initialize_schema() -> None:
    global _pool
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with connect(get_settings().database_url, row_factory=dict_row) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(schema)
        _seed_bootstrap_workspace(conn)
        _migrate_tenant_schema(conn)
        bootstrap_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"briefcase:{get_settings().bootstrap_workspace_name.casefold()}",
        )
        conn.execute(
            "INSERT INTO corpus_stats (workspace_id) VALUES (%s) ON CONFLICT (workspace_id) DO NOTHING",
            (bootstrap_id,),
        )
    _pool = ConnectionPool(
        get_settings().database_url,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        configure=register_vector,
        open=False,
    )
    _pool.open(wait=True)


def _migrate_tenant_schema(conn: Connection) -> None:
    workspace = conn.execute(
        "SELECT id FROM workspaces WHERE name = %s ORDER BY created_at LIMIT 1",
        (get_settings().bootstrap_workspace_name,),
    ).fetchone()
    if not workspace:
        raise RuntimeError("Failed to initialize the bootstrap workspace.")
    workspace_id = workspace["id"]
    user = conn.execute(
        "SELECT id FROM users WHERE email = %s",
        (get_settings().bootstrap_admin_email.casefold(),),
    ).fetchone()
    if not user:
        raise RuntimeError("Failed to initialize the bootstrap administrator.")
    user_id = user["id"]
    conn.execute("UPDATE documents SET workspace_id = %s WHERE workspace_id IS NULL", (workspace_id,))
    conn.execute("UPDATE sessions SET workspace_id = %s WHERE workspace_id IS NULL", (workspace_id,))
    conn.execute("UPDATE sessions SET user_id = %s WHERE user_id IS NULL", (user_id,))
    conn.execute(
        """UPDATE messages AS messages SET workspace_id = sessions.workspace_id, user_id = sessions.user_id
           FROM sessions WHERE messages.session_id = sessions.id
             AND (messages.workspace_id IS NULL OR messages.user_id IS NULL)"""
    )
    conn.execute("ALTER TABLE documents ALTER COLUMN workspace_id SET NOT NULL")
    conn.execute("ALTER TABLE sessions ALTER COLUMN workspace_id SET NOT NULL")
    conn.execute("ALTER TABLE sessions ALTER COLUMN user_id SET NOT NULL")
    conn.execute("ALTER TABLE messages ALTER COLUMN workspace_id SET NOT NULL")
    conn.execute("ALTER TABLE messages ALTER COLUMN user_id SET NOT NULL")
    conn.execute("""
        DO $$
        BEGIN
            ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_url_version_key;
            ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_workspace_id_source_url_version_key;
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'documents_workspace_source_version_unique'
                  AND conrelid = 'documents'::regclass
            ) THEN
                ALTER TABLE documents ADD CONSTRAINT documents_workspace_source_version_unique
                    UNIQUE (workspace_id, source_url, version);
            END IF;
        END $$;
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS documents_active_idx ON documents (workspace_id, active, jurisdiction)"
    )

    term_columns = {
        row["column_name"]
        for row in conn.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'term_stats'"""
        ).fetchall()
    }
    corpus_columns = {
        row["column_name"]
        for row in conn.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'corpus_stats'"""
        ).fetchall()
    }
    if "workspace_id" not in term_columns or "workspace_id" not in corpus_columns:
        conn.execute("DROP TABLE IF EXISTS term_stats")
        conn.execute("DROP TABLE IF EXISTS corpus_stats")
        conn.execute(
            """CREATE TABLE term_stats (
                   workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                   term TEXT NOT NULL,
                   document_frequency BIGINT NOT NULL,
                   PRIMARY KEY (workspace_id, term)
               )"""
        )
        conn.execute(
            """CREATE TABLE corpus_stats (
                   workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
                   active_chunks BIGINT NOT NULL DEFAULT 0,
                   total_tokens BIGINT NOT NULL DEFAULT 0
               )"""
        )
        conn.execute(
            """INSERT INTO corpus_stats (workspace_id, active_chunks, total_tokens)
               SELECT d.id, COUNT(c.id), COALESCE(SUM(c.token_count), 0)
               FROM workspaces AS d
               LEFT JOIN documents ON documents.workspace_id = d.id AND documents.active
               LEFT JOIN chunks AS c ON c.document_id = documents.id
               GROUP BY d.id"""
        )
        conn.execute(
            """INSERT INTO term_stats (workspace_id, term, document_frequency)
               SELECT documents.workspace_id, terms.term, COUNT(DISTINCT terms.chunk_id)
               FROM chunk_terms AS terms
               JOIN chunks ON chunks.id = terms.chunk_id
               JOIN documents ON documents.id = chunks.document_id
               WHERE documents.active
               GROUP BY documents.workspace_id, terms.term"""
        )


def _seed_bootstrap_workspace(conn: Connection) -> None:
    settings = get_settings()
    workspace_id = uuid.uuid5(uuid.NAMESPACE_URL, f"briefcase:{settings.bootstrap_workspace_name.casefold()}")
    conn.execute(
        "INSERT INTO workspaces (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (workspace_id, settings.bootstrap_workspace_name),
    )
    email = settings.bootstrap_admin_email.casefold().strip()
    admin = conn.execute("SELECT id FROM users WHERE email = %s", (email,)).fetchone()
    if not admin:
        password = settings.bootstrap_admin_password or settings.api_key
        if len(password) < 12:
            raise RuntimeError("Set BOOTSTRAP_ADMIN_PASSWORD (at least 12 characters) or a strong API_KEY.")
        conn.execute(
            """INSERT INTO users (id, workspace_id, email, password_hash, role)
               VALUES (%s, %s, %s, %s, 'platform_admin')""",
            (uuid.uuid4(), workspace_id, email, auth.hash_password(password)),
        )


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute(
            """SELECT id::text AS id, workspace_id::text AS workspace_id, email,
                      password_hash, role, active
               FROM users WHERE email = %s""",
            (email.casefold().strip(),),
        ).fetchone()


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    try:
        parsed_user_id = uuid.UUID(user_id)
    except ValueError:
        return None
    with connection() as conn:
        return conn.execute(
            """SELECT id::text AS id, workspace_id::text AS workspace_id, email, role, active
               FROM users WHERE id = %s""",
            (parsed_user_id,),
        ).fetchone()


def create_user(
    workspace_id: uuid.UUID,
    email: str,
    password_hash: str,
    role: str,
) -> dict[str, str]:
    user_id = uuid.uuid4()
    with connection() as conn:
        row = conn.execute(
            """INSERT INTO users (id, workspace_id, email, password_hash, role)
               VALUES (%s, %s, %s, %s, %s)
               RETURNING id::text AS id, workspace_id::text AS workspace_id, email, role""",
            (user_id, workspace_id, email.casefold().strip(), password_hash, role),
        ).fetchone()
    return dict(row)


def create_workspace(name: str, email: str, password_hash: str) -> dict[str, str]:
    workspace_id, user_id = uuid.uuid4(), uuid.uuid4()
    with connection() as conn:
        conn.execute("INSERT INTO workspaces (id, name) VALUES (%s, %s)", (workspace_id, name))
        conn.execute(
            "INSERT INTO corpus_stats (workspace_id) VALUES (%s)",
            (workspace_id,),
        )
        user = conn.execute(
            """INSERT INTO users (id, workspace_id, email, password_hash, role)
               VALUES (%s, %s, %s, %s, 'admin')
               RETURNING id::text AS id, workspace_id::text AS workspace_id, email, role""",
            (user_id, workspace_id, email.casefold().strip(), password_hash),
        ).fetchone()
    return {"id": str(workspace_id), "name": name, "admin_email": user["email"]}


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def ingest_document(
    *,
    workspace_id: uuid.UUID,
    source_url: str,
    title: str,
    jurisdiction: str,
    court: str,
    case_number: str,
    document_date: date | None,
    document_hash: str,
    chunks: list[tuple[str, str]],
    embeddings: list[list[float]],
) -> dict[str, Any]:
    if len(chunks) != len(embeddings):
        raise ValueError("Each document chunk must have a corresponding embedding.")
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))", (str(workspace_id), source_url))
        current = conn.execute(
            """SELECT id, version, document_hash FROM documents
               WHERE workspace_id = %s AND source_url = %s AND active FOR UPDATE""",
            (workspace_id, source_url),
        ).fetchone()
        if current and current["document_hash"] == document_hash:
            conn.execute(
                """UPDATE documents SET title = %s, jurisdiction = %s, court = %s,
                          case_number = %s, document_date = %s
                   WHERE id = %s AND workspace_id = %s""",
                (title, jurisdiction, court, case_number, document_date, current["id"], workspace_id),
            )
            return {"document_id": str(current["id"]), "version": current["version"], "unchanged": True}
        version = current["version"] + 1 if current else 1
        if current:
            conn.execute("UPDATE documents SET active = FALSE WHERE id = %s", (current["id"],))
            old_counts = conn.execute(
                """SELECT COUNT(*) AS chunks, COALESCE(SUM(token_count), 0) AS tokens
                   FROM chunks WHERE document_id = %s""",
                (current["id"],),
            ).fetchone()
            conn.execute(
                """UPDATE corpus_stats
                   SET active_chunks = active_chunks - %s, total_tokens = total_tokens - %s
                   WHERE workspace_id = %s""",
                (old_counts["chunks"], old_counts["tokens"], workspace_id),
            )
            conn.execute(
                """UPDATE term_stats AS stats
                   SET document_frequency = stats.document_frequency - old_counts.frequency
                   FROM (
                       SELECT terms.term, COUNT(*) AS frequency
                       FROM chunk_terms AS terms
                       JOIN chunks ON chunks.id = terms.chunk_id
                       WHERE chunks.document_id = %s
                       GROUP BY terms.term
                   ) AS old_counts
                   WHERE stats.term = old_counts.term AND stats.workspace_id = %s""",
                (current["id"], workspace_id),
            )
            conn.execute(
                "DELETE FROM term_stats WHERE workspace_id = %s AND document_frequency <= 0",
                (workspace_id,),
            )
        document_id = uuid.uuid4()
        conn.execute(
            """INSERT INTO documents
               (id, workspace_id, source_url, title, jurisdiction, court, case_number, document_date, document_hash, version)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                document_id, workspace_id, source_url, title, jurisdiction, court, case_number,
                document_date, document_hash, version,
            ),
        )
        indexed_terms: list[tuple[uuid.UUID, str, int]] = []
        term_document_frequencies: Counter[str] = Counter()
        total_tokens = 0
        for index, ((section_title, content), embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = uuid.uuid4()
            frequencies = Counter(_tokens(f"{section_title}\n{content}"))
            total_tokens += sum(frequencies.values())
            term_document_frequencies.update(frequencies.keys())
            conn.execute(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, section_title, content, embedding, token_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (chunk_id, document_id, index, section_title, content, Vector(embedding), sum(frequencies.values())),
            )
            indexed_terms.extend((chunk_id, term, frequency) for term, frequency in frequencies.items())
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunk_terms (chunk_id, term, term_frequency) VALUES (%s, %s, %s)",
                indexed_terms,
            )
            cursor.executemany(
                """INSERT INTO term_stats (workspace_id, term, document_frequency) VALUES (%s, %s, %s)
                   ON CONFLICT (workspace_id, term) DO UPDATE
                   SET document_frequency = term_stats.document_frequency + EXCLUDED.document_frequency""",
                [(workspace_id, term, frequency) for term, frequency in term_document_frequencies.items()],
            )
        conn.execute(
            """UPDATE corpus_stats
               SET active_chunks = active_chunks + %s, total_tokens = total_tokens + %s
               WHERE workspace_id = %s""",
            (len(chunks), total_tokens, workspace_id),
        )
    return {"document_id": str(document_id), "version": version, "unchanged": False}


def hybrid_search(
    workspace_id: uuid.UUID,
    query: str,
    embedding: list[float],
    jurisdiction: str | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    query_terms = list(dict.fromkeys(_tokens(query)))
    sql = """
        WITH dense AS (
            SELECT c.id AS chunk_id,
                   ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.workspace_id = %s AND d.active AND (%s::text IS NULL OR d.jurisdiction = %s)
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        ),
        bm25_candidates AS (
            SELECT terms.chunk_id,
                   SUM(
                       LN(1.0 + (corpus.active_chunks - stats.document_frequency + 0.5)
                           / (stats.document_frequency + 0.5))
                       * (terms.term_frequency * 2.2)
                       / (terms.term_frequency + 1.2 * (
                           0.25 + 0.75 * chunks.token_count / NULLIF(corpus.average_length, 0)
                       ))
                   ) AS score
            FROM unnest(%s::text[]) AS query_terms(term)
            JOIN term_stats AS stats ON stats.term = query_terms.term AND stats.workspace_id = %s
            JOIN chunk_terms AS terms ON terms.term = query_terms.term
            JOIN chunks ON chunks.id = terms.chunk_id
            JOIN documents ON documents.id = chunks.document_id
            CROSS JOIN (
                SELECT active_chunks, total_tokens::float / NULLIF(active_chunks, 0) AS average_length
                FROM corpus_stats WHERE workspace_id = %s
            ) AS corpus
            WHERE documents.workspace_id = %s AND documents.active
              AND (%s::text IS NULL OR documents.jurisdiction = %s)
            GROUP BY terms.chunk_id, corpus.active_chunks, corpus.average_length
            ORDER BY score DESC
            LIMIT %s
        ),
        sparse AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank
            FROM bm25_candidates
        ),
        fused AS (
            SELECT chunk_id, SUM(1.0 / (60 + rank)) AS hybrid_score
            FROM (SELECT chunk_id, rank FROM dense UNION ALL SELECT chunk_id, rank FROM sparse) candidates
            GROUP BY chunk_id
        )
        SELECT c.id::text AS chunk_id, d.id::text AS document_id, d.title, d.source_url,
               d.jurisdiction, d.court, d.case_number, d.document_date::text AS document_date,
               d.version, c.section_title, c.content, fused.hybrid_score
        FROM fused
        JOIN chunks c ON c.id = fused.chunk_id
        JOIN documents d ON d.id = c.document_id
        ORDER BY fused.hybrid_score DESC
        LIMIT %s
    """
    args = (
        Vector(embedding), workspace_id, jurisdiction, jurisdiction, Vector(embedding), settings.retrieval_candidates,
        query_terms, workspace_id, workspace_id, workspace_id, jurisdiction, jurisdiction, settings.retrieval_candidates,
        settings.retrieval_candidates,
    )
    with connection() as conn:
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        return list(conn.execute(sql, args).fetchall())


def list_documents(workspace_id: uuid.UUID) -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            """SELECT id::text AS id, title, source_url, jurisdiction, court, case_number,
                      document_date::text AS document_date, version, created_at
               FROM documents WHERE workspace_id = %s AND active ORDER BY created_at DESC""",
            (workspace_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_history(
    session_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    limit: int = 8,
) -> list[dict[str, str]]:
    with connection() as conn:
        conn.execute(
            """INSERT INTO sessions (id, workspace_id, user_id) VALUES (%s, %s, %s)
               ON CONFLICT (id) DO NOTHING""",
            (session_id, workspace_id, user_id),
        )
        session = conn.execute(
            "SELECT workspace_id, user_id FROM sessions WHERE id = %s",
            (session_id,),
        ).fetchone()
        if not session or session["workspace_id"] != workspace_id or session["user_id"] != user_id:
            raise PermissionError("The session does not belong to the authenticated user and workspace.")
        rows = conn.execute(
            """SELECT role, content, citations FROM messages WHERE session_id = %s
                 AND workspace_id = %s AND user_id = %s ORDER BY id DESC LIMIT %s""",
            (session_id, workspace_id, user_id, limit),
        ).fetchall()
    history = []
    for row in reversed(rows):
        content = row["content"]
        citations = row["citations"] or []
        if row["role"] == "assistant" and citations:
            prior_sources = [
                {
                    key: source[key]
                    for key in ("id", "title", "source_url", "section_title", "excerpt")
                    if key in source
                }
                for source in citations
            ]
            content += (
                "\nPreviously cited sources (context only; cite evidence retrieved for the current turn): "
                + json.dumps(prior_sources)
            )
        history.append({"role": row["role"], "content": content})
    return history


def save_turn(
    session_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> None:
    with connection() as conn:
        session = conn.execute(
            """SELECT id FROM sessions WHERE id = %s AND workspace_id = %s AND user_id = %s FOR UPDATE""",
            (session_id, workspace_id, user_id),
        ).fetchone()
        if not session:
            raise PermissionError("The session does not belong to the authenticated user and workspace.")
        conn.execute(
            """INSERT INTO messages (session_id, workspace_id, user_id, role, content)
               VALUES (%s, %s, %s, 'user', %s)""",
            (session_id, workspace_id, user_id, question),
        )
        conn.execute(
            """INSERT INTO messages (session_id, workspace_id, user_id, role, content, citations)
               VALUES (%s, %s, %s, 'assistant', %s, %s)""",
            (session_id, workspace_id, user_id, answer, Jsonb(citations)),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = NOW() WHERE id = %s AND workspace_id = %s AND user_id = %s",
            (session_id, workspace_id, user_id),
        )
