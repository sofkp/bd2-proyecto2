from pathlib import Path

import numpy as np

from backend.api.db import get_conn, vec_to_pg


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- Texto: búsqueda full-text con índice GIN sobre tsvector
CREATE TABLE IF NOT EXISTS pg_text_docs (
    id       SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    title    TEXT,
    snippet  TEXT,
    source   TEXT,
    content  TEXT NOT NULL,
    tsv      tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_pg_text_gin ON pg_text_docs USING GIN(tsv);

-- Imágenes: pgvector HNSW (SIFT BoVW, 50 dims)
CREATE TABLE IF NOT EXISTS pg_image_docs (
    id        SERIAL PRIMARY KEY,
    chunk_id  TEXT UNIQUE NOT NULL,
    filename  TEXT,
    image_url TEXT,
    title     TEXT,
    embedding vector(50)
);
CREATE INDEX IF NOT EXISTS idx_pg_image_hnsw
    ON pg_image_docs USING hnsw(embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

-- Audio: pgvector HNSW (MFCC BoW, 50 dims)
CREATE TABLE IF NOT EXISTS pg_audio_docs (
    id        SERIAL PRIMARY KEY,
    chunk_id  TEXT UNIQUE NOT NULL,
    filename  TEXT,
    genre     TEXT,
    title     TEXT,
    audio_url TEXT,
    embedding vector(50)
);
CREATE INDEX IF NOT EXISTS idx_pg_audio_hnsw
    ON pg_audio_docs USING hnsw(embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
"""


def create_schema() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


def index_text(txt_dirs: list[Path]) -> int:
    txt_files: list[Path] = []
    for d in txt_dirs:
        if d.exists():
            txt_files.extend(sorted(d.glob("*.txt")))

    if not txt_files:
        return 0

    rows = []
    for f in txt_files:
        try:
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                continue
            lines = [l for l in content.splitlines() if l.strip()]
            title = lines[0].strip() if lines else f.stem
            snippet = " ".join(lines[1:]).strip()[:300] if len(lines) > 1 else ""
            rows.append((f.stem, title, snippet, f.stem, content))
        except Exception:
            continue

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pg_text_docs")
            cur.executemany(
                """INSERT INTO pg_text_docs (chunk_id, title, snippet, source, content)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                rows,
            )
        conn.commit()
    return len(rows)


def index_images(image_histograms: dict, image_metadata: dict) -> int:
    rows = []
    for chunk_id, hist in image_histograms.items():
        meta = image_metadata.get(chunk_id, {})
        rows.append((
            chunk_id,
            meta.get("filename", ""),
            meta.get("image_url", ""),
            meta.get("title", chunk_id),
            vec_to_pg(hist),
        ))

    if not rows:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pg_image_docs")
            cur.executemany(
                """INSERT INTO pg_image_docs (chunk_id, filename, image_url, title, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                rows,
            )
        conn.commit()
    return len(rows)


def index_audio(audio_histograms: dict, audio_metadata: dict) -> int:
    rows = []
    for chunk_id, hist in audio_histograms.items():
        meta = audio_metadata.get(chunk_id, {})
        rows.append((
            chunk_id,
            meta.get("filename", ""),
            meta.get("genre", ""),
            meta.get("title", chunk_id),
            meta.get("audio_url", ""),
            vec_to_pg(hist),
        ))

    if not rows:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pg_audio_docs")
            cur.executemany(
                """INSERT INTO pg_audio_docs (chunk_id, filename, genre, title, audio_url, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                rows,
            )
        conn.commit()
    return len(rows)
