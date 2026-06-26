import time

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from backend.api.db import get_conn, vec_to_pg
from backend.api.image_pipeline import image_pipeline
from backend.api.mfcc_pipeline import mfcc_pipeline

router = APIRouter(prefix="/postgres", tags=["postgres"])


class PgTextRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


def _wrap(results, query_ms: float, index_type: str) -> dict:
    return {
        "results": results,
        "stats": {
            "query_ms": round(query_ms, 2),
            "index_type": index_type,
            "n_comparisons": len(results),
        },
    }


@router.post("/search/text")
def pg_search_text(request: PgTextRequest) -> dict:
    t0 = time.perf_counter()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, title, snippet, source,
                           ts_rank(tsv, plainto_tsquery('english', %s)) AS score
                    FROM pg_text_docs
                    WHERE tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (request.query, request.query, request.k),
                )
                rows = cur.fetchall()
    except Exception as e:
        return _wrap([], 0.0, "GIN tsvector")

    ms = (time.perf_counter() - t0) * 1000
    results = [
        {
            "chunk_id": r["chunk_id"],
            "score": round(float(r["score"]), 4),
            "metadata": {
                "title": r["title"] or r["chunk_id"],
                "snippet": r["snippet"] or "",
                "source": r["source"] or "",
            },
        }
        for r in rows
    ]
    return _wrap(results, ms, "GIN tsvector")


@router.post("/search/image")
async def pg_search_image(file: UploadFile = File(...), k: int = 10) -> dict:
    image_bytes = await file.read()
    # Reusar el pipeline para extraer el histograma de la query
    if not image_pipeline.ready:
        return _wrap([], 0.0, "pgvector HNSW")

    import io
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    chunks = image_pipeline._splitter.split_image(img_array, document_id="query")
    patches = [c["content"] for c in chunks]
    query_hist = image_pipeline._build_histogram(patches, img_array)

    t0 = time.perf_counter()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, filename, image_url, title,
                           GREATEST(0.0, 1.0 - POWER(embedding <-> %s::vector, 2) / 2.0) AS score
                    FROM pg_image_docs
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (vec_to_pg(query_hist), vec_to_pg(query_hist), k),
                )
                rows = cur.fetchall()
    except Exception:
        return _wrap([], 0.0, "pgvector HNSW")

    ms = (time.perf_counter() - t0) * 1000
    results = [
        {
            "chunk_id": r["chunk_id"],
            "score": round(float(r["score"]), 4),
            "metadata": {
                "title": r["title"] or r["chunk_id"],
                "filename": r["filename"],
                "image_url": r["image_url"],
            },
        }
        for r in rows
    ]
    return _wrap(results, ms, "pgvector HNSW")


@router.post("/search/audio")
async def pg_search_audio(file: UploadFile = File(...), k: int = 10) -> dict:
    audio_bytes = await file.read()
    if not mfcc_pipeline.ready:
        return _wrap([], 0.0, "pgvector HNSW")

    import io
    import librosa
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
    chunks = mfcc_pipeline._splitter.split_audio(audio, document_id="query")
    windows = [c["content"] for c in chunks]
    query_hist = mfcc_pipeline._build_histogram(windows)

    t0 = time.perf_counter()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, filename, genre, title, audio_url,
                           GREATEST(0.0, 1.0 - POWER(embedding <-> %s::vector, 2) / 2.0) AS score
                    FROM pg_audio_docs
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (vec_to_pg(query_hist), vec_to_pg(query_hist), k),
                )
                rows = cur.fetchall()
    except Exception:
        return _wrap([], 0.0, "pgvector HNSW")

    ms = (time.perf_counter() - t0) * 1000
    results = [
        {
            "chunk_id": r["chunk_id"],
            "score": round(float(r["score"]), 4),
            "metadata": {
                "title": r["title"] or r["chunk_id"],
                "filename": r["filename"],
                "genre": r["genre"],
                "audio_url": r["audio_url"],
            },
        }
        for r in rows
    ]
    return _wrap(results, ms, "pgvector HNSW")
