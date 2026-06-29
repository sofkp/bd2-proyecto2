"""
Fase 4 — Benchmark Audio (GTZAN)
Mide latencia, throughput, RAM, I/O y precisión@K del AudioSearchIndex
para escalas 1K, 10K y ~60K chunks. Cada canción es 1 entrada del índice.

Uso:
    python experiments/bench_audio.py
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from sklearn.cluster import MiniBatchKMeans

from backend.src.extractor.mfcc import MFCCExtractor
from backend.src.index.audio_search import AudioSearchIndex
from backend.src.split.split_audio import SplitAudio

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS  = 50
N_QUERIES   = 50
K           = 10
SPLITTER    = SplitAudio(sample_rate=22050, window_seconds=1.0, hop_seconds=0.5)
EXTRACTOR   = MFCCExtractor(n_mfcc=20, sample_rate=22050)


def _frames_sample(path: str, max_frames: int = 100) -> np.ndarray:
    """Carga un archivo y retorna una muestra de frames MFCC para KMeans."""
    try:
        chunks = SPLITTER.split_file(path)
        windows = [c["content"] for c in chunks]
        frames = EXTRACTOR.extract(windows)          # (M, 20)
        if frames.shape[0] == 0:
            return frames
        idx = np.random.choice(frames.shape[0], size=min(max_frames, frames.shape[0]), replace=False)
        return frames[idx]
    except Exception:
        return np.empty((0, 20), dtype=np.float32)


def _build_histogram(windows: list, kmeans: MiniBatchKMeans) -> np.ndarray:
    hist = np.zeros(N_CLUSTERS, dtype=np.float32)
    for w in windows:
        frames = EXTRACTOR.extract([w])
        if frames.shape[0] == 0:
            continue
        for c in kmeans.predict(frames):
            hist[c] += 1.0
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def _index_audio_pg(histograms: dict, genres: dict) -> bool:
    """Inserta histogramas en pg_audio_docs para benchmark pgvector."""
    try:
        from backend.api.postgres_indexer import create_schema
        from backend.api.db import get_conn, vec_to_pg
        create_schema()
        rows = [
            (sid, sid + ".wav", genres.get(sid, ""), sid, "", vec_to_pg(hist))
            for sid, hist in histograms.items()
        ]
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pg_audio_docs")
                cur.executemany(
                    """INSERT INTO pg_audio_docs
                           (chunk_id, filename, genre, title, audio_url, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s::vector)
                       ON CONFLICT (chunk_id) DO NOTHING""",
                    rows,
                )
            conn.commit()
        print(f"  PostgreSQL: {len(rows)} canciones indexadas en pg_audio_docs")
        return True
    except Exception as e:
        print(f"  PostgreSQL: no disponible ({e})")
        return False


def _try_pgvector_audio_search(
    query_vec: np.ndarray, k: int = 10
) -> tuple[list, float]:
    """Búsqueda pgvector HNSW en pg_audio_docs. Retorna (rows, latency_ms)."""
    try:
        from backend.api.db import get_conn, vec_to_pg
        t0 = time.perf_counter()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT chunk_id, genre,
                              embedding <-> %s::vector AS dist
                       FROM pg_audio_docs ORDER BY dist LIMIT %s""",
                    (vec_to_pg(query_vec), k),
                )
                rows = cur.fetchall()
        return rows, (time.perf_counter() - t0) * 1000
    except Exception:
        return [], -1.0


def run_scale(label: str) -> dict:
    """Ejecuta benchmark para una escala dada (1k, 10k, 60k)."""
    manifest_path = DATA_DIR / f"audio_{label}.json"
    if not manifest_path.exists():
        print(f"  [SKIP] Manifiesto no encontrado: {manifest_path.name}")
        return {}
    songs = json.loads(manifest_path.read_text())
    print(f"\n[{label.upper()}] {len(songs)} canciones")

    tracemalloc.start()
    io_before = psutil.disk_io_counters()
    t_total_start = time.perf_counter()

    # ── Paso 1: muestreo de frames para KMeans ────────────────────────────────
    print("  Paso 1/4 extrayendo frames para codebook...")
    t0 = time.perf_counter()
    frame_samples = []
    for s in songs:
        sample = _frames_sample(s["path"])
        if sample.shape[0] > 0:
            frame_samples.append(sample)
    frame_matrix = np.vstack(frame_samples) if frame_samples else np.empty((0, 20))
    extract_ms = (time.perf_counter() - t0) * 1000

    # ── Paso 2: entrenar codebook ─────────────────────────────────────────────
    print(f"  Paso 2/4 entrenando KMeans con {len(frame_matrix)} frames...")
    t0 = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3,
                             max_iter=100, batch_size=1024)
    kmeans.fit(frame_matrix)
    codebook_ms = (time.perf_counter() - t0) * 1000

    # ── Paso 3: construir histogramas e indexar ───────────────────────────────
    print("  Paso 3/4 construyendo histogramas e indexando...")
    index       = AudioSearchIndex()
    histograms  = {}
    genres      = {}
    t0 = time.perf_counter()
    n_indexed   = 0
    for s in songs:
        try:
            chunks   = SPLITTER.split_file(s["path"], document_id=s["song_id"])
            windows  = [c["content"] for c in chunks]
            hist     = _build_histogram(windows, kmeans)
            index.add_record({
                "chunk_id": s["song_id"],
                "modality": "audio",
                "histogram": hist.tolist(),
                "metadata": {"genre": s["genre"], "song_id": s["song_id"]},
            })
            histograms[s["song_id"]] = hist
            genres[s["song_id"]]     = s["genre"]
            n_indexed += 1
        except Exception:
            continue
    index_ms = (time.perf_counter() - t0) * 1000
    build_ms = (time.perf_counter() - t_total_start) * 1000

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    io_after = psutil.disk_io_counters()
    io_reads  = (io_after.read_bytes - io_before.read_bytes) / 1024**2

    # ── Paso 4: consultas ─────────────────────────────────────────────────────
    print(f"  Paso 4/4 ejecutando {N_QUERIES} queries...")
    song_ids = list(histograms.keys())
    qids = np.random.choice(song_ids, size=min(N_QUERIES, len(song_ids)), replace=False)

    latencies: list[float] = []
    precisions: list[float] = []
    for qid in qids:
        t0 = time.perf_counter()
        results = index.search(histograms[qid], k=K)
        latencies.append((time.perf_counter() - t0) * 1000)
        qgenre   = genres[qid]
        relevant = sum(1 for r in results if r.metadata.get("genre") == qgenre)
        precisions.append(relevant / len(results) if results else 0.0)

    avg_lat    = float(np.mean(latencies))
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    # ── pgvector HNSW ────────────────────────────────────────────────────────────
    pg_ok = _index_audio_pg(histograms, genres)
    pg_latencies: list[float] = []
    pg_precisions: list[float] = []
    if pg_ok:
        for qid in qids[:min(20, len(qids))]:
            rows, lat = _try_pgvector_audio_search(histograms[qid], k=K)
            if lat >= 0:
                pg_latencies.append(lat)
                qgenre = genres[qid]
                rel = sum(1 for r in rows if r.get("genre") == qgenre)
                pg_precisions.append(rel / len(rows) if rows else 0.0)
    pg_avg   = round(float(np.mean(pg_latencies)), 3) if pg_latencies else None
    pg_p_at_k = round(float(np.mean(pg_precisions)), 4) if pg_precisions else None

    result = {
        "scale": label, "n_songs": len(songs), "n_indexed": n_indexed,
        "n_codewords": N_CLUSTERS, "k": K,
        "extract_ms": round(extract_ms, 1),
        "codebook_ms": round(codebook_ms, 1),
        "index_ms": round(index_ms, 1),
        "build_total_ms": round(build_ms, 1),
        "avg_latency_ms": round(avg_lat, 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "throughput_qps": round(throughput, 1),
        "peak_ram_mb": round(peak_ram / 1024**2, 2),
        "io_read_mb": round(io_reads, 2),
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "pgvector_avg_latency_ms": pg_avg,
        "pgvector_precision_at_k": pg_p_at_k,
        "pgvector_available": pg_avg is not None,
    }
    print(f"  → {n_indexed} songs | lat={avg_lat:.2f}ms | QPS={throughput:.0f} "
          f"| P@{K}={result['precision_at_k']:.3f} | RAM={result['peak_ram_mb']}MB"
          + (f" | pgvector={pg_avg}ms" if pg_avg else " | pgvector=no disponible"))
    return result


if __name__ == "__main__":
    np.random.seed(42)
    results = []
    for label in ["1k", "10k", "60k"]:
        r = run_scale(label)
        if r:
            results.append(r)

    out = RESULTS_DIR / "audio_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResultados guardados en {out}")
