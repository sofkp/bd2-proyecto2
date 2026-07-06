"""
Fase 4 - Benchmark Audio (FMA 100K)
Mide latencia, throughput, RAM, I/O y precisión@K del AudioSearchIndex
para escalas 1K, 10K y 100K archivos. Cada cancion es 1 entrada del indice.

Uso:
    python experiments/bench_audio.py
"""

import json
import os
import sys
import time
import tracemalloc
import warnings
from collections import Counter
from contextlib import contextmanager, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import soundfile as sf
from sklearn.cluster import MiniBatchKMeans

from backend.src.extractor.mfcc import MFCCExtractor
from backend.src.index.audio_search import AudioSearchIndex
from backend.src.split.split_audio import SplitAudio

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS  = 512
N_QUERIES   = 50
K           = 10
SPLITTER    = SplitAudio(sample_rate=22050, window_seconds=3.0, hop_seconds=3.0)
EXTRACTOR   = MFCCExtractor(n_mfcc=20, sample_rate=22050)
MAX_AUDIO_SECONDS = 10.0
CODEBOOK_FILE_SAMPLE = 2_000
MAX_FRAMES_PER_FILE = 10


@contextmanager
def _quiet_audio_decode():
    with open(os.devnull, "w") as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stderr(devnull):
                yield


def _resolve_data_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    normalized = path.replace("\\", "/")
    for marker in ("data/full/", "data/samples/"):
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            mapped = ROOT / marker.rstrip("/") / rel
            if mapped.exists():
                return str(mapped)

    return path


def _load_windows(path: str, document_id: str = "audio") -> list:
    import librosa

    with _quiet_audio_decode():
        info = sf.info(path)
        frames = min(int(info.samplerate * MAX_AUDIO_SECONDS), info.frames)
        if frames <= 0:
            return []
        audio, sr = sf.read(path, frames=frames, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SPLITTER.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SPLITTER.sample_rate)
    if audio is None or len(audio) == 0:
        return []
    return [c["content"] for c in SPLITTER.split_audio(audio, document_id=document_id, source_path=path)]


def _frames_sample(path: str, max_frames: int = MAX_FRAMES_PER_FILE) -> np.ndarray:
    """Carga un archivo y retorna una muestra de frames MFCC para KMeans."""
    try:
        windows = _load_windows(path)
        if not windows:
            return np.empty((0, 20), dtype=np.float32)
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
    """Ejecuta benchmark para una escala dada (1k, 10k, 100k)."""
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
    sample_songs = songs[:min(CODEBOOK_FILE_SAMPLE, len(songs))]
    skipped_sample = 0
    for i, s in enumerate(sample_songs, start=1):
        sample = _frames_sample(_resolve_data_path(s["path"]))
        if sample.shape[0] > 0:
            frame_samples.append(sample)
        else:
            skipped_sample += 1
        if i % 1000 == 0:
            print(f"    muestra codebook: {i}/{len(sample_songs)} archivos...")
    frame_matrix = np.vstack(frame_samples) if frame_samples else np.empty((0, 20))
    extract_ms = (time.perf_counter() - t0) * 1000
    print(
        f"    {len(frame_samples)} archivos usados para codebook "
        f"({skipped_sample} omitidos, max {MAX_AUDIO_SECONDS:.0f}s/audio)"
    )
    if frame_matrix.shape[0] < N_CLUSTERS:
        print("  [SKIP] No hay suficientes frames validos para entrenar KMeans.")
        tracemalloc.stop()
        return {}

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
    skipped_index = 0
    for i, s in enumerate(songs, start=1):
        try:
            windows  = _load_windows(_resolve_data_path(s["path"]), document_id=s["song_id"])
            if not windows:
                skipped_index += 1
                continue
            hist     = _build_histogram(windows, kmeans)
            if hist.sum() == 0:
                skipped_index += 1
                continue
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
            skipped_index += 1
            continue
        if i % 1000 == 0:
            print(f"    indexados: {i}/{len(songs)} archivos procesados...")
    index_ms = (time.perf_counter() - t0) * 1000
    build_ms = (time.perf_counter() - t_total_start) * 1000

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    io_after = psutil.disk_io_counters()
    io_reads  = (io_after.read_bytes - io_before.read_bytes) / 1024**2

    # ── Paso 4: consultas ─────────────────────────────────────────────────────
    print(f"  Paso 4/4 ejecutando {N_QUERIES} queries...")
    song_ids = list(histograms.keys())
    if not song_ids:
        print("  [SKIP] No se indexo ningun audio valido.")
        return {}
    qids = np.random.choice(song_ids, size=min(N_QUERIES, len(song_ids)), replace=False)

    genre_counts = Counter(genres.values())
    latencies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    tp_values, fp_values, fn_values = [], [], []
    for qid in qids:
        t0 = time.perf_counter()
        results = [r for r in index.search(histograms[qid], k=K + 1) if r.chunk_id != qid][:K]
        latencies.append((time.perf_counter() - t0) * 1000)
        qgenre   = genres[qid]
        tp = sum(1 for r in results if r.metadata.get("genre") == qgenre)
        fp = len(results) - tp
        total_relevant = max(genre_counts[qgenre] - 1, 0)
        fn = max(total_relevant - tp, 0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        tp_values.append(tp)
        fp_values.append(fp)
        fn_values.append(fn)

    avg_lat    = float(np.mean(latencies))
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    # ── pgvector HNSW ────────────────────────────────────────────────────────────
    pg_ok = _index_audio_pg(histograms, genres)
    pg_latencies: list[float] = []
    pg_precisions: list[float] = []
    pg_recalls: list[float] = []
    if pg_ok:
        for qid in qids[:min(20, len(qids))]:
            rows, lat = _try_pgvector_audio_search(histograms[qid], k=K + 1)
            rows = [r for r in rows if r["chunk_id"] != qid][:K]
            if lat >= 0:
                pg_latencies.append(lat)
                qgenre = genres[qid]
                tp = sum(1 for r in rows if r.get("genre") == qgenre)
                fp = len(rows) - tp
                total_relevant = max(genre_counts[qgenre] - 1, 0)
                fn = max(total_relevant - tp, 0)
                pg_precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
                pg_recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    pg_avg   = round(float(np.mean(pg_latencies)), 3) if pg_latencies else None
    pg_p_at_k = round(float(np.mean(pg_precisions)), 4) if pg_precisions else None
    pg_r_at_k = round(float(np.mean(pg_recalls)), 4) if pg_recalls else None

    result = {
        "scale": label, "n_songs": len(songs), "n_indexed": n_indexed,
        "n_skipped": skipped_index,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "codebook_file_sample": len(sample_songs),
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
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "avg_tp_at_k": round(float(np.mean(tp_values)), 2),
        "avg_fp_at_k": round(float(np.mean(fp_values)), 2),
        "avg_fn_at_k": round(float(np.mean(fn_values)), 2),
        "pgvector_avg_latency_ms": pg_avg,
        "pgvector_precision_at_k": pg_p_at_k,
        "pgvector_recall_at_k": pg_r_at_k,
        "pgvector_available": pg_avg is not None,
    }
    print(f"  → {n_indexed} songs | lat={avg_lat:.2f}ms | QPS={throughput:.0f} "
          f"| P@{K}={result['precision_at_k']:.3f} | RAM={result['peak_ram_mb']}MB"
          + (f" | pgvector={pg_avg}ms" if pg_avg else " | pgvector=no disponible"))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark audio Fase 4")
    parser.add_argument(
        "--scales", nargs="+", default=["1k", "10k", "100k"],
        help="Escalas a ejecutar (default: 1k 10k 100k)",
    )
    args = parser.parse_args()

    np.random.seed(42)

    existing = {}
    out = RESULTS_DIR / "audio_results.json"
    if out.exists():
        for r in json.loads(out.read_text()):
            existing[r["scale"]] = r

    for label in args.scales:
        r = run_scale(label)
        if r:
            existing[label] = r

    scale_rank = {scale: i for i, scale in enumerate(["1k", "10k", "100k"])}
    results = sorted(existing.values(), key=lambda r: scale_rank.get(r["scale"], 999))
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResultados guardados en {out}")
