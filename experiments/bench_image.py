"""
Fase 4 — Benchmark Imagen (Fashion Product samples)
Mide latencia, throughput, RAM, I/O y precisión@K del VisualSearchIndex
y compara contra pgvector HNSW (si Docker está disponible).

Uso:
    python experiments/bench_image.py
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

from backend.src.extractor.sift import SIFTExtractor
from backend.src.index.visual_search import VisualSearchIndex
from backend.src.split.split_image import SplitImage

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS       = 50
N_QUERIES        = 50
K                = 10
DEFAULT_PATCH_SIZE = 32   # Tiny ImageNet: 64x64 → patches de 32x32
DEFAULT_STRIDE     = 16
CODEBOOK_SAMPLE  = 5_000  # máx imágenes para entrenar KMeans (limita RAM)
SPLITTER   = SplitImage(patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE)
EXTRACTOR  = SIFTExtractor()


def _descriptors_sample(path: str, max_kp: int = 30) -> np.ndarray:
    """
    Extrae muestra de descriptores SIFT de una imagen para entrenar KMeans.
    Retorna array (n, 128) o vacío. No retiene patches en memoria.
    """
    try:
        chunks = SPLITTER.split_file(path)
        patches = [c["content"] for c in chunks]
        descs_list = EXTRACTOR.transform(patches)
        rows = []
        for d in descs_list:
            if d.shape[0] > 0:
                idx = np.random.choice(d.shape[0], size=min(max_kp, d.shape[0]), replace=False)
                rows.append(d[idx])
        return np.vstack(rows) if rows else np.empty((0, 128), dtype=np.float32)
    except Exception:
        return np.empty((0, 128), dtype=np.float32)


def _image_histogram(path: str, kmeans: MiniBatchKMeans) -> np.ndarray:
    """Construye histograma BoVW de una imagen sin retener descriptores."""
    hist = np.zeros(N_CLUSTERS, dtype=np.float32)
    try:
        chunks  = SPLITTER.split_file(path)
        patches = [c["content"] for c in chunks]
        descs_list = EXTRACTOR.transform(patches)
        for d in descs_list:
            if d.shape[0] > 0:
                for c in kmeans.predict(d):
                    hist[c] += 1.0
    except Exception:
        pass
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def _try_pgvector_search(query_vec: np.ndarray, k: int = 10) -> tuple[list, float]:
    """Intenta búsqueda con pgvector HNSW. Retorna (results, latency_ms)."""
    try:
        from backend.api.db import get_conn, vec_to_pg
        t0 = time.perf_counter()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT chunk_id, embedding <-> %s::vector AS dist
                       FROM pg_image_docs ORDER BY dist LIMIT %s""",
                    (vec_to_pg(query_vec), k),
                )
                rows = cur.fetchall()
        return rows, (time.perf_counter() - t0) * 1000
    except Exception:
        return [], -1.0


def run_scale(label: str) -> dict:
    manifest_path = DATA_DIR / f"image_{label}.json"
    if not manifest_path.exists():
        print(f"  [SKIP] Manifiesto no encontrado: {manifest_path.name}")
        return {}
    manifest = json.loads(manifest_path.read_text())
    print(f"\n[{label.upper()}] {len(manifest)} imágenes")

    tracemalloc.start()
    io_before = psutil.disk_io_counters()
    t_total   = time.perf_counter()

    # ── Paso 1: muestrear descriptores para KMeans (memoria acotada) ──────────
    print(f"  Paso 1/4 muestreando descriptores SIFT (max {CODEBOOK_SAMPLE} imgs)...")
    t0 = time.perf_counter()
    sample_items = manifest[:CODEBOOK_SAMPLE]
    frame_rows: list[np.ndarray] = []
    valid_paths: set[str] = set()
    for item in sample_items:
        d = _descriptors_sample(item["path"])
        if d.shape[0] > 0:
            frame_rows.append(d)
            valid_paths.add(item["path"])
    extract_ms = (time.perf_counter() - t0) * 1000
    print(f"    {len(frame_rows)} imágenes con descriptores en muestra  ({extract_ms:.0f}ms)")

    if not frame_rows:
        tracemalloc.stop()
        return {}

    # ── Paso 2: entrenar codebook ─────────────────────────────────────────────
    print("  Paso 2/4 entrenando codebook K-Means...")
    t0 = time.perf_counter()
    frame_matrix = np.vstack(frame_rows)
    del frame_rows   # liberar memoria
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42,
                             n_init=3, max_iter=100, batch_size=512)
    kmeans.fit(frame_matrix)
    del frame_matrix  # liberar memoria
    codebook_ms = (time.perf_counter() - t0) * 1000

    # ── Paso 3: histogramas streaming (una imagen a la vez) ───────────────────
    print("  Paso 3/4 construyendo histogramas e indexando (streaming)...")
    t0 = time.perf_counter()
    index       = VisualSearchIndex()
    histograms  = {}
    categories  = {}
    valid_items = []
    for item in manifest:
        hist = _image_histogram(item["path"], kmeans)
        if hist.sum() == 0:
            continue   # imagen sin keypoints detectables
        index.add_record({
            "chunk_id": item["id"], "modality": "image",
            "histogram": hist.tolist(),
            "metadata": {
                "filename": Path(item["path"]).name,
                "id":       item["id"],
                "category": item.get("category", "unknown"),
            },
        })
        histograms[item["id"]] = hist
        categories[item["id"]] = item.get("category", "unknown")
        valid_items.append(item)
    index_ms = (time.perf_counter() - t0) * 1000
    build_ms  = (time.perf_counter() - t_total) * 1000

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    io_after = psutil.disk_io_counters()
    io_reads = (io_after.read_bytes - io_before.read_bytes) / 1024 ** 2
    print(f"    {len(valid_items)} imágenes indexadas  ({index_ms:.0f}ms)")

    # ── Paso 4: consultas ─────────────────────────────────────────────────────
    img_ids = list(histograms.keys())
    qids = np.random.choice(img_ids, size=min(N_QUERIES, len(img_ids)), replace=False)
    latencies: list[float]  = []
    precisions: list[float] = []
    for qid in qids:
        t0 = time.perf_counter()
        results = index.search(histograms[qid], k=K)
        latencies.append((time.perf_counter() - t0) * 1000)
        qcat = categories[qid]
        rel  = sum(1 for r in results if r.metadata.get("category") == qcat)
        precisions.append(rel / len(results) if results else 0.0)

    avg_lat    = float(np.mean(latencies))
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    # ── pgvector: indexar histogramas y consultar ─────────────────────────────
    pg_ok = False
    try:
        from backend.api.postgres_indexer import create_schema
        from backend.api.db import get_conn, vec_to_pg
        create_schema()
        rows_pg = [
            (item["id"], Path(item["path"]).name, "", item["id"],
             vec_to_pg(histograms[item["id"]]))
            for item in valid_items
        ]
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pg_image_docs")
                cur.executemany(
                    """INSERT INTO pg_image_docs
                           (chunk_id, filename, image_url, title, embedding)
                       VALUES (%s, %s, %s, %s, %s::vector)
                       ON CONFLICT (chunk_id) DO NOTHING""",
                    rows_pg,
                )
            conn.commit()
        print(f"  PostgreSQL: {len(rows_pg)} imágenes indexadas en pg_image_docs")
        pg_ok = True
    except Exception as e:
        print(f"  PostgreSQL: no disponible ({e})")

    pg_latencies: list[float] = []
    pg_precisions: list[float] = []
    if pg_ok:
        for qid in qids[:min(20, len(qids))]:
            rows, lat = _try_pgvector_search(histograms[qid], k=K)
            if lat >= 0:
                pg_latencies.append(lat)
                qcat = categories[qid]
                rel  = sum(1 for r in rows if categories.get(r["chunk_id"], "") == qcat)
                pg_precisions.append(rel / len(rows) if rows else 0.0)
    pg_avg    = round(float(np.mean(pg_latencies)),  3) if pg_latencies  else None
    pg_p_at_k = round(float(np.mean(pg_precisions)), 4) if pg_precisions else None

    result = {
        "scale": label, "n_images": len(manifest), "n_indexed": len(valid_items),
        "n_codewords": N_CLUSTERS, "k": K,
        "extract_ms":    round(extract_ms, 1),
        "codebook_ms":   round(codebook_ms, 1),
        "index_ms":      round(index_ms, 1),
        "build_total_ms": round(build_ms, 1),
        "avg_latency_ms": round(avg_lat, 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "throughput_qps": round(throughput, 1),
        "peak_ram_mb":   round(peak_ram / 1024 ** 2, 2),
        "io_read_mb":    round(io_reads, 2),
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "pgvector_avg_latency_ms": pg_avg,
        "pgvector_precision_at_k": pg_p_at_k,
        "pgvector_available": pg_avg is not None,
    }
    print(f"  → {len(valid_items)} imgs | lat={avg_lat:.3f}ms | QPS={throughput:.0f} "
          f"| P@{K}={result['precision_at_k']:.3f} | RAM={result['peak_ram_mb']:.1f}MB"
          + (f" | pgvector={pg_avg}ms P@{K}={pg_p_at_k}" if pg_avg else " | pgvector=no disponible"))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark imagen Fase 4")
    parser.add_argument(
        "--scales", nargs="+", default=["1k", "10k", "100k"],
        help="Escalas a ejecutar (default: 1k 10k 100k)"
    )
    args = parser.parse_args()

    np.random.seed(42)

    out = RESULTS_DIR / "image_results.json"
    existing: dict[str, dict] = {}
    if out.exists():
        for r in json.loads(out.read_text()):
            existing[r["scale"]] = r

    for label in args.scales:
        r = run_scale(label)
        if r:
            existing[label] = r

    ordered = [existing[s] for s in ["1k", "10k", "100k"] if s in existing]
    out.write_text(json.dumps(ordered, indent=2))
    print(f"\nResultados guardados en {out}")
