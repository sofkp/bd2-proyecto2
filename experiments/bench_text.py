"""
Fase 4 — Benchmark Texto (AG News)
Mide latencia, throughput, RAM, I/O y precisión@K del InvertedIndex + SPIMI
y compara contra PostgreSQL GIN (si Docker está disponible).

Uso:
    python experiments/bench_text.py
"""

import json
import re
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil

from backend.src.extractor.tfidf import TFIDFExtractor
from backend.src.codebook.codebook_text import CodebookText
from backend.src.index.inverted_index import InvertedIndex
from backend.src.index.spimi import SpimiIndexer
from backend.src.split.split_text import SplitText

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CODEBOOK_K  = 1000
N_QUERIES   = 50
K           = 10
SPLITTER    = SplitText(min_chars=40, max_chars=800)
EXTRACTOR   = TFIDFExtractor(language="english")


def _load_chunks_from_files(manifest: list[dict]) -> list[dict]:
    """Carga y fragmenta documentos desde archivos .txt."""
    all_chunks = []
    for item in manifest:
        try:
            chunks = SPLITTER.split_file(item["path"])
            all_chunks.extend(chunks)
        except Exception:
            continue
    return all_chunks


def _load_chunks_inline(manifest: list[dict]) -> list[dict]:
    """Fragmenta documentos con texto inline (20newsgroups, AG News, etc.)."""
    all_chunks = []
    for item in manifest:
        try:
            doc_id = item.get("source", f"doc_{len(all_chunks)}")
            chunks = SPLITTER.split_text(
                item["text"],
                document_id=doc_id,
            )
            cat = item.get("category", "unknown")
            for c in chunks:
                c["metadata"]["category"] = cat
            all_chunks.extend(chunks)
        except Exception:
            continue
    return all_chunks


# Mantener alias para compatibilidad con código antiguo
def _ensure_unique_chunk_ids(chunks: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        base_id = chunk["chunk_id"]
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1
        if count == 0:
            continue
        chunk.setdefault("metadata", {})["source_chunk_id"] = base_id
        chunk["chunk_id"] = f"{base_id}__dup{count}_{idx}"
    return chunks


_load_chunks_newsgroups = _load_chunks_inline


def _build_histogram(chunk_tf: dict, codebook: dict) -> np.ndarray:
    hist = np.zeros(len(codebook), dtype=np.float32)
    for word, count in chunk_tf.get("tf", {}).items():
        if word in codebook:
            idx = codebook[word]["index"]
            idf = codebook[word]["idf"]
            hist[idx] += count * idf
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def _index_chunks_pg(chunks: list[dict]) -> bool:
    """Inserta chunks en pg_text_docs y crea schema si no existe."""
    try:
        from backend.api.postgres_indexer import create_schema
        from backend.api.db import get_conn
        create_schema()
        rows = [
            (c["chunk_id"], c["chunk_id"], "", "bench", c["content"])
            for c in chunks if c.get("content")
        ]
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
        print(f"  PostgreSQL: {len(rows)} chunks indexados en pg_text_docs")
        return True
    except Exception as e:
        print(f"  PostgreSQL: no disponible ({e})")
        return False


def _or_tsquery(query: str) -> str:
    """Construye una tsquery OR simple a partir de terminos alfanumericos."""
    terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    return " | ".join(terms) if terms else "emptyqueryterm"


def _try_gin_search(query: str, k: int = 10) -> tuple[list, float]:
    """Intenta consulta GIN flexible en PostgreSQL. Retorna (results, latency_ms)."""
    try:
        from backend.api.db import get_conn
        or_query = _or_tsquery(query)
        t0 = time.perf_counter()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """WITH q AS (
                           SELECT
                             websearch_to_tsquery('english', %s) AS web_q,
                             to_tsquery('english', %s) AS or_q
                       )
                       SELECT chunk_id,
                              ts_rank(tsv, q.web_q) + 0.5 * ts_rank(tsv, q.or_q) AS score
                       FROM pg_text_docs, q
                       WHERE tsv @@ q.web_q OR tsv @@ q.or_q
                       ORDER BY score DESC LIMIT %s""",
                    (query, or_query, k),
                )
                rows = cur.fetchall()
        return rows, (time.perf_counter() - t0) * 1000
    except Exception:
        return [], -1.0


def run_scale(label: str, codebook_k: int = CODEBOOK_K) -> dict:
    manifest_path = DATA_DIR / f"text_{label}.json"
    if not manifest_path.exists():
        print(f"  [SKIP] Manifiesto no encontrado: {manifest_path.name}")
        return {}
    manifest = json.loads(manifest_path.read_text())
    print(f"\n[{label.upper()}] {len(manifest)} documentos fuente | codebook_k={codebook_k}")

    tracemalloc.start()
    io_before  = psutil.disk_io_counters()
    t_total    = time.perf_counter()

    # ── Paso 1: cargar y fragmentar ────────────────────────────────────────────
    t0 = time.perf_counter()
    has_file_paths = "path" in manifest[0]
    chunks = _load_chunks_from_files(manifest) if has_file_paths else _load_chunks_inline(manifest)
    chunks = _ensure_unique_chunk_ids(chunks)
    split_ms = (time.perf_counter() - t0) * 1000
    print(f"  Paso 1/4 split → {len(chunks)} chunks  ({split_ms:.0f}ms)")

    # ── Paso 2: extraer TF crudos ──────────────────────────────────────────────
    t0 = time.perf_counter()
    chunks_tf = EXTRACTOR.extract(chunks)
    extract_ms = (time.perf_counter() - t0) * 1000
    print(f"  Paso 2/4 TF-IDF extract  ({extract_ms:.0f}ms)")

    # ── Paso 3: codebook ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cb = CodebookText(top_k=codebook_k)
    cb.build_codebook(chunks_tf)
    codebook_ms = (time.perf_counter() - t0) * 1000
    print(f"  Paso 3/4 codebook ({len(cb.codebook)} términos)  ({codebook_ms:.0f}ms)")

    # ── Paso 4: histogramas + índice ───────────────────────────────────────────
    t0 = time.perf_counter()
    records: list[dict] = []
    categories: dict[str, str] = {}
    for chunk, ctf in zip(chunks, chunks_tf):
        hist = _build_histogram(ctf, cb.codebook)
        cid  = chunk["chunk_id"]
        cat  = chunk.get("metadata", {}).get("category", "unknown")
        records.append({
            "chunk_id": cid, "modality": "text",
            "histogram": hist, "metadata": {"category": cat},
        })
        categories[cid] = cat
    index = InvertedIndex()
    spimi_blocks = index.build_with_spimi(records, block_size=1000)
    index_ms  = (time.perf_counter() - t0) * 1000
    build_ms  = (time.perf_counter() - t_total) * 1000

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    io_after = psutil.disk_io_counters()
    io_reads = (io_after.read_bytes - io_before.read_bytes) / 1024 ** 2
    print(f"  Paso 4/4 índice ({len(index)} chunks)  ({index_ms:.0f}ms)")

    # ── Consultas propias ──────────────────────────────────────────────────────
    chunk_ids = list(categories.keys())
    category_counts = Counter(categories.values())
    valid_queries = [
        qid for qid in chunk_ids
        if categories[qid] != "unknown" and category_counts[categories[qid]] > 1
    ]
    if not valid_queries:
        valid_queries = chunk_ids
    qids = np.random.choice(
        valid_queries,
        size=min(N_QUERIES, len(valid_queries)),
        replace=False,
    )
    latencies, precisions, recalls = [], [], []
    tp_values, fp_values, fn_values = [], [], []
    for qid in qids:
        t0 = time.perf_counter()
        results = [r for r in index.search(index.get_histogram(qid), k=K + 1) if r.chunk_id != qid][:K]
        latencies.append((time.perf_counter() - t0) * 1000)
        qcat = categories[qid]
        tp = sum(1 for r in results if r.metadata.get("category") == qcat)
        fp = len(results) - tp
        total_relevant = max(category_counts[qcat] - 1, 0)
        fn = max(total_relevant - tp, 0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        tp_values.append(tp)
        fp_values.append(fp)
        fn_values.append(fn)

    avg_lat    = float(np.mean(latencies))
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    # ── Indexar en PostgreSQL y consultar GIN ────────────────────────────────
    pg_ok = _index_chunks_pg(chunks)
    gin_latencies: list[float] = []
    gin_precisions: list[float] = []
    gin_recalls: list[float] = []
    if pg_ok:
        chunk_content = {c["chunk_id"]: c["content"] for c in chunks}
        for qid in qids[:min(20, len(qids))]:
            query_text = chunk_content.get(qid, "")[:200]
            rows, lat = _try_gin_search(query_text, k=K + 1)
            rows = [r for r in rows if r["chunk_id"] != qid][:K]
            if lat >= 0 and rows:
                gin_latencies.append(lat)
                qcat = categories[qid]
                tp = sum(1 for r in rows if categories.get(r["chunk_id"], "") == qcat)
                fp = len(rows) - tp
                total_relevant = max(category_counts[qcat] - 1, 0)
                fn = max(total_relevant - tp, 0)
                gin_precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
                gin_recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    gin_avg   = round(float(np.mean(gin_latencies)), 3)  if gin_latencies  else None
    gin_p_at_k = round(float(np.mean(gin_precisions)), 4) if gin_precisions else None
    gin_r_at_k = round(float(np.mean(gin_recalls)), 4) if gin_recalls else None

    result = {
        "scale": label, "n_source_docs": len(manifest), "n_chunks": len(chunks),
        "n_codewords": len(cb.codebook), "spimi_blocks": spimi_blocks, "k": K,
        "effectiveness_metric": "precision@k and recall@k by dataset category",
        "ground_truth": "AG News category; query document excluded from top-k",
        "n_eval_queries": int(len(qids)),
        "split_ms": round(split_ms, 1), "extract_ms": round(extract_ms, 1),
        "codebook_ms": round(codebook_ms, 1), "index_ms": round(index_ms, 1),
        "build_total_ms": round(build_ms, 1),
        "avg_latency_ms": round(avg_lat, 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "throughput_qps": round(throughput, 1),
        "peak_ram_mb": round(peak_ram / 1024 ** 2, 2),
        "io_read_mb": round(io_reads, 2),
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "avg_tp_at_k": round(float(np.mean(tp_values)), 2),
        "avg_fp_at_k": round(float(np.mean(fp_values)), 2),
        "avg_fn_at_k": round(float(np.mean(fn_values)), 2),
        "gin_avg_latency_ms": gin_avg,
        "gin_precision_at_k": gin_p_at_k,
        "gin_recall_at_k": gin_r_at_k,
        "gin_available": gin_avg is not None,
    }
    print(f"  → {len(chunks)} chunks | lat={avg_lat:.2f}ms | QPS={throughput:.0f} "
          f"| P@{K}={result['precision_at_k']:.3f} | RAM={result['peak_ram_mb']}MB"
          + (f" | GIN={gin_avg}ms P@{K}={gin_p_at_k}" if gin_avg else " | GIN=no disponible"))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark texto Fase 4")
    parser.add_argument(
        "--scales", nargs="+", default=["1k", "10k", "100k"],
        help="Escalas a ejecutar (default: 1k 10k 100k)"
    )
    parser.add_argument(
        "--codebook-sizes", nargs="+", type=int, default=None,
        help="Evalua distintos tamaños de diccionario y guarda text_codebook_k_results.json"
    )
    args = parser.parse_args()

    np.random.seed(42)

    if args.codebook_sizes:
        results = []
        for label in args.scales:
            for size in args.codebook_sizes:
                r = run_scale(label, codebook_k=size)
                if r:
                    results.append(r)
        out = RESULTS_DIR / "text_codebook_k_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResultados de sensibilidad k guardados en {out}")
        raise SystemExit(0)

    # Cargar resultados existentes para no perder otras escalas
    out = RESULTS_DIR / "text_results.json"
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
