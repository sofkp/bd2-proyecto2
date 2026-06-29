"""
Fase 4 — Generar gráficos comparativos
Lee JSON de resultados (audio, texto, imagen) y genera gráficos.

Uso:
    python experiments/plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
GRAPHICS_DIR = Path(__file__).parent / "grafica_analisis"
GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

# Configurar estilo
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "custom": "#2ecc71",  # Verde
    "postgre": "#e74c3c",  # Rojo
    "pgvector": "#3498db",  # Azul
}


def plot_audio_latency():
    """Gráfico: Latencia promedio (Audio) - Custom vs pgvector por escala."""
    path = RESULTS_DIR / "audio_results.json"
    if not path.exists():
        print(f"  [SKIP] {path.name} no existe")
        return

    results = json.loads(path.read_text())

    scales = []
    custom_lat = []
    pgvector_lat = []

    for r in results:
        scales.append(r["scale"].upper())
        custom_lat.append(r["avg_latency_ms"])
        pg_lat = r.get("pgvector_avg_latency_ms")
        pgvector_lat.append(pg_lat if pg_lat is not None else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scales))
    w = 0.35

    ax.bar(x - w / 2, custom_lat, w, label="AudioSearchIndex (Custom)", color=COLORS["custom"])
    # Filtrar pgvector nulos para la visualización
    pgvector_lat_plot = [v if v > 0 else 0 for v in pgvector_lat]
    if any(v > 0 for v in pgvector_lat_plot):
        ax.bar(x + w / 2, pgvector_lat_plot, w, label="pgvector HNSW", color=COLORS["pgvector"])

    ax.set_xlabel("Escala (chunks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latencia promedio (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Audio: Latencia promedio por escala", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out = GRAPHICS_DIR / "audio_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_audio_throughput():
    """Gráfico: Throughput (QPS) - Audio por escala."""
    path = RESULTS_DIR / "audio_results.json"
    if not path.exists():
        return

    results = json.loads(path.read_text())

    scales = [r["scale"].upper() for r in results]
    qps = [r["throughput_qps"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scales, qps, color=COLORS["custom"], edgecolor="black", linewidth=1.5)

    ax.set_xlabel("Escala (chunks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (QPS)", fontsize=12, fontweight="bold")
    ax.set_title("Audio: Throughput por escala", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Agregar valores en las barras
    for i, v in enumerate(qps):
        ax.text(i, v + max(qps) * 0.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    out = GRAPHICS_DIR / "audio_throughput.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_audio_precision():
    """Gráfico: Precision@10 — Audio: índice propio vs pgvector por escala."""
    path = RESULTS_DIR / "audio_results.json"
    if not path.exists():
        return

    results = json.loads(path.read_text())

    scales        = [r["scale"].upper() for r in results]
    precision     = [r["precision_at_k"] for r in results]
    pg_precision  = [r.get("pgvector_precision_at_k") or 0 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scales))
    w = 0.35

    ax.bar(x - w / 2, precision,    w, label="AudioSearchIndex (Custom)", color=COLORS["custom"], edgecolor="black")
    if any(v > 0 for v in pg_precision):
        ax.bar(x + w / 2, pg_precision, w, label="pgvector HNSW",              color=COLORS["pgvector"], edgecolor="black")

    ax.set_xlabel("Escala (canciones)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision@10", fontsize=12, fontweight="bold")
    ax.set_title("Audio: Precision@10 por escala (Custom vs pgvector)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(precision):
        ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(pg_precision):
        if v > 0:
            ax.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    out = GRAPHICS_DIR / "audio_precision.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_text_latency():
    """Gráfico: Latencia promedio (Texto) - Custom vs GIN por escala."""
    path = RESULTS_DIR / "text_results.json"
    if not path.exists():
        print(f"  [SKIP] {path.name} no existe")
        return

    results = json.loads(path.read_text())

    scales = []
    custom_lat = []
    gin_lat = []

    for r in results:
        scales.append(r["scale"].upper())
        custom_lat.append(r["avg_latency_ms"])
        gin = r.get("gin_avg_latency_ms")
        gin_lat.append(gin if gin is not None else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scales))
    w = 0.35

    ax.bar(x - w / 2, custom_lat, w, label="InvertedIndex (Custom)", color=COLORS["custom"])
    gin_lat_plot = [v if v > 0 else 0 for v in gin_lat]
    if any(v > 0 for v in gin_lat_plot):
        ax.bar(x + w / 2, gin_lat_plot, w, label="PostgreSQL GIN", color=COLORS["postgre"])

    ax.set_xlabel("Escala (chunks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latencia promedio (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Texto: Latencia promedio por escala", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out = GRAPHICS_DIR / "text_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_text_throughput():
    """Gráfico: Throughput (QPS) - Texto por escala."""
    path = RESULTS_DIR / "text_results.json"
    if not path.exists():
        return

    results = json.loads(path.read_text())

    scales = [r["scale"].upper() for r in results]
    qps = [r["throughput_qps"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scales, qps, color=COLORS["custom"], edgecolor="black", linewidth=1.5)

    ax.set_xlabel("Escala (chunks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (QPS)", fontsize=12, fontweight="bold")
    ax.set_title("Texto: Throughput por escala", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(qps):
        ax.text(i, v + max(qps) * 0.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    out = GRAPHICS_DIR / "text_throughput.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_image_latency():
    """Gráfico: Latencia — Imagen por escala."""
    path = RESULTS_DIR / "image_results.json"
    if not path.exists():
        print(f"  [SKIP] {path.name} no existe")
        return

    results = json.loads(path.read_text())

    if not results:
        return

    scales     = [r["scale"].upper() for r in results]
    custom_lat = [r["avg_latency_ms"] for r in results]
    pgvector_lat = [r.get("pgvector_avg_latency_ms") or 0 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scales))
    w = 0.35

    ax.bar(x - w / 2, custom_lat, w, label="VisualSearchIndex (Custom)", color=COLORS["custom"])
    pgvector_plot = [v if v > 0 else 0 for v in pgvector_lat]
    if any(v > 0 for v in pgvector_plot):
        ax.bar(x + w / 2, pgvector_plot, w, label="pgvector HNSW", color=COLORS["pgvector"])

    ax.set_xlabel("Escala (imágenes)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latencia promedio (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Imagen: Latencia promedio por escala (SIFT BoVW)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(custom_lat):
        ax.text(i - w / 2, v + max(custom_lat) * 0.02, f"{v:.2f}ms",
                ha="center", fontsize=10, fontweight="bold")

    out = GRAPHICS_DIR / "image_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_image_precision():
    """Gráfico: Precision@10 — Imagen: índice propio vs pgvector por escala."""
    path = RESULTS_DIR / "image_results.json"
    if not path.exists():
        return

    results = json.loads(path.read_text())
    if not results or "precision_at_k" not in results[0]:
        print("  [SKIP] Sin datos de precisión en image_results.json")
        return

    scales      = [r["scale"].upper() for r in results]
    precision   = [r["precision_at_k"] for r in results]
    pg_prec     = [r.get("pgvector_precision_at_k") or 0 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scales))
    w = 0.35

    ax.bar(x - w / 2, precision, w, label="VisualSearchIndex (Custom)", color=COLORS["custom"],   edgecolor="black")
    if any(v > 0 for v in pg_prec):
        ax.bar(x + w / 2, pg_prec,  w, label="pgvector HNSW",             color=COLORS["pgvector"], edgecolor="black")

    ax.set_xlabel("Escala (imágenes)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision@10", fontsize=12, fontweight="bold")
    ax.set_title("Imagen: Precision@10 por escala (Custom vs pgvector, SIFT BoVW)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylim([0, max(precision) * 1.5 + 0.05])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(precision):
        ax.text(i - w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(pg_prec):
        if v > 0:
            ax.text(i + w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    out = GRAPHICS_DIR / "image_precision.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_text_precision():
    """Gráfico: Precision@10 — Texto: índice propio vs GIN por escala."""
    path = RESULTS_DIR / "text_results.json"
    if not path.exists():
        return

    results = json.loads(path.read_text())

    scales       = [r["scale"].upper() for r in results]
    precision    = [r["precision_at_k"] for r in results]
    gin_prec     = [r.get("gin_precision_at_k") or 0 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scales))
    w = 0.35

    ax.bar(x - w / 2, precision, w, label="InvertedIndex (Custom)", color=COLORS["custom"], edgecolor="black")
    if any(v > 0 for v in gin_prec):
        ax.bar(x + w / 2, gin_prec, w, label="PostgreSQL GIN",         color=COLORS["postgre"], edgecolor="black")

    ax.set_xlabel("Escala (chunks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision@10", fontsize=12, fontweight="bold")
    ax.set_title("Texto: Precision@10 por escala (Custom vs GIN)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(precision):
        ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(gin_prec):
        if v > 0:
            ax.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    out = GRAPHICS_DIR / "text_precision.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_text_ram():
    """Gráfico: Consumo de RAM - Texto por escala."""
    path = RESULTS_DIR / "text_results.json"
    if not path.exists():
        return

    results = json.loads(path.read_text())

    scales = [r["scale"].upper() for r in results]
    ram = [r["peak_ram_mb"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scales, ram, color="#e67e22", edgecolor="black", linewidth=1.5)

    ax.set_xlabel("Escala (chunks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("RAM pico (MB)", fontsize=12, fontweight="bold")
    ax.set_title("Texto: Consumo de RAM por escala", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(ram):
        ax.text(i, v + max(ram) * 0.02, f"{v:.1f} MB", ha="center", fontsize=10, fontweight="bold")

    out = GRAPHICS_DIR / "text_ram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_scalability():
    """Gráfico: Curvas de escalabilidad latencia vs chunks para todas las modalidades."""
    audio_path = RESULTS_DIR / "audio_results.json"
    text_path  = RESULTS_DIR / "text_results.json"
    image_path = RESULTS_DIR / "image_results.json"

    fig, ax = plt.subplots(figsize=(12, 7))

    if audio_path.exists():
        audio_res = json.loads(audio_path.read_text())
        ax_chunks = [r["n_indexed"] for r in audio_res]
        ax_lat    = [r["avg_latency_ms"] for r in audio_res]
        ax.plot(ax_chunks, ax_lat, "o-", color=COLORS["pgvector"], linewidth=2.5,
                markersize=8, label="Audio (AudioSearchIndex)")
        for x, y in zip(ax_chunks, ax_lat):
            ax.annotate(f"{y:.3f}ms", (x, y), textcoords="offset points",
                        xytext=(5, 8), fontsize=9)

    if text_path.exists():
        text_res = json.loads(text_path.read_text())
        tx_chunks = [r["n_chunks"] for r in text_res]
        tx_lat    = [r["avg_latency_ms"] for r in text_res]
        ax.plot(tx_chunks, tx_lat, "s-", color=COLORS["custom"], linewidth=2.5,
                markersize=8, label="Texto (InvertedIndex + SPIMI)")
        for x, y in zip(tx_chunks, tx_lat):
            ax.annotate(f"{y:.2f}ms", (x, y), textcoords="offset points",
                        xytext=(5, 8), fontsize=9)

    if image_path.exists():
        image_res = json.loads(image_path.read_text())
        if image_res:
            im_chunks = [r["n_indexed"] for r in image_res]
            im_lat    = [r["avg_latency_ms"] for r in image_res]
            ax.plot(im_chunks, im_lat, "^-", color=COLORS["postgre"], linewidth=2.5,
                    markersize=8, label="Imagen (VisualSearchIndex SIFT)")
            for x, y in zip(im_chunks, im_lat):
                ax.annotate(f"{y:.2f}ms", (x, y), textcoords="offset points",
                            xytext=(5, -14), fontsize=9)

    ax.set_xlabel("Número de entradas indexadas", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latencia promedio (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Escalabilidad: Latencia vs Entradas Indexadas (todas las modalidades)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    out = GRAPHICS_DIR / "scalability_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_comparison_all_modalities():
    """Gráfico: Comparación de latencia entre las 3 modalidades a escala máxima disponible."""
    audio_path = RESULTS_DIR / "audio_results.json"
    text_path  = RESULTS_DIR / "text_results.json"
    image_path = RESULTS_DIR / "image_results.json"

    data = {}

    if audio_path.exists():
        audio_res = json.loads(audio_path.read_text())
        # Usar la escala mediana (10K) para comparación justa
        audio_10k = next((r for r in audio_res if r["scale"] == "10k"), None)
        if audio_10k:
            data["Audio (10K)"] = audio_10k["avg_latency_ms"]

    if text_path.exists():
        text_res = json.loads(text_path.read_text())
        # Usar 10K también para texto (comparación justa)
        text_10k = next((r for r in text_res if r["scale"] == "10k"), None)
        if text_10k:
            data["Texto (10K)"] = text_10k["avg_latency_ms"]

    if image_path.exists():
        image_res = json.loads(image_path.read_text())
        if image_res:
            data["Imagen (1K)"] = image_res[0]["avg_latency_ms"]

    if not data:
        print(f"  [SKIP] No hay datos suficientes para comparación global")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    modalities = list(data.keys())
    latencies = list(data.values())

    bars = ax.bar(modalities, latencies, color=[COLORS["custom"]] * len(modalities),
                  edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Latencia promedio (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Comparación: Latencia por modalidad (Índices Custom)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(latencies):
        ax.text(i, v + max(latencies) * 0.05, f"{v:.2f}ms", ha="center", fontsize=11, fontweight="bold")

    out = GRAPHICS_DIR / "comparison_all_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)


def plot_precision_comparison():
    """Gráfico: Precisión@10 comparativa — Custom vs pgvector/GIN para las 3 modalidades (escala 10K)."""
    audio_path = RESULTS_DIR / "audio_results.json"
    text_path  = RESULTS_DIR / "text_results.json"
    image_path = RESULTS_DIR / "image_results.json"

    modalities, custom_prec, db_prec, db_labels = [], [], [], []

    if audio_path.exists():
        r = next((x for x in json.loads(audio_path.read_text()) if x["scale"] == "10k"), None)
        if r:
            modalities.append("Audio (10K)")
            custom_prec.append(r["precision_at_k"])
            db_prec.append(r.get("pgvector_precision_at_k") or 0)
            db_labels.append("pgvector")

    if text_path.exists():
        r = next((x for x in json.loads(text_path.read_text()) if x["scale"] == "10k"), None)
        if r:
            modalities.append("Texto (10K)")
            custom_prec.append(r["precision_at_k"])
            db_prec.append(r.get("gin_precision_at_k") or 0)
            db_labels.append("GIN")

    if image_path.exists():
        r = next((x for x in json.loads(image_path.read_text()) if x["scale"] == "10k"), None)
        if r:
            modalities.append("Imagen (10K)")
            custom_prec.append(r["precision_at_k"])
            db_prec.append(r.get("pgvector_precision_at_k") or 0)
            db_labels.append("pgvector")

    if not modalities:
        print("  [SKIP] No hay datos para comparación de precisión")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(modalities))
    w = 0.35

    bars1 = ax.bar(x - w / 2, custom_prec, w, label="Índice propio (Custom)", color=COLORS["custom"], edgecolor="black")
    bars2 = ax.bar(x + w / 2, db_prec,     w, label="pgvector / GIN (PostgreSQL)", color=COLORS["pgvector"], edgecolor="black")

    ax.set_ylabel("Precision@10", fontsize=12, fontweight="bold")
    ax.set_title("Comparativa de Precisión@10: Índice Propio vs PostgreSQL (escala 10K)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=11)
    ax.set_ylim([0, 1.2])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(custom_prec):
        ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(db_prec):
        lbl = f"{v:.3f}" if v > 0 else "N/M"
        ax.text(i + w / 2, max(v, 0.02) + 0.02, lbl, ha="center", fontsize=10, fontweight="bold")

    out = GRAPHICS_DIR / "comparison_precision.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  ✓ {out.name}")
    plt.close(fig)



    print("Generando gráficos...\n[Audio]")
    plot_audio_latency()
    plot_audio_throughput()
    plot_audio_precision()

    print("\n[Texto]")
    plot_text_latency()
    plot_text_throughput()
    plot_text_precision()
    plot_text_ram()

    print("\n[Imagen]")
    plot_image_latency()
    plot_image_precision()

    print("\n[Comparación y Escalabilidad]")
    plot_comparison_all_modalities()
    plot_scalability()
    plot_precision_comparison()

    print(f"\nGráficos guardados en: {GRAPHICS_DIR}/")
