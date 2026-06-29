"""
Fase 4 — Preparación de datos
Genera manifiestos JSON para las 3 escalas × 3 modalidades.
Salida: experiments/data/
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GTZAN_DIR = ROOT / "datasets" / "audio" / "Data" / "genres_original"
ARXIV_DIR = ROOT / "data" / "full" / "arxiv"
IMG_DIR   = ROOT / "data" / "samples" / "images"

# ~59 chunks/song con window=1s hop=0.5s sobre 30s clips
AUDIO_SCALES = {"1k": 17, "10k": 170, "60k": 1000}


def _save(name: str, data: list) -> Path:
    out = OUT_DIR / name
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


def build_audio_manifests() -> None:
    all_wavs = sorted(GTZAN_DIR.rglob("*.wav"))
    if not all_wavs:
        print("  [WARN] No se encontraron WAVs en", GTZAN_DIR)
        return
    for label, n_songs in AUDIO_SCALES.items():
        songs = all_wavs[:n_songs]
        manifest = [
            {"path": str(p), "genre": p.parent.name, "song_id": p.stem}
            for p in songs
        ]
        out = _save(f"audio_{label}.json", manifest)
        print(f"  audio {label:>3}: {len(songs):>4} songs → ~{len(songs)*59:>6} chunks  →  {out.name}")


def build_text_manifests() -> None:
    arxiv_files = sorted(ARXIV_DIR.glob("*.txt"))
    manifest_1k = [
        {"path": str(p), "source": "arxiv", "category": "cs"}
        for p in arxiv_files
    ]
    out = _save("text_1k.json", manifest_1k)
    print(f"  text  1k : {len(manifest_1k):>4} arxiv docs  →  {out.name}")

    try:
        from sklearn.datasets import fetch_20newsgroups
        print("  Descargando 20newsgroups (puede tardar la primera vez)...")
        ng = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        items = [
            {"text": t[:3000], "source": "20news", "category": ng.target_names[c]}
            for t, c in zip(ng.data, ng.target)
            if len(t.strip()) > 80
        ]
        manifest_10k = items[:10000]
        out = _save("text_10k.json", manifest_10k)
        print(f"  text 10k : {len(manifest_10k):>4} 20newsgroups docs  →  {out.name}")
    except Exception as exc:
        print(f"  [WARN] 20newsgroups no disponible: {exc}")


def build_image_manifests() -> None:
    jpgs = sorted(IMG_DIR.glob("*.jpg"))
    manifest = [{"path": str(p), "id": p.stem} for p in jpgs]
    out = _save("image_100.json", manifest)
    print(f"  images   : {len(manifest):>4} JPGs  →  {out.name}")


if __name__ == "__main__":
    print("Generando manifiestos para Fase 4...\n")
    print("[AUDIO]")
    build_audio_manifests()
    print("\n[TEXTO]")
    build_text_manifests()
    print("\n[IMÁGENES]")
    build_image_manifests()
    print(f"\nListo — manifiestos guardados en {OUT_DIR}")
