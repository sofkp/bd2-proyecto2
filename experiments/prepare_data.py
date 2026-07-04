"""
Fase 4 - Preparacion de datos.
Genera manifiestos para las 3 modalidades oficiales:
texto = AG News, audio = FMA 100K, imagen = Fashion200K.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIRS = [
    ROOT / "data" / "full" / "audio" / "fma_100k",
    ROOT / "data" / "full" / "audio",
]
AUDIO_SCALES = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}


def _save(name: str, data: list) -> Path:
    out = OUT_DIR / name
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


def build_audio_manifests() -> None:
    audio_files = []
    seen = set()
    for audio_dir in AUDIO_DIRS:
        if not audio_dir.exists():
            continue
        for path in sorted(audio_dir.rglob("*")):
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            audio_files.append(path)

    if not audio_files:
        print("  [WARN] No se encontraron archivos de audio en data/full/audio/fma_100k")
        return

    for label, n_songs in AUDIO_SCALES.items():
        songs = audio_files[:n_songs]
        manifest = [
            {"path": str(p), "genre": p.parent.name, "song_id": p.stem}
            for p in songs
        ]
        out = _save(f"audio_{label}.json", manifest)
        print(f"  audio {label:>4}: {len(songs):>6} archivos  ->  {out.name}")


def build_text_manifests() -> None:
    from experiments.prepare_text_data import main as prepare_text

    prepare_text()


def build_image_manifests() -> None:
    from experiments.prepare_image_data import main as prepare_image

    prepare_image()


if __name__ == "__main__":
    print("Generando manifiestos para Fase 4...\n")
    print("[AUDIO - FMA]")
    build_audio_manifests()
    print("\n[TEXTO - AG News]")
    build_text_manifests()
    print("\n[IMAGEN - Fashion200K]")
    build_image_manifests()
    print(f"\nListo - manifiestos guardados en {OUT_DIR}")
