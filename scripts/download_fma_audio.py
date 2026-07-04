"""Download FMA 100K WAV, materialize a deterministic audio subset, and create manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


DATASET = "noahbadoa/fma-dataset-100k-music-wav-files"
ROOT = Path(__file__).resolve().parents[1]
DATA_FULL = ROOT / "data" / "full"
DEFAULT_TARGET = DATA_FULL / "audio" / "fma_100k"
OUT_DIR = ROOT / "experiments" / "data"
SCALES = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
SUPPORTED = {".wav", ".mp3", ".ogg", ".flac"}
MIN_FREE_GB = 220


def check_free_space(path: Path, min_gb: int = MIN_FREE_GB) -> None:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1024**3
    if free_gb < min_gb:
        raise RuntimeError(
            f"Espacio insuficiente para FMA: hay {free_gb:.1f} GB libres en {path}, "
            f"pero se recomiendan al menos {min_gb} GB. "
            "Kaggle descarga un archivo grande y luego lo extrae antes de mover/limpiar."
        )


def download() -> Path:
    DATA_FULL.mkdir(parents=True, exist_ok=True)
    check_free_space(DATA_FULL)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_FULL / ".kaggle_cache"))

    import kagglehub

    print(f"Descargando dataset Kaggle: {DATASET}")
    try:
        return Path(kagglehub.dataset_download(DATASET))
    except OSError as exc:
        raise RuntimeError(
            "Fallo de escritura al descargar FMA. Normalmente esto pasa por falta de "
            "espacio en disco o por un problema del volumen montado de Docker."
        ) from exc


def list_audio_files(dataset_path: Path) -> list[Path]:
    return sorted(
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def genre_for(path: Path, dataset_root: Path) -> str:
    rel = path.relative_to(dataset_root)
    return rel.parts[0] if len(rel.parts) > 1 else "fma"


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    allowed = (DATA_FULL / ".kaggle_cache").resolve()
    if resolved == allowed or allowed in resolved.parents:
        shutil.rmtree(resolved, ignore_errors=True)


def materialize_subset(
    dataset_path: Path,
    files: list[Path],
    target_dir: Path,
    limit: int,
    move: bool,
    clean_cache: bool,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = files[:limit]
    for index, source in enumerate(selected, start=1):
        rel = source.relative_to(dataset_path)
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        if move:
            shutil.move(str(source), str(dest))
        else:
            shutil.copy2(source, dest)
        if index % 5000 == 0:
            print(f"Materializados {index}/{len(selected)} audios...")

    if clean_cache:
        cache_dir = Path(os.environ.get("KAGGLEHUB_CACHE", DATA_FULL / ".kaggle_cache"))
        _safe_rmtree(cache_dir)

    return target_dir


def write_manifests(dataset_path: Path, files: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, n_songs in SCALES.items():
        manifest = [
            {
                "path": str(path),
                "genre": genre_for(path, dataset_path),
                "song_id": path.stem,
                "dataset": "fma_100k",
            }
            for path in files[:n_songs]
        ]
        out = OUT_DIR / f"audio_{label}.json"
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{out}: {len(manifest)} songs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--clean-cache", action="store_true")
    args = parser.parse_args()

    materialized_files = list_audio_files(args.target) if args.materialize and args.target.exists() else []
    if args.materialize and len(materialized_files) >= min(args.limit, SCALES["100k"]):
        dataset_path = args.target
        files = materialized_files
        print("Using existing materialized dataset:", dataset_path)
        print("Materialized audio files:", len(files))
    else:
        dataset_path = download()
        files = list_audio_files(dataset_path)
        print("Path to dataset files:", dataset_path)
        print("Audio files found:", len(files))

        if args.materialize:
            dataset_path = materialize_subset(
                dataset_path=dataset_path,
                files=files,
                target_dir=args.target,
                limit=args.limit,
                move=args.move,
                clean_cache=args.clean_cache,
            )
            files = list_audio_files(dataset_path)
            print("Materialized dataset path:", dataset_path)
            print("Materialized audio files:", len(files))

    if args.write_manifests:
        write_manifests(dataset_path, files)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
