"""
Prepara manifests de imagen para benchmarks Fase 4.

Dataset: Tiny ImageNet (zh-plus/tiny-imagenet)
  - 100K train + 10K valid = 110K imágenes total
  - 64×64 px RGB, 200 clases (WordNet IDs)

Descarga las imágenes y genera:
  experiments/data/image_1k.json   →   1 000 imágenes
  experiments/data/image_10k.json  →  10 000 imágenes
  experiments/data/image_100k.json → 100 000 imágenes

Uso:
    python experiments/prepare_image_data.py
"""

import json
import random
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "data"
IMAGES_DIR = Path(__file__).parent.parent / "data" / "full" / "tiny_imagenet"

SCALES = {
    "1k":   1_000,
    "10k":  10_000,
    "100k": 100_000,
}
SEED = 42

# Nombres legibles para las primeras 10 clases más usadas (para logs)
SYNSET_NAMES = {
    "n02124075": "cat",
    "n02281787": "lycaenid_butterfly",
    "n02802426": "basketball",
    "n02808440": "bathtub",
    "n02894605": "breakwater",
    "n02909870": "bucket",
    "n03000684": "chain_saw",
    "n03014705": "chest",
    "n03255030": "dumbbell",
    "n04149813": "scoreboard",
}


def load_and_save_tiny_imagenet() -> list[dict]:
    """
    Carga Tiny ImageNet desde HuggingFace, guarda imágenes en disco
    y retorna lista de dicts con {path, id, category}.
    """
    from datasets import load_dataset
    from PIL import Image as PILImage

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando Tiny ImageNet (zh-plus/tiny-imagenet)...")
    ds_full = load_dataset("zh-plus/tiny-imagenet")
    label_names = ds_full["train"].features["label"].names  # WordNet IDs

    # Combinar train + valid
    all_splits = list(ds_full["train"]) + list(ds_full["valid"])
    print(f"  → {len(all_splits)} imágenes totales, {len(label_names)} clases")

    docs = []
    saved = 0

    for i, item in enumerate(all_splits):
        class_id = label_names[item["label"]]       # e.g. 'n01443537'
        img_dir  = IMAGES_DIR / class_id
        img_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / f"{i:07d}.jpg"

        if not img_path.exists():
            pil_img = item["image"]
            if not isinstance(pil_img, PILImage.Image):
                pil_img = PILImage.fromarray(pil_img)
            pil_img.convert("RGB").save(img_path, format="JPEG", quality=90)
            saved += 1

        docs.append({
            "path":     str(img_path),
            "id":       f"tinyimg_{i:07d}",
            "category": class_id,
        })

        if (i + 1) % 10_000 == 0:
            print(f"  {i+1}/{len(all_splits)} procesadas ({saved} guardadas)...")

    print(f"  Guardadas {saved} imágenes nuevas en {IMAGES_DIR}/")
    return docs


def write_manifest(docs: list[dict], n: int, label: str) -> None:
    """Muestreo balanceado por categoría y guarda manifest."""
    random.seed(SEED)

    by_cat: dict[str, list[dict]] = {}
    for d in docs:
        by_cat.setdefault(d["category"], []).append(d)

    n_cats    = len(by_cat)
    per_cat   = n // n_cats
    remainder = n - per_cat * n_cats

    selected: list[dict] = []
    for i, (cat, items) in enumerate(sorted(by_cat.items())):
        take = per_cat + (1 if i < remainder else 0)
        take = min(take, len(items))
        selected.extend(random.sample(items, take))

    random.shuffle(selected)

    out = DATA_DIR / f"image_{label}.json"
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    cats_repr = len(set(d["category"] for d in selected))
    print(f"  [{label.upper()}] {len(selected)} imágenes, {cats_repr} categorías → {out.name}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_and_save_tiny_imagenet()

    for label, n in SCALES.items():
        out_path = DATA_DIR / f"image_{label}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing and "category" in existing[0] and "tinyimg" in existing[0].get("id", ""):
                print(f"  [{label.upper()}] Ya existe con Tiny ImageNet, saltando.")
                continue
        write_manifest(docs, n, label)

    print(f"\nManifests listos en {DATA_DIR}/")
    print(f"Imágenes en {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
