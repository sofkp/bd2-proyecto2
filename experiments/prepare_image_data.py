"""
Prepara manifests de imagen para benchmarks Fase 4.

Dataset: Marqo/fashion200k
  - 202K filas de ropa/moda
  - columnas: image, category1, category2, category3, text, item_ID

Descarga/guarda imagenes y genera:
  experiments/data/image_1k.json
  experiments/data/image_10k.json
  experiments/data/image_100k.json
"""

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = Path(__file__).parent.parent / "data" / "full" / "fashion200k"

SCALES = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
}
SEED = 42
MAX_IMAGES = max(SCALES.values())


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return value.strip("._") or "unknown"


def load_and_save_fashion200k() -> list[dict]:
    """Carga Fashion200K desde HuggingFace, guarda 100K imagenes y retorna manifest base."""
    from datasets import load_dataset
    from PIL import Image as PILImage

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando Fashion200K (Marqo/fashion200k)...")
    ds = load_dataset("Marqo/fashion200k", split="data")
    print(f"  -> {len(ds)} filas disponibles")

    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    docs: list[dict] = []
    saved = 0

    for row_idx in indices:
        if len(docs) >= MAX_IMAGES:
            break

        item = ds[row_idx]
        category = _safe_name(item.get("category1", "fashion"))
        item_id = _safe_name(item.get("item_ID", f"fashion_{row_idx:07d}"))
        img_dir = IMAGES_DIR / category
        img_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / f"{item_id}.jpg"
        if not img_path.exists():
            pil_img = item["image"]
            if not isinstance(pil_img, PILImage.Image):
                pil_img = PILImage.fromarray(pil_img)
            pil_img.convert("RGB").save(img_path, format="JPEG", quality=90)
            saved += 1

        docs.append({
            "path": str(img_path),
            "id": f"fashion200k_{item_id}",
            "category": category,
            "category2": item.get("category2", ""),
            "category3": item.get("category3", ""),
            "text": item.get("text", ""),
        })

        if len(docs) % 10_000 == 0:
            print(f"  {len(docs)}/{MAX_IMAGES} procesadas ({saved} guardadas)...")

    print(f"  Guardadas {saved} imagenes nuevas en {IMAGES_DIR}/")
    return docs


def write_manifest(docs: list[dict], n: int, label: str) -> None:
    selected = docs[:n]
    out = DATA_DIR / f"image_{label}.json"
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    cats_repr = len(set(d["category"] for d in selected))
    print(f"  [{label.upper()}] {len(selected)} imagenes, {cats_repr} categorias -> {out.name}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_and_save_fashion200k()

    for label, n in SCALES.items():
        out_path = DATA_DIR / f"image_{label}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing and str(existing[0].get("id", "")).startswith("fashion200k_"):
                print(f"  [{label.upper()}] Ya existe con Fashion200K, saltando.")
                continue
        write_manifest(docs, n, label)

    print(f"\nManifests listos en {DATA_DIR}/")
    print(f"Imagenes en {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
