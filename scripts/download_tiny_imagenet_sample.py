"""
Descarga una MUESTRA pequeña de Tiny ImageNet (zh-plus/tiny-imagenet) para
probar la app en vivo, en vez del dataset completo (110K imágenes, ~6-8GB).

Usa streaming: solo baja los bytes de las N imágenes que realmente toma,
sin cachear el dataset completo en disco.

Guarda en data/full/tiny_imagenet/<clase>/<id>.jpg — la misma estructura
que ya lee backend/api/image_pipeline.py (recursivamente).

Uso:
    python scripts/download_tiny_imagenet_sample.py [N]
    (N por defecto: 150, el máximo que la app realmente usa)
"""
import sys
from pathlib import Path

N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "full" / "tiny_imagenet"


def main() -> None:
    from datasets import load_dataset
    from PIL import Image as PILImage

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Descargando {N_IMAGES} imágenes de Tiny ImageNet (streaming, sin cachear el dataset completo)...")
    ds = load_dataset("zh-plus/tiny-imagenet", split="train", streaming=True)
    label_names = ds.features["label"].names  # WordNet IDs
    # El dataset viene ordenado por clase; sin shuffle, tomar las primeras N
    # secuenciales da SIEMPRE la misma clase (ej. todo peces). Se mezcla con
    # un buffer para obtener variedad real de categorías.
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    saved = 0
    for i, item in enumerate(ds):
        if saved >= N_IMAGES:
            break

        class_id = label_names[item["label"]]
        img_dir = OUTPUT_DIR / class_id
        img_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / f"{i:07d}.jpg"
        pil_img = item["image"]
        if not isinstance(pil_img, PILImage.Image):
            pil_img = PILImage.fromarray(pil_img)
        pil_img.convert("RGB").save(img_path, format="JPEG", quality=90)
        saved += 1

    print(f"Listo: {saved} imágenes guardadas en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
