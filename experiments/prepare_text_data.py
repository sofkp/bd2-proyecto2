"""
Prepara manifests de texto para benchmarks Fase 4.

Dataset: AG News (fancyzhx/ag_news) — 120K artículos de noticias, 4 categorías.
Categorías: World (0), Sports (1), Business (2), Sci/Tech (3)

Genera:
  experiments/data/text_1k.json   →   1 000 docs  (~1 000 chunks)
  experiments/data/text_10k.json  →  10 000 docs  (~10 000 chunks)
  experiments/data/text_100k.json → 100 000 docs  (~100 000 chunks)

Uso:
    python experiments/prepare_text_data.py
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SCALES = {
    "1k":   1_000,
    "10k":  10_000,
    "100k": 100_000,
}

SEED = 42


def load_agnews() -> list[dict]:
    """Carga AG News completo desde HuggingFace (cacheado localmente)."""
    from datasets import load_dataset

    print("Cargando AG News (fancyzhx/ag_news)...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    label_names = ds.features["label"].names   # ['World','Sports','Business','Sci/Tech']

    docs = []
    for i, item in enumerate(ds):
        docs.append({
            "text":     item["text"],
            "source":   f"agnews_{i}",
            "category": label_names[item["label"]],
        })
    print(f"  → {len(docs)} documentos cargados, categorías: {label_names}")
    return docs


def write_manifest(docs: list[dict], n: int, label: str) -> None:
    """Selecciona n docs balanceados por categoría y guarda el manifest."""
    random.seed(SEED)

    # Agrupar por categoría para muestreo balanceado
    by_cat: dict[str, list[dict]] = {}
    for d in docs:
        cat = d["category"]
        by_cat.setdefault(cat, []).append(d)

    n_cats  = len(by_cat)
    per_cat = n // n_cats        # docs por categoría
    remainder = n - per_cat * n_cats

    selected: list[dict] = []
    for i, (cat, items) in enumerate(sorted(by_cat.items())):
        take = per_cat + (1 if i < remainder else 0)
        take = min(take, len(items))
        selected.extend(random.sample(items, take))

    random.shuffle(selected)

    out = DATA_DIR / f"text_{label}.json"
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    print(f"  [{label.upper()}] {len(selected)} docs → {out.name}")


def main() -> None:
    docs = load_agnews()

    for label, n in SCALES.items():
        out_path = DATA_DIR / f"text_{label}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            # Solo reescribir si no es el mismo dataset (AG News, no ArXiv/newsgroups)
            if existing and "source" in existing[0] and existing[0]["source"].startswith("agnews"):
                print(f"  [{label.upper()}] Ya existe con AG News, saltando.")
                continue
        write_manifest(docs, n, label)

    print("\nManifests listos en experiments/data/")


if __name__ == "__main__":
    main()
