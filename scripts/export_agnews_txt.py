"""
Convierte el manifest de AG News (generado por experiments/prepare_text_data.py)
en archivos .txt individuales dentro de data/full/agnews/, en el mismo formato
que ya usa la app en vivo para texto (primera línea = título/categoría, resto =
contenido) — así el backend los indexa automáticamente al arrancar, igual que
data/full/arxiv.

Requiere haber corrido antes:
    python experiments/prepare_text_data.py

Uso:
    python scripts/export_agnews_txt.py
"""
import json
from pathlib import Path

MANIFEST = Path(__file__).parent.parent / "experiments" / "data" / "text_1k.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "full" / "agnews"


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(
            f"No existe {MANIFEST}.\nCorre primero:\n"
            f"  python experiments/prepare_text_data.py"
        )

    docs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        out_path = OUTPUT_DIR / f"{doc['source']}.txt"
        out_path.write_text(f"{doc['category']}\n\n{doc['text']}", encoding="utf-8")

    print(f"Listo: {len(docs)} artículos de AG News exportados a {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
