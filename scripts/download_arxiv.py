"""
Descarga abstracts de ArXiv via su API pública (sin token, sin Kaggle).
Guarda cada paper como .txt en data/full/arxiv/
"""
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "full" / "arxiv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "machine+learning",
    "computer+vision",
    "natural+language+processing",
    "information+retrieval",
    "deep+learning",
]
MAX_PER_QUERY = 80
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_papers(query: str, max_results: int) -> list[dict]:
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{query}&start=0&max_results={max_results}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        id_el = entry.find("atom:id", NS)
        if title_el is None or summary_el is None or id_el is None:
            continue
        papers.append({
            "id": id_el.text.split("/")[-1].replace(".", "_"),
            "title": title_el.text.strip().replace("\n", " "),
            "abstract": summary_el.text.strip().replace("\n", " "),
        })
    return papers


def main() -> None:
    saved = 0
    seen: set[str] = set()

    for query in QUERIES:
        print(f"Descargando '{query}'...", end=" ", flush=True)
        try:
            papers = fetch_papers(query, MAX_PER_QUERY)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        count = 0
        for p in papers:
            if p["id"] in seen:
                continue
            seen.add(p["id"])
            path = OUTPUT_DIR / f"{p['id']}.txt"
            path.write_text(
                f"{p['title']}\n\n{p['abstract']}",
                encoding="utf-8",
            )
            count += 1
            saved += 1

        print(f"{count} papers  (total {saved})")
        time.sleep(1)  # respeta rate-limit del API

    print(f"\nListo: {saved} papers guardados en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
