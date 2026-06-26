from pathlib import Path

import numpy as np

from backend.src.codebook.codebook_text import CodebookText
from backend.src.extractor.tfidf import TFIDFExtractor
from backend.src.index.inverted_index import InvertedIndex
from backend.src.split.split_text import SplitText


class TextPipeline:
    def __init__(self) -> None:
        self._splitter = SplitText()
        self._extractor = TFIDFExtractor(language="english")
        self._codebook = CodebookText(top_k=200)
        self._index = InvertedIndex()
        self._vocab: dict = {}
        self.ready = False
        self.indexed_docs: int = 0
        self.indexed_chunks: int = 0

    def index_directory(self, data_dir: Path) -> None:
        self.index_directories([data_dir])

    def index_directories(self, dirs: list[Path]) -> None:
        txt_files: list[Path] = []
        for d in dirs:
            if d.exists():
                txt_files.extend(sorted(d.glob("*.txt")))

        if not txt_files:
            return

        all_chunks: list[dict] = []
        for txt_file in txt_files:
            chunks = self._splitter.split_file(str(txt_file))
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        extracted = self._extractor.extract(all_chunks)
        self._vocab = self._codebook.build_codebook(extracted)

        doc_titles: dict[str, str] = {}
        for chunk in all_chunks:
            doc_id = chunk.get("doc_id") or chunk["chunk_id"].rsplit("_text_", 1)[0]
            if doc_id not in doc_titles:
                lines = [l for l in chunk["content"].strip().splitlines() if l.strip()]
                first_line = lines[0].strip() if lines else ""
                doc_titles[doc_id] = first_line if len(first_line) > 10 else Path(
                    chunk["metadata"].get("source_path", "")).stem

        self._index = InvertedIndex()
        for chunk, feat in zip(all_chunks, extracted):
            histogram = self._build_histogram(feat["tf"])
            doc_id = chunk.get("doc_id") or chunk["chunk_id"].rsplit("_text_", 1)[0]
            title = doc_titles.get(doc_id, Path(chunk["metadata"].get("source_path", "")).stem)

            content = chunk["content"]
            lines = [l for l in content.strip().splitlines() if l.strip()]

            if lines and lines[0].strip() == title:
                snippet = " ".join(lines[1:]).strip()[:300]
            else:
                snippet = content.strip()[:300]
            if not snippet:
                snippet = content.strip()[:300]

            self._index.add_record({
                "chunk_id": chunk["chunk_id"],
                "modality": "text",
                "histogram": histogram.tolist(),
                "metadata": {
                    "title": title,
                    "snippet": snippet,
                    "source": Path(chunk["metadata"].get("source_path", "")).stem,
                    "doc_id": doc_id,
                },
            })

        self.ready = True
        self.indexed_docs = len(txt_files)
        self.indexed_chunks = len(all_chunks)

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not self.ready:
            return []

        query_feat = self._extractor.extract([
            {"doc_id": "query", "chunk_id": "query_0", "text": query}
        ])
        query_hist = self._build_histogram(query_feat[0]["tf"])
        raw = self._index.search(query_hist, k=k * 3)

        seen: set[str] = set()
        results: list[dict] = []
        for r in raw:
            doc_id = r.metadata.get("doc_id") or r.chunk_id.rsplit("_text_", 1)[0]
            if doc_id not in seen:
                seen.add(doc_id)
                results.append({
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "metadata": r.metadata,
                })
            if len(results) == k:
                break

        return results

    def index_stats(self) -> dict:
        histograms = self._index._histograms
        n = len(histograms)
        dim = int(next(iter(histograms.values())).shape[0]) if n > 0 else 0
        index_mb = round(n * dim * 4 / (1024 * 1024), 3)
        return {"n_comparisons": n, "vector_dim": dim, "index_mb": index_mb}

    def _build_histogram(self, tf: dict) -> np.ndarray:
        size = len(self._vocab)
        hist = np.zeros(size, dtype=np.float32)
        for word, count in tf.items():
            if word in self._vocab:
                idx = self._vocab[word]["index"]
                idf = self._vocab[word]["idf"]
                hist[idx] = count * idf
        return hist


text_pipeline = TextPipeline()
