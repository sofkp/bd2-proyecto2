from pathlib import Path

import numpy as np
import pandas as pd

from backend.src.codebook.codebook_text import CodebookText
from backend.src.extractor.tfidf import TFIDFExtractor
from backend.src.index.audio_search import AudioSearchIndex
from backend.src.index.inverted_index import InvertedIndex
from backend.src.split.split_text import SplitText

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms",
]
MAX_SONGS = 1500


class MusicPipeline:
    def __init__(self) -> None:
        self._splitter = SplitText()
        self._extractor = TFIDFExtractor(language="english")
        self._codebook = CodebookText(top_k=200)
        self._lyrics_index = InvertedIndex()
        self._audio_index = AudioSearchIndex()
        self._vocab: dict = {}
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None
        self.ready = False
        self.indexed_songs = 0

    def load_csv(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path)
        df = df[df["language"] == "en"].dropna(subset=["lyrics"])
        df = df.drop_duplicates(subset=["track_id"]).head(MAX_SONGS).reset_index(drop=True)

        self._build_lyrics_pipeline(df)
        self._build_audio_pipeline(df)

        self.ready = True
        self.indexed_songs = len(df)

    def search_by_lyrics(self, query: str, k: int = 10) -> list[dict]:
        if not self.ready:
            return []
        feat = self._extractor.extract([{"doc_id": "q", "chunk_id": "q0", "text": query}])
        hist = self._to_histogram(feat[0]["tf"])
        raw = self._lyrics_index.search(hist, k=k * 3)

        seen: set[str] = set()
        results: list[dict] = []
        for r in raw:
            track_id = r.chunk_id.split("_text_")[0]
            if track_id not in seen:
                seen.add(track_id)
                results.append({
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "metadata": r.metadata,
                })
            if len(results) == k:
                break
        return results

    def search_by_audio_features(self, features: list[float], k: int = 10) -> list[dict]:
        if not self.ready:
            return []
        arr = self._normalize_audio(np.array(features, dtype=float))
        results = self._audio_index.search(arr, k=k)
        return [
            {"chunk_id": r.chunk_id, "score": round(r.score, 4), "metadata": r.metadata}
            for r in results
        ]

    def audio_feature_names(self) -> list[str]:
        return AUDIO_FEATURES

    def index_stats(self) -> dict:
        histograms = self._lyrics_index._histograms
        n = len(histograms)
        dim = int(next(iter(histograms.values())).shape[0]) if n > 0 else 0
        index_mb = round(n * dim * 4 / (1024 * 1024), 3)
        return {"n_comparisons": n, "vector_dim": dim, "index_mb": index_mb}

    def _build_lyrics_pipeline(self, df: pd.DataFrame) -> None:
        all_chunks: list[dict] = []
        meta_map: dict[str, dict] = {}

        for _, row in df.iterrows():
            meta_map[str(row["track_id"])] = {
                "track_name": row["track_name"],
                "artist": row["track_artist"],
                "genre": row["playlist_genre"],
                "subgenre": row["playlist_subgenre"],
            }
            chunks = self._splitter.split_text(
                text=str(row["lyrics"]),
                document_id=str(row["track_id"]),
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        extracted = self._extractor.extract(all_chunks)
        self._vocab = self._codebook.build_codebook(extracted)

        for chunk, feat in zip(all_chunks, extracted):
            track_id = chunk["doc_id"]
            song = meta_map.get(track_id, {})
            hist = self._to_histogram(feat["tf"])
            self._lyrics_index.add_record({
                "chunk_id": chunk["chunk_id"],
                "modality": "text",
                "histogram": hist.tolist(),
                "metadata": {
                    "track_name": song.get("track_name", ""),
                    "artist": song.get("artist", ""),
                    "genre": song.get("genre", ""),
                    "subgenre": song.get("subgenre", ""),
                    "snippet": chunk["content"][:200],
                },
            })

    def _build_audio_pipeline(self, df: pd.DataFrame) -> None:
        feat_df = df[AUDIO_FEATURES].astype(float)
        self._feat_min = feat_df.min().values
        self._feat_max = feat_df.max().values

        for _, row in df.iterrows():
            raw = row[AUDIO_FEATURES].values.astype(float)
            norm = self._normalize_audio(raw)
            self._audio_index.add_record({
                "chunk_id": str(row["track_id"]),
                "modality": "audio",
                "histogram": norm.tolist(),
                "metadata": {
                    "track_name": row["track_name"],
                    "artist": row["track_artist"],
                    "genre": row["playlist_genre"],
                    "tempo": round(float(row["tempo"]), 1),
                    "energy": round(float(row["energy"]), 3),
                    "danceability": round(float(row["danceability"]), 3),
                    "valence": round(float(row["valence"]), 3),
                },
            })

    def _to_histogram(self, tf: dict) -> np.ndarray:
        hist = np.zeros(len(self._vocab), dtype=np.float32)
        for word, count in tf.items():
            if word in self._vocab:
                idx = self._vocab[word]["index"]
                hist[idx] = count * self._vocab[word]["idf"]
        return hist

    def _normalize_audio(self, values: np.ndarray) -> np.ndarray:
        denom = self._feat_max - self._feat_min
        denom[denom == 0] = 1.0
        return ((values - self._feat_min) / denom).astype(np.float32)


music_pipeline = MusicPipeline()
