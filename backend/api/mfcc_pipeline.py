import io
from pathlib import Path

import numpy as np
import librosa
from sklearn.cluster import MiniBatchKMeans

from backend.src.extractor.mfcc import MFCCExtractor
from backend.src.index.audio_search import AudioSearchIndex
from backend.src.split.split_audio import SplitAudio

N_CLUSTERS = 50
MAX_FILES = 150
SUPPORTED = {".wav", ".mp3", ".ogg", ".flac"}


class MFCCPipeline:
    def __init__(self) -> None:
        self._splitter = SplitAudio(sample_rate=22050, window_seconds=3.0, hop_seconds=1.5)
        self._extractor = MFCCExtractor(n_mfcc=20, sample_rate=22050)
        self._kmeans: MiniBatchKMeans | None = None
        self._index = AudioSearchIndex()
        self.ready = False
        self.indexed_files = 0

    def index_directory(self, audio_dir: Path) -> None:
        audio_files: list[Path] = []
        for ext in SUPPORTED:
            audio_files.extend(audio_dir.rglob(f"*{ext}"))
        audio_files = sorted(audio_files)[:MAX_FILES]

        if not audio_files:
            return

        per_file: list[dict] = []
        all_frames: list[np.ndarray] = []

        for audio_file in audio_files:
            try:
                chunks = self._splitter.split_file(str(audio_file), document_id=audio_file.stem)
                windows = [c["content"] for c in chunks]
                frames = self._extractor.extract(windows)  # (M, n_mfcc)
                if frames.shape[0] == 0:
                    continue
                all_frames.append(frames)
                genre = audio_file.parent.name
                per_file.append({
                    "file_id": audio_file.stem,
                    "filename": audio_file.name,
                    "genre": genre,
                    "windows": windows,
                    "audio_url": f"/audio/{genre}/{audio_file.name}",
                })
            except Exception:
                continue

        if not all_frames:
            return

        stacked = np.vstack(all_frames)
        self._kmeans = MiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=42, n_init=3, max_iter=100, batch_size=1024
        )
        self._kmeans.fit(stacked)

        for item in per_file:
            hist = self._build_histogram(item["windows"])
            self._index.add_record({
                "chunk_id": item["file_id"],
                "modality": "audio",
                "histogram": hist.tolist(),
                "metadata": {
                    "filename": item["filename"],
                    "genre": item["genre"],
                    "title": item["file_id"].replace(".", " ").replace("_", " ").title(),
                    "audio_url": item["audio_url"],
                },
            })

        self.ready = True
        self.indexed_files = len(per_file)

    def search(self, audio_bytes: bytes, k: int = 10) -> list[dict]:
        if not self.ready or self._kmeans is None:
            return []

        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
        chunks = self._splitter.split_audio(audio, document_id="query")
        windows = [c["content"] for c in chunks]
        query_hist = self._build_histogram(windows)

        results = self._index.search(query_hist, k=k)
        return [
            {
                "chunk_id": r.chunk_id,
                "score": round(max(0.0, 1.0 - r.score ** 2 / 2.0), 4),
                "metadata": r.metadata,
            }
            for r in results
        ]

    def index_stats(self) -> dict:
        histograms = self._index._histograms
        n = len(histograms)
        dim = int(next(iter(histograms.values())).shape[0]) if n > 0 else 0
        index_mb = round(n * dim * 4 / (1024 * 1024), 3)
        return {"n_comparisons": n, "vector_dim": dim, "index_mb": index_mb}

    def _build_histogram(self, windows: list[np.ndarray]) -> np.ndarray:
        hist = np.zeros(N_CLUSTERS, dtype=np.float32)
        if self._kmeans is None or not windows:
            return hist

        for window in windows:
            frames = self._extractor.extract([window])  # (n_frames, n_mfcc) de una ventana
            if frames.shape[0] == 0:
                continue
            assignments = self._kmeans.predict(frames)
            for c in assignments:
                hist[c] += 1.0

        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist


mfcc_pipeline = MFCCPipeline()
