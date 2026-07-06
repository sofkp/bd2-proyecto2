import io
from pathlib import Path
from urllib.parse import quote

import numpy as np
import librosa

from backend.src.codebook.codebook_kmeans import VectorCodebook
from backend.src.extractor.mfcc import MFCCExtractor
from backend.src.index.audio_search import AudioSearchIndex
from backend.src.split.split_audio import SplitAudio

N_CLUSTERS = 512
SUPPORTED = {".wav", ".mp3", ".ogg", ".flac"}
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_SAMPLES_DIR = DATA_DIR / "samples"
DATA_FULL_DIR = DATA_DIR / "full"


class MFCCPipeline:
    def __init__(self) -> None:
        self._splitter = SplitAudio(sample_rate=22050, window_seconds=3.0, hop_seconds=1.5)
        self._extractor = MFCCExtractor(n_mfcc=20, sample_rate=22050)
        self._codebook: VectorCodebook | None = None
        self._index = AudioSearchIndex()
        self.ready = False
        self.indexed_files = 0

    def index_directory(self, audio_dir: Path) -> None:
        self.index_directories([audio_dir])

    def index_directories(self, audio_dirs: list[Path], max_files: int | None = None) -> None:
        audio_files: list[Path] = []
        for audio_dir in audio_dirs:
            found = sorted(
                f for f in audio_dir.rglob("*") if f.suffix.lower() in SUPPORTED
            )
            audio_files.extend(found)
            if max_files is not None and len(audio_files) >= max_files:
                audio_files = audio_files[:max_files]
                break

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
                file_id = self._file_id(audio_file)
                per_file.append({
                    "file_id": file_id,
                    "filename": audio_file.name,
                    "genre": genre,
                    "windows": windows,
                    "audio_url": self._audio_url(audio_file),
                })
            except Exception:
                continue

        if not all_frames:
            return

        stacked = np.vstack(all_frames)
        self._codebook = VectorCodebook(
            n_clusters=N_CLUSTERS, random_state=42, minibatch=True,
            n_init=3, max_iter=100, batch_size=1024,
        )
        self._codebook.build_codebook(stacked)

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
        if not self.ready or self._codebook is None:
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
        if self._codebook is None or not windows:
            return hist

        for window in windows:
            frames = self._extractor.extract([window])  # (n_frames, n_mfcc) de una ventana
            if frames.shape[0] == 0:
                continue
            assignments = self._codebook.predict(frames)
            for c in assignments:
                hist[c] += 1.0

        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def _file_id(self, audio_file: Path) -> str:
        try:
            rel = audio_file.relative_to(DATA_DIR)
            return "_".join(rel.with_suffix("").parts)
        except ValueError:
            return f"{audio_file.parent.name}_{audio_file.stem}"

    def _audio_url(self, audio_file: Path) -> str:
        try:
            rel = audio_file.relative_to(DATA_SAMPLES_DIR / "audio")
            return "/audio-samples/" + quote(rel.as_posix())
        except ValueError:
            pass

        try:
            rel = audio_file.relative_to(DATA_FULL_DIR)
            return "/audio-full/" + quote(rel.as_posix())
        except ValueError:
            return ""


mfcc_pipeline = MFCCPipeline()
