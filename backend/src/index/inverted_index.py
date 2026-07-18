from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.src.index.metrics import cosine_similarity, top_k
from backend.src.index.models import HistogramRecord, SearchResult, validate_histogram_record


@dataclass(frozen=True)
class Posting:
    """Frequency of a codeword inside a chunk."""

    chunk_id: str
    frequency: float


class InvertedIndex:
    """In-memory inverted index for text codeword histograms."""

    def __init__(self) -> None:
        self._postings: dict[int, list[Posting]] = {}
        self._histograms: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    def add_record(self, record: dict) -> None:
        """Add one codebook histogram record to the index."""
        self.add_histogram(validate_histogram_record(record))

    def build_with_spimi(
        self,
        records: list[dict],
        block_size: int = 1000,
        work_dir: str | Path | None = None,
    ) -> int:
        """Build postings through SPIMI blocks persisted in secondary storage."""
        from backend.src.index.spimi import SpimiIndexer

        self._postings = {}
        self._histograms = {}
        self._metadata = {}

        validated = [validate_histogram_record(record) for record in records]
        for record in validated:
            if record.modality != "text":
                raise ValueError("inverted index only accepts text histograms")
            if record.chunk_id in self._histograms:
                raise ValueError(f"chunk_id already indexed: {record.chunk_id}")
            self._histograms[record.chunk_id] = record.histogram
            self._metadata[record.chunk_id] = record.metadata

        indexer = SpimiIndexer(block_size=block_size)
        if work_dir is not None:
            block_files = indexer.create_block_files(records, work_dir)
            self._postings = indexer.merge_block_files(block_files)
            return len(block_files)

        with TemporaryDirectory(prefix="spimi_blocks_") as tmp:
            block_files = indexer.create_block_files(records, tmp)
            self._postings = indexer.merge_block_files(block_files)
            return len(block_files)

    def add_histogram(self, record: HistogramRecord) -> None:
        """Index one validated text histogram."""
        if record.modality != "text":
            raise ValueError("inverted index only accepts text histograms")
        if record.chunk_id in self._histograms:
            raise ValueError(f"chunk_id already indexed: {record.chunk_id}")

        self._histograms[record.chunk_id] = record.histogram
        self._metadata[record.chunk_id] = record.metadata
        for codeword, frequency in enumerate(record.histogram):
            if frequency > 0:
                self._postings.setdefault(codeword, []).append(
                    Posting(record.chunk_id, float(frequency))
                )

    def get_postings(self, codeword: int) -> list[Posting]:
        """Return postings for one codeword."""
        return list(self._postings.get(codeword, []))

    def candidate_chunk_ids(self, query_histogram: np.ndarray) -> set[str]:
        """Return chunks sharing at least one non-zero query codeword."""
        query = np.asarray(query_histogram, dtype=float)
        if query.ndim != 1:
            raise ValueError("query_histogram must be a 1D array")

        return {
            posting.chunk_id
            for codeword, frequency in enumerate(query)
            if frequency > 0
            for posting in self.get_postings(codeword)
        }

    def get_histogram(self, chunk_id: str) -> np.ndarray:
        """Return the stored histogram for a chunk."""
        return self._histograms[chunk_id]

    def get_metadata(self, chunk_id: str) -> dict:
        """Return metadata stored for a chunk."""
        return self._metadata.get(chunk_id, {})

    def search(self, query_histogram: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Search text chunks ranked by cosine similarity."""
        query = self._validate_query(query_histogram)
        candidates = self.candidate_chunk_ids(query)

        ranked = top_k(
            candidates,
            score_fn=lambda chunk_id: cosine_similarity(
                query, self.get_histogram(chunk_id)
            ),
            k=k,
        )
        return [
            SearchResult(chunk_id, score, self.get_metadata(chunk_id))
            for chunk_id, score in ranked
            if score > 0
        ]

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._histograms)

    def _validate_query(self, query_histogram: np.ndarray) -> np.ndarray:
        query = np.asarray(query_histogram, dtype=float)
        if query.ndim != 1:
            raise ValueError("query_histogram must be a 1D array")
        if not self._histograms:
            return query
        indexed_shape = next(iter(self._histograms.values())).shape
        if query.shape != indexed_shape:
            raise ValueError("query_histogram must match indexed histogram shape")
        return query
