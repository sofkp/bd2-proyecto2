from typing import Any

import numpy as np

from backend.src.index.metrics import l2_distance, top_k
from backend.src.index.models import HistogramRecord, SearchResult, validate_histogram_record


class VisualSearchIndex:
    """In-memory search index for visual codeword histograms."""

    def __init__(self) -> None:
        self._histograms: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add_record(self, record: dict[str, Any]) -> None:
        """Add one image histogram record to the visual index."""
        self.add_histogram(validate_histogram_record(record))

    def add_histogram(self, record: HistogramRecord) -> None:
        """Store one validated image histogram."""
        if record.modality != "image":
            raise ValueError("visual search only accepts image histograms")
        if record.chunk_id in self._histograms:
            raise ValueError(f"chunk_id already indexed: {record.chunk_id}")

        self._histograms[record.chunk_id] = record.histogram
        self._metadata[record.chunk_id] = record.metadata

    def search(self, query_histogram: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Return image chunks ranked by ascending L2 distance."""
        query = self._validate_query(query_histogram)
        ranked = top_k(
            self._histograms,
            score_fn=lambda chunk_id: -l2_distance(query, self._histograms[chunk_id]),
            k=k,
        )
        return [
            SearchResult(chunk_id, -score, self._metadata.get(chunk_id, {}))
            for chunk_id, score in ranked
        ]

    def __len__(self) -> int:
        """Return the number of indexed image chunks."""
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
