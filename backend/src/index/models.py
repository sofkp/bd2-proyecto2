from dataclasses import dataclass, field
from typing import Any

import numpy as np

VALID_MODALITIES = {"text", "image", "audio"}


@dataclass(frozen=True)
class HistogramRecord:
    """Validated histogram produced by the codebook module."""

    chunk_id: str
    modality: str
    histogram: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Result returned by retrieval modules to the API layer."""

    chunk_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the API-compatible representation of the result."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "metadata": self.metadata,
        }


def validate_histogram_record(record: dict[str, Any]) -> HistogramRecord:
    """Validate and normalize a histogram record from the codebook module."""
    chunk_id = record.get("chunk_id")
    modality = record.get("modality")
    histogram = record.get("histogram")
    metadata = record.get("metadata", {})

    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id must be a non-empty string")
    if modality not in VALID_MODALITIES:
        raise ValueError("modality must be one of: audio, image, text")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    histogram_array = np.asarray(histogram, dtype=float)
    if histogram_array.ndim != 1 or histogram_array.size == 0:
        raise ValueError("histogram must be a non-empty 1D array")
    if not np.all(np.isfinite(histogram_array)):
        raise ValueError("histogram values must be finite numbers")

    return HistogramRecord(
        chunk_id=chunk_id,
        modality=modality,
        histogram=histogram_array,
        metadata=metadata,
    )


def format_search_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Convert search results into the API response contract."""
    return [result.to_dict() for result in results]
