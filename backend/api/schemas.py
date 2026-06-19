from typing import Any, Literal

from pydantic import BaseModel, Field


class HistogramPayload(BaseModel):
    """Histogram record accepted from the codebook module."""

    chunk_id: str = Field(min_length=1)
    modality: Literal["text", "image", "audio"]
    histogram: list[float] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Request body for histogram-based retrieval."""

    records: list[HistogramPayload]
    query_histogram: list[float] = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class SearchResultResponse(BaseModel):
    """API response for one retrieved chunk."""

    chunk_id: str
    score: float
    metadata: dict[str, Any]
