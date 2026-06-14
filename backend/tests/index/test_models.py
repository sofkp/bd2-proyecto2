import numpy as np
import pytest

from backend.src.index.models import (
    SearchResult,
    format_search_results,
    validate_histogram_record,
)


def test_validate_histogram_record_accepts_codebook_contract() -> None:
    """Valid codebook records should become typed histogram records."""
    record = validate_histogram_record(
        {
            "chunk_id": "text_doc_0",
            "modality": "text",
            "histogram": [1, 0, 3],
            "metadata": {"source": "doc.txt"},
        }
    )

    assert record.chunk_id == "text_doc_0"
    assert record.modality == "text"
    assert np.array_equal(record.histogram, np.array([1.0, 0.0, 3.0]))
    assert record.metadata == {"source": "doc.txt"}


def test_validate_histogram_record_rejects_unknown_modality() -> None:
    """Only text, image, and audio modalities are allowed."""
    with pytest.raises(ValueError, match="modality"):
        validate_histogram_record(
            {
                "chunk_id": "video_0",
                "modality": "video",
                "histogram": [1, 2],
            }
        )


def test_validate_histogram_record_rejects_2d_histogram() -> None:
    """Histograms must use the 1D codebook output contract."""
    with pytest.raises(ValueError, match="1D"):
        validate_histogram_record(
            {
                "chunk_id": "image_0",
                "modality": "image",
                "histogram": [[1, 2]],
            }
        )


def test_format_search_results_matches_api_contract() -> None:
    """Search results should serialize to the API output contract."""
    formatted = format_search_results(
        [SearchResult("audio_1", 0.8, {"title": "song"})]
    )

    assert formatted == [
        {
            "chunk_id": "audio_1",
            "score": 0.8,
            "metadata": {"title": "song"},
        }
    ]
