import numpy as np
import pytest

from backend.src.index.audio_search import AudioSearchIndex


def test_audio_search_ranks_by_lowest_l2_distance() -> None:
    """Audio search should return nearest histograms first."""
    index = AudioSearchIndex()
    index.add_record(
        {
            "chunk_id": "exact",
            "modality": "audio",
            "histogram": [0, 1],
            "metadata": {"song": "demo"},
        }
    )
    index.add_record({"chunk_id": "near", "modality": "audio", "histogram": [0, 2]})
    index.add_record({"chunk_id": "far", "modality": "audio", "histogram": [0, 7]})

    results = index.search(np.array([0, 1]), k=2)

    assert [result.chunk_id for result in results] == ["exact", "near"]
    assert results[0].score == pytest.approx(0.0)
    assert results[0].metadata == {"song": "demo"}


def test_audio_search_respects_top_k_limit() -> None:
    """Audio search should limit results to k."""
    index = AudioSearchIndex()
    index.add_record({"chunk_id": "a", "modality": "audio", "histogram": [0, 0]})
    index.add_record({"chunk_id": "b", "modality": "audio", "histogram": [1, 0]})

    results = index.search(np.array([0, 0]), k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_audio_search_rejects_non_audio_records() -> None:
    """Audio indexes should only accept audio histograms."""
    index = AudioSearchIndex()

    with pytest.raises(ValueError, match="audio histograms"):
        index.add_record({"chunk_id": "img", "modality": "image", "histogram": [1]})


def test_audio_search_rejects_duplicate_chunk_id() -> None:
    """Chunk ids should be unique in the audio index."""
    index = AudioSearchIndex()
    record = {"chunk_id": "audio", "modality": "audio", "histogram": [1, 2]}
    index.add_record(record)

    with pytest.raises(ValueError, match="already indexed"):
        index.add_record(record)


def test_audio_search_rejects_query_with_different_shape() -> None:
    """Query histograms should match indexed audio histogram shape."""
    index = AudioSearchIndex()
    index.add_record({"chunk_id": "audio", "modality": "audio", "histogram": [1, 2]})

    with pytest.raises(ValueError, match="match indexed"):
        index.search(np.array([1, 2, 3]))
