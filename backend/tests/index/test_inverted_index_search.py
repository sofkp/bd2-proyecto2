import numpy as np
import pytest

from backend.src.index.inverted_index import InvertedIndex


def test_search_ranks_candidates_by_cosine_similarity() -> None:
    """Search should return matching chunks ordered by cosine score."""
    index = InvertedIndex()
    index.add_record(
        {
            "chunk_id": "exact",
            "modality": "text",
            "histogram": [2, 0, 0],
            "metadata": {"rank": 1},
        }
    )
    index.add_record({"chunk_id": "partial", "modality": "text", "histogram": [1, 1, 0]})
    index.add_record({"chunk_id": "miss", "modality": "text", "histogram": [0, 0, 3]})

    results = index.search(np.array([1, 0, 0]), k=2)

    assert [result.chunk_id for result in results] == ["exact", "partial"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata == {"rank": 1}


def test_search_respects_top_k_limit() -> None:
    """Search should limit results to the requested k."""
    index = InvertedIndex()
    index.add_record({"chunk_id": "a", "modality": "text", "histogram": [1, 0]})
    index.add_record({"chunk_id": "b", "modality": "text", "histogram": [1, 1]})

    results = index.search(np.array([1, 0]), k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_search_rejects_query_with_different_shape() -> None:
    """Query histograms should match indexed histogram shape."""
    index = InvertedIndex()
    index.add_record({"chunk_id": "a", "modality": "text", "histogram": [1, 2]})

    with pytest.raises(ValueError, match="match indexed"):
        index.search(np.array([1, 2, 3]))
