import numpy as np
import pytest

from backend.src.index.visual_search import VisualSearchIndex


def test_visual_search_ranks_by_lowest_l2_distance() -> None:
    """Visual search should return nearest histograms first."""
    index = VisualSearchIndex()
    index.add_record(
        {
            "chunk_id": "exact",
            "modality": "image",
            "histogram": [1, 0],
            "metadata": {"product": "shoe"},
        }
    )
    index.add_record({"chunk_id": "near", "modality": "image", "histogram": [2, 0]})
    index.add_record({"chunk_id": "far", "modality": "image", "histogram": [5, 0]})

    results = index.search(np.array([1, 0]), k=2)

    assert [result.chunk_id for result in results] == ["exact", "near"]
    assert results[0].score == pytest.approx(0.0)
    assert results[0].metadata == {"product": "shoe"}


def test_visual_search_respects_top_k_limit() -> None:
    """Visual search should limit results to k."""
    index = VisualSearchIndex()
    index.add_record({"chunk_id": "a", "modality": "image", "histogram": [0, 0]})
    index.add_record({"chunk_id": "b", "modality": "image", "histogram": [1, 0]})

    results = index.search(np.array([0, 0]), k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_visual_search_rejects_non_image_records() -> None:
    """Visual indexes should only accept image histograms."""
    index = VisualSearchIndex()

    with pytest.raises(ValueError, match="image histograms"):
        index.add_record({"chunk_id": "text", "modality": "text", "histogram": [1]})


def test_visual_search_rejects_duplicate_chunk_id() -> None:
    """Chunk ids should be unique in the visual index."""
    index = VisualSearchIndex()
    record = {"chunk_id": "img", "modality": "image", "histogram": [1, 2]}
    index.add_record(record)

    with pytest.raises(ValueError, match="already indexed"):
        index.add_record(record)


def test_visual_search_rejects_query_with_different_shape() -> None:
    """Query histograms should match indexed visual histogram shape."""
    index = VisualSearchIndex()
    index.add_record({"chunk_id": "img", "modality": "image", "histogram": [1, 2]})

    with pytest.raises(ValueError, match="match indexed"):
        index.search(np.array([1, 2, 3]))
