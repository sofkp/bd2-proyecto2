import numpy as np
import pytest

from backend.src.index.inverted_index import InvertedIndex


def test_add_record_builds_postings_for_non_zero_codewords() -> None:
    """Only non-zero codeword frequencies should create postings."""
    index = InvertedIndex()

    index.add_record(
        {
            "chunk_id": "text_doc_0",
            "modality": "text",
            "histogram": [2, 0, 1],
            "metadata": {"title": "Doc"},
        }
    )

    assert len(index) == 1
    assert index.get_postings(0)[0].chunk_id == "text_doc_0"
    assert index.get_postings(0)[0].frequency == 2.0
    assert index.get_postings(1) == []
    assert index.get_metadata("text_doc_0") == {"title": "Doc"}


def test_candidate_chunk_ids_returns_matching_chunks() -> None:
    """Candidates should share at least one non-zero query codeword."""
    index = InvertedIndex()
    index.add_record({"chunk_id": "a", "modality": "text", "histogram": [1, 0, 0]})
    index.add_record({"chunk_id": "b", "modality": "text", "histogram": [0, 2, 0]})
    index.add_record({"chunk_id": "c", "modality": "text", "histogram": [0, 0, 3]})

    candidates = index.candidate_chunk_ids(np.array([1, 1, 0]))

    assert candidates == {"a", "b"}


def test_get_histogram_returns_stored_vector() -> None:
    """Indexed histograms should be retrievable by chunk id."""
    index = InvertedIndex()
    index.add_record({"chunk_id": "a", "modality": "text", "histogram": [1, 2]})

    assert np.array_equal(index.get_histogram("a"), np.array([1.0, 2.0]))


def test_add_record_rejects_non_text_modality() -> None:
    """The inverted index should only receive text histograms."""
    index = InvertedIndex()

    with pytest.raises(ValueError, match="text histograms"):
        index.add_record(
            {"chunk_id": "image_0", "modality": "image", "histogram": [1, 2]}
        )


def test_add_record_rejects_duplicate_chunk_id() -> None:
    """Chunk ids must be unique inside one index."""
    index = InvertedIndex()
    record = {"chunk_id": "a", "modality": "text", "histogram": [1, 2]}
    index.add_record(record)

    with pytest.raises(ValueError, match="already indexed"):
        index.add_record(record)
