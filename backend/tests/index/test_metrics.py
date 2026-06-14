import numpy as np
import pytest

from backend.src.index.metrics import (
    cosine_similarity,
    l2_distance,
    normalize_vector,
    top_k,
)


def test_cosine_similarity_returns_expected_score() -> None:
    """Cosine similarity should compare vectors by orientation."""
    score = cosine_similarity(np.array([1, 0]), np.array([1, 1]))

    assert score == pytest.approx(0.70710678)


def test_cosine_similarity_returns_zero_for_empty_direction() -> None:
    """Zero vectors should not raise division errors."""
    score = cosine_similarity(np.array([0, 0]), np.array([1, 1]))

    assert score == 0.0


def test_l2_distance_returns_expected_distance() -> None:
    """L2 distance should compute Euclidean distance."""
    distance = l2_distance(np.array([1, 2]), np.array([4, 6]))

    assert distance == pytest.approx(5.0)


def test_normalize_vector_keeps_zero_vector() -> None:
    """Zero vectors should remain zero after normalization."""
    normalized = normalize_vector(np.array([0, 0]))

    assert np.array_equal(normalized, np.array([0.0, 0.0]))


def test_top_k_returns_best_items_with_scores() -> None:
    """Top-k should sort by descending score by default."""
    results = top_k(["a", "bbb", "cc"], score_fn=len, k=2)

    assert results == [("bbb", 3.0), ("cc", 2.0)]


def test_metrics_reject_vectors_with_different_shapes() -> None:
    """Vector metrics should reject incompatible inputs."""
    with pytest.raises(ValueError, match="same shape"):
        cosine_similarity(np.array([1, 2]), np.array([1, 2, 3]))
