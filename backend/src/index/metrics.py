from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of a numeric vector."""
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values)
    if norm == 0:
        return np.zeros_like(values, dtype=float)
    return values / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    _validate_same_shape(left_values, right_values)

    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_values, right_values) / denominator)


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    _validate_same_shape(left_values, right_values)
    return float(np.linalg.norm(left_values - right_values))


def top_k(
    items: Iterable[T],
    score_fn: Callable[[T], float],
    k: int = 10,
    reverse: bool = True,
) -> list[tuple[T, float]]:
    """Return the top-k items paired with their score."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scored_items = [(item, float(score_fn(item))) for item in items]
    return sorted(scored_items, key=lambda pair: pair[1], reverse=reverse)[:k]


def _validate_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError("vectors must have the same shape")
    if left.ndim != 1:
        raise ValueError("vectors must be 1D arrays")
