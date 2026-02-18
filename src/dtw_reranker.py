"""DTW-based reranking utilities for melody contours."""

from __future__ import annotations

import numpy as np


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, window_ratio: float = 0.2) -> float:
    a = np.asarray(seq_a, dtype=np.float32)
    b = np.asarray(seq_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return float("inf")

    n, m = int(a.size), int(b.size)
    band = max(abs(n - m), int(max(n, m) * window_ratio))

    inf = np.float32(1e12)
    prev = np.full(m + 1, inf, dtype=np.float32)
    curr = np.full(m + 1, inf, dtype=np.float32)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr.fill(inf)
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return float(prev[m] / float(n + m))


def key_invariant_dtw(
    query_contour: np.ndarray,
    candidate_contour: np.ndarray,
    transpose_range: int = 6,
    window_ratio: float = 0.2,
) -> tuple[float, int]:
    best_dist = float("inf")
    best_shift = 0
    for shift in range(-transpose_range, transpose_range + 1):
        shifted = candidate_contour + float(shift)
        dist = dtw_distance(query_contour, shifted, window_ratio=window_ratio)
        if dist < best_dist:
            best_dist = dist
            best_shift = shift
    return best_dist, best_shift


def dtw_similarity(distance: float, temperature: float = 3.0) -> float:
    if not np.isfinite(distance):
        return 0.0
    return float(np.exp(-distance / max(temperature, 1e-6)))

