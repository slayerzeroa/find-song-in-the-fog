"""FFT-based similarity scoring between tracks."""

from __future__ import annotations

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


def pitch_overlap_score(notes_a: list[str], notes_b: list[str]) -> float:
    if not notes_a or not notes_b:
        return 0.0
    set_a, set_b = set(notes_a), set(notes_b)
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(inter / max(union, 1))


def combined_similarity(
    query_fft: np.ndarray,
    candidate_fft: np.ndarray,
    query_notes: list[str],
    candidate_notes: list[str],
    fft_weight: float = 0.75,
) -> dict[str, float]:
    fft_score = cosine_similarity(query_fft, candidate_fft)
    pitch_score = pitch_overlap_score(query_notes, candidate_notes)
    total = (fft_weight * fft_score) + ((1.0 - fft_weight) * pitch_score)
    return {
        "fft": fft_score,
        "pitch": pitch_score,
        "total": float(total),
    }
