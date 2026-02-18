"""Melody contour extraction and embedding utilities for QBH (query-by-humming)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from audio_analyzer import load_audio


@dataclass
class SegmentConfig:
    segment_seconds: float = 12.0
    segment_hop_seconds: float = 6.0
    min_voiced_ratio: float = 0.35
    max_segments_per_song: int = 4
    embedding_dim: int = 128
    contour_max_len: int = 512
    energy_percentile: float = 20.0
    frame_length: int = 2048
    hop_length: int = 256


@dataclass
class MelodySegment:
    start_sec: float
    end_sec: float
    contour: np.ndarray
    embedding: np.ndarray


@dataclass
class QueryMelody:
    contour: np.ndarray
    embedding: np.ndarray


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if values.size == target_len:
        return values.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _unit_norm(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values) + 1e-12
    return (values / norm).astype(np.float32, copy=False)


def _normalize_pitch(midi_seq: np.ndarray) -> np.ndarray:
    if midi_seq.size == 0:
        return midi_seq.astype(np.float32)
    center = np.median(midi_seq)
    normalized = midi_seq - center
    return normalized.astype(np.float32)


def _trim_and_downsample(contour: np.ndarray, max_len: int) -> np.ndarray:
    if contour.size <= max_len:
        return contour.astype(np.float32, copy=False)
    return _resample_1d(contour, max_len)


def build_embedding_from_contour(contour: np.ndarray, embedding_dim: int) -> np.ndarray:
    base = _resample_1d(contour, embedding_dim)
    delta = np.diff(base, prepend=base[:1])

    # Blend level contour + local pitch movement for robust melody matching.
    embedding = np.concatenate([base, delta], axis=0)
    embedding = embedding.astype(np.float32)
    return _unit_norm(embedding)


def extract_pitch_midi_sequence(y: np.ndarray, sr: int, cfg: SegmentConfig) -> np.ndarray:
    f0 = librosa.yin(
        y,
        fmin=65.0,
        fmax=1200.0,
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length)[0]
    if rms.size != f0.size:
        rms = _resample_1d(rms.astype(np.float32), f0.size)

    midi = librosa.hz_to_midi(f0)
    midi = np.asarray(midi, dtype=np.float32)
    midi[~np.isfinite(midi)] = np.nan

    energy_floor = np.percentile(rms, cfg.energy_percentile)
    voiced_mask = np.isfinite(midi) & (f0 > 0) & (rms >= energy_floor)
    midi[~voiced_mask] = np.nan
    return midi


def extract_song_segments(path: str | Path, cfg: SegmentConfig) -> list[MelodySegment]:
    y, sr = load_audio(path)
    midi = extract_pitch_midi_sequence(y, sr, cfg)

    frame_sec = cfg.hop_length / float(sr)
    seg_frames = max(1, int(round(cfg.segment_seconds / frame_sec)))
    hop_frames = max(1, int(round(cfg.segment_hop_seconds / frame_sec)))

    segments: list[MelodySegment] = []
    for start in range(0, max(1, midi.size - seg_frames + 1), hop_frames):
        end = min(midi.size, start + seg_frames)
        window = midi[start:end]
        voiced = window[np.isfinite(window)]
        if window.size == 0:
            continue
        voiced_ratio = float(voiced.size) / float(window.size)
        if voiced_ratio < cfg.min_voiced_ratio or voiced.size < 8:
            continue

        contour = _normalize_pitch(voiced)
        contour = _trim_and_downsample(contour, cfg.contour_max_len)
        embedding = build_embedding_from_contour(contour, cfg.embedding_dim)

        segments.append(
            MelodySegment(
                start_sec=float(start * frame_sec),
                end_sec=float(end * frame_sec),
                contour=contour,
                embedding=embedding,
            )
        )

    if not segments:
        return []

    # Keep top-N segments by contour variability to prioritize melodic parts.
    scored = []
    for seg in segments:
        variability = float(np.std(seg.contour))
        scored.append((variability, seg))
    scored.sort(key=lambda item: item[0], reverse=True)
    limited = [seg for _, seg in scored[: cfg.max_segments_per_song]]
    return sorted(limited, key=lambda seg: seg.start_sec)


def extract_query_melody(path: str | Path, cfg: SegmentConfig) -> QueryMelody:
    y, sr = load_audio(path)
    midi = extract_pitch_midi_sequence(y, sr, cfg)
    voiced = midi[np.isfinite(midi)]
    if voiced.size < 8:
        raise ValueError("Not enough voiced frames in query audio.")

    contour = _normalize_pitch(voiced)
    contour = _trim_and_downsample(contour, cfg.contour_max_len)
    embedding = build_embedding_from_contour(contour, cfg.embedding_dim)
    return QueryMelody(contour=contour, embedding=embedding)
