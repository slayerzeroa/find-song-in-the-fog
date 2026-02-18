"""Audio feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    """Compact representation of one track."""

    path: str
    duration: float
    tempo: float
    rms_mean: float
    spectral_centroid_mean: float
    chroma_mean: list[float]
    fft_signature: list[float]
    top_notes: list[str]


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr_loaded = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    y = librosa.util.normalize(y)
    return y, sr_loaded


def _build_fft_signature(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    mean_spectrum = np.mean(mag, axis=1)
    signature = np.log1p(mean_spectrum)
    signature /= np.linalg.norm(signature) + 1e-12
    return signature


def extract_audio_features(path: str | Path, top_notes: list[str]) -> AudioFeatures:
    y, sr = load_audio(path)

    duration = librosa.get_duration(y=y, sr=sr)
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo_arr).reshape(-1)[0])

    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    fft_signature = _build_fft_signature(y)

    return AudioFeatures(
        path=str(path),
        duration=float(duration),
        tempo=tempo,
        rms_mean=float(np.mean(rms)),
        spectral_centroid_mean=float(np.mean(centroid)),
        chroma_mean=np.mean(chroma, axis=1).tolist(),
        fft_signature=fft_signature.tolist(),
        top_notes=top_notes,
    )
