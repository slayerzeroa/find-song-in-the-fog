"""Pitch decomposition and pitch-space conversion."""

from __future__ import annotations

import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(freq: np.ndarray) -> np.ndarray:
    midi = librosa.hz_to_midi(freq)
    midi = np.where(np.isfinite(midi), midi, np.nan)
    return midi


def midi_to_note_name(midi_number: float) -> str:
    note_index = int(np.round(midi_number)) % 12
    octave = int(np.floor(np.round(midi_number) / 12.0) - 1)
    return f"{NOTE_NAMES[note_index]}{octave}"


def extract_pitch_track(
    y: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1200.0,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> np.ndarray:
    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    # YIN can emit boundary values for unvoiced frames; keep only finite positive values.
    return f0[(np.isfinite(f0)) & (f0 > 0)]


def top_pitch_notes(f0_track: np.ndarray, top_k: int = 8) -> list[str]:
    if f0_track.size == 0:
        return []

    midi = hz_to_midi(f0_track)
    midi = midi[np.isfinite(midi)]
    if midi.size == 0:
        return []

    bins = np.round(midi).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1]

    notes: list[str] = []
    for idx in order[:top_k]:
        notes.append(midi_to_note_name(float(values[idx])))
    return notes
