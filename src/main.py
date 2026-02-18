"""CLI entry point for legacy and ANN+DTW based humming search."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ann_index import ANNConfig, build_index, load_bundle, save_bundle, search
from audio_analyzer import extract_audio_features, load_audio
from dtw_reranker import dtw_similarity, key_invariant_dtw
from fft_matcher import combined_similarity
from melody_embedding import SegmentConfig, extract_query_melody, extract_song_segments
from pitch_extractor import extract_pitch_track, top_pitch_notes

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _audio_files(directory: Path) -> list[Path]:
    files = [p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def index_songs(song_dir: Path, out_file: Path) -> None:
    songs = _audio_files(song_dir)
    if not songs:
        raise ValueError(f"No audio files found in {song_dir}")

    records = []
    for path in songs:
        y, sr = load_audio(path)
        f0 = extract_pitch_track(y, sr)
        notes = top_pitch_notes(f0)
        features = extract_audio_features(path, notes)
        records.append(features.__dict__)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(records)} songs -> {out_file}")


def search_hum(query_file: Path, index_file: Path, top_k: int) -> None:
    index = json.loads(index_file.read_text(encoding="utf-8"))
    if not index:
        raise ValueError("Index is empty.")

    y, sr = load_audio(query_file)
    query_f0 = extract_pitch_track(y, sr)
    query_notes = top_pitch_notes(query_f0)
    query_features = extract_audio_features(query_file, query_notes)
    query_fft = np.array(query_features.fft_signature, dtype=np.float64)

    scored: list[tuple[dict, dict[str, float]]] = []
    for item in index:
        cand_fft = np.array(item["fft_signature"], dtype=np.float64)
        scores = combined_similarity(
            query_fft=query_fft,
            candidate_fft=cand_fft,
            query_notes=query_features.top_notes,
            candidate_notes=item.get("top_notes", []),
        )
        scored.append((item, scores))

    scored.sort(key=lambda x: x[1]["total"], reverse=True)

    print(f"Query: {query_file}")
    print(f"Detected notes: {', '.join(query_features.top_notes) if query_features.top_notes else '(none)'}")
    print("\nTop matches")
    for rank, (item, score) in enumerate(scored[:top_k], start=1):
        print(
            f"{rank:>2}. {item['path']} | total={score['total']:.4f} "
            f"(fft={score['fft']:.4f}, pitch={score['pitch']:.4f})"
        )


def _to_segment_config(args: argparse.Namespace) -> SegmentConfig:
    return SegmentConfig(
        segment_seconds=args.segment_seconds,
        segment_hop_seconds=args.segment_hop_seconds,
        min_voiced_ratio=args.min_voiced_ratio,
        max_segments_per_song=args.max_segments_per_song,
        embedding_dim=args.embedding_dim,
        contour_max_len=args.contour_max_len,
        energy_percentile=args.energy_percentile,
    )


def _to_ann_config(args: argparse.Namespace) -> ANNConfig:
    return ANNConfig(
        index_type=args.ann_type,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )


def index_songs_ann(song_dir: Path, out_dir: Path, seg_cfg: SegmentConfig, ann_cfg: ANNConfig) -> None:
    songs = _audio_files(song_dir)
    if not songs:
        raise ValueError(f"No audio files found in {song_dir}")

    metadata: list[dict] = []
    embeddings: list[np.ndarray] = []
    skipped = 0

    for song_path in songs:
        segments = extract_song_segments(song_path, seg_cfg)
        if not segments:
            skipped += 1
            continue

        for seg in segments:
            metadata.append(
                {
                    "path": str(song_path),
                    "start_sec": float(seg.start_sec),
                    "end_sec": float(seg.end_sec),
                    "contour": np.round(seg.contour, 4).astype(float).tolist(),
                }
            )
            embeddings.append(seg.embedding.astype(np.float32, copy=False))

    if not embeddings:
        raise ValueError("Failed to extract any melody segment embeddings.")

    matrix = np.vstack(embeddings).astype(np.float32)
    index, _ = build_index(matrix, ann_cfg)
    save_bundle(out_dir, index, metadata, ann_cfg)

    (out_dir / "segment_config.json").write_text(
        json.dumps(asdict(seg_cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Indexed {len(metadata)} segments from {len(songs)} songs -> {out_dir}")
    if skipped:
        print(f"Skipped {skipped} songs without enough voiced melody.")


def _load_segment_config(index_dir: Path) -> SegmentConfig:
    path = index_dir / "segment_config.json"
    if not path.exists():
        return SegmentConfig()
    return SegmentConfig(**json.loads(path.read_text(encoding="utf-8")))


def search_hum_ann(
    query_file: Path,
    index_dir: Path,
    top_k: int,
    candidate_k: int,
    ann_weight: float,
    transpose_range: int,
    dtw_window_ratio: float,
    min_score: float,
) -> None:
    index, metadata, _ = load_bundle(index_dir)
    if not metadata:
        raise ValueError("ANN metadata is empty.")
    ann_weight = float(np.clip(ann_weight, 0.0, 1.0))
    min_score = float(np.clip(min_score, 0.0, 1.0))

    seg_cfg = _load_segment_config(index_dir)
    query = extract_query_melody(query_file, seg_cfg)

    cand_k = max(top_k, min(candidate_k, len(metadata)))
    ann_scores, ann_ids = search(index=index, query_embedding=query.embedding, top_k=cand_k)

    per_song: dict[str, dict] = {}
    for raw_score, seg_id in zip(ann_scores.tolist(), ann_ids.tolist()):
        if seg_id < 0 or seg_id >= len(metadata):
            continue
        item = metadata[seg_id]
        contour = np.asarray(item.get("contour", []), dtype=np.float32)
        if contour.size < 4:
            continue

        dtw_dist, shift = key_invariant_dtw(
            query_contour=query.contour,
            candidate_contour=contour,
            transpose_range=transpose_range,
            window_ratio=dtw_window_ratio,
        )
        dtw_score = dtw_similarity(dtw_dist)
        ann_score = max(0.0, min(1.0, (float(raw_score) + 1.0) * 0.5))
        total = (ann_weight * ann_score) + ((1.0 - ann_weight) * dtw_score)

        song_path = item["path"]
        entry = {
            "path": song_path,
            "total": float(total),
            "ann": ann_score,
            "dtw": dtw_score,
            "dtw_dist": float(dtw_dist),
            "transpose": int(shift),
            "start_sec": float(item.get("start_sec", 0.0)),
            "end_sec": float(item.get("end_sec", 0.0)),
        }
        prev = per_song.get(song_path)
        if prev is None or entry["total"] > prev["total"]:
            per_song[song_path] = entry

    ranked = sorted(per_song.values(), key=lambda row: row["total"], reverse=True)
    confident = [row for row in ranked if row["total"] >= min_score][:top_k]
    if not confident:
        print(f"No confident match candidates found. (min_score={min_score:.2f})")
        return

    print(f"Query: {query_file}")
    print(f"Candidates searched: {cand_k}")
    print("\nTop matches (ANN + DTW)")
    for rank, row in enumerate(confident, start=1):
        print(
            f"{rank:>2}. {row['path']} | total={row['total']:.4f} "
            f"(ann={row['ann']:.4f}, dtw={row['dtw']:.4f}, shift={row['transpose']:+d}, "
            f"seg={row['start_sec']:.1f}-{row['end_sec']:.1f}s)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find song from humming (legacy FFT or ANN+DTW)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index song files")
    p_index.add_argument("--songs", type=Path, required=True, help="Folder with candidate songs")
    p_index.add_argument("--out", type=Path, default=Path("data/index.json"), help="Index output JSON")

    p_query = sub.add_parser("query", help="Search song from hummed audio")
    p_query.add_argument("--hum", type=Path, required=True, help="Hummed audio file")
    p_query.add_argument("--index", type=Path, default=Path("data/index.json"), help="Index JSON")
    p_query.add_argument("--top-k", type=int, default=5, help="How many matches to show")

    p_ann_index = sub.add_parser("index-ann", help="Build ANN melody index (FAISS)")
    p_ann_index.add_argument("--songs", type=Path, required=True, help="Folder with candidate songs")
    p_ann_index.add_argument("--out-dir", type=Path, default=Path("data/ann_index"), help="Output dir for ANN files")
    p_ann_index.add_argument("--segment-seconds", type=float, default=12.0, help="Seconds per segment")
    p_ann_index.add_argument("--segment-hop-seconds", type=float, default=6.0, help="Hop seconds between segments")
    p_ann_index.add_argument("--max-segments-per-song", type=int, default=4, help="Max segments kept per song")
    p_ann_index.add_argument("--min-voiced-ratio", type=float, default=0.35, help="Minimum voiced ratio per segment")
    p_ann_index.add_argument("--embedding-dim", type=int, default=128, help="Base contour embedding dimension")
    p_ann_index.add_argument("--contour-max-len", type=int, default=512, help="Stored contour max length")
    p_ann_index.add_argument("--energy-percentile", type=float, default=20.0, help="RMS gate percentile for voiced frames")
    p_ann_index.add_argument("--ann-type", choices=["ivfpq", "hnsw", "flat"], default="ivfpq", help="FAISS index type")
    p_ann_index.add_argument("--nlist", type=int, default=256, help="IVF cluster count")
    p_ann_index.add_argument("--m", type=int, default=16, help="IVFPQ sub-vector count")
    p_ann_index.add_argument("--nbits", type=int, default=8, help="IVFPQ bits per code")
    p_ann_index.add_argument("--nprobe", type=int, default=16, help="IVFPQ probe count")
    p_ann_index.add_argument("--hnsw-m", type=int, default=32, help="HNSW graph degree")
    p_ann_index.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction")
    p_ann_index.add_argument("--ef-search", type=int, default=64, help="HNSW efSearch")

    p_ann_query = sub.add_parser("query-ann", help="Search humming with ANN + DTW reranking")
    p_ann_query.add_argument("--hum", type=Path, required=True, help="Hummed audio file")
    p_ann_query.add_argument("--index-dir", type=Path, default=Path("data/ann_index"), help="ANN index directory")
    p_ann_query.add_argument("--top-k", type=int, default=5, help="How many songs to show")
    p_ann_query.add_argument("--candidate-k", type=int, default=80, help="ANN candidate pool before rerank")
    p_ann_query.add_argument("--ann-weight", type=float, default=0.55, help="Weight for ANN score in final score")
    p_ann_query.add_argument("--transpose-range", type=int, default=6, help="Semitone shift range for DTW")
    p_ann_query.add_argument("--dtw-window-ratio", type=float, default=0.2, help="Sakoe-Chiba band ratio")
    p_ann_query.add_argument("--min-score", type=float, default=0.45, help="Minimum confidence score")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        index_songs(args.songs, args.out)
    elif args.command == "query":
        search_hum(args.hum, args.index, args.top_k)
    elif args.command == "index-ann":
        seg_cfg = _to_segment_config(args)
        ann_cfg = _to_ann_config(args)
        index_songs_ann(args.songs, args.out_dir, seg_cfg, ann_cfg)
    elif args.command == "query-ann":
        search_hum_ann(
            query_file=args.hum,
            index_dir=args.index_dir,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            ann_weight=args.ann_weight,
            transpose_range=args.transpose_range,
            dtw_window_ratio=args.dtw_window_ratio,
            min_score=args.min_score,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
