"""FAISS index build/load helpers for melody embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


@dataclass
class ANNConfig:
    index_type: str = "ivfpq"  # ivfpq | hnsw | flat
    nlist: int = 256
    m: int = 16
    nbits: int = 8
    nprobe: int = 16
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64


def _ensure_faiss() -> None:
    if faiss is None:
        raise RuntimeError(
            "faiss is not installed. Install with `pip install faiss-cpu`."
        ) from _FAISS_IMPORT_ERROR


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def _as_faiss_array(matrix: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    if out.ndim != 2:
        raise ValueError("matrix must be 2D")
    return out


def _build_ivfpq(embeddings: np.ndarray, cfg: ANNConfig):
    _ensure_faiss()
    if embeddings.shape[0] < 2:
        raise ValueError("IVF-PQ requires at least 2 vectors.")
    dim = embeddings.shape[1]
    if cfg.m <= 0 or cfg.m > dim:
        raise ValueError(f"Invalid m ({cfg.m}) for embedding dim ({dim}).")
    if dim % cfg.m != 0:
        raise ValueError(f"For IVFPQ, embedding dim ({dim}) must be divisible by m ({cfg.m}).")

    nlist = max(1, min(cfg.nlist, embeddings.shape[0]))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, cfg.m, cfg.nbits, faiss.METRIC_INNER_PRODUCT)

    # Ensure we have enough training points even for smaller corpora.
    min_train_rows = max(nlist * 40, 1 << cfg.nbits)
    train_rows = max(min_train_rows, min(embeddings.shape[0], 4096))
    if train_rows <= embeddings.shape[0]:
        train_set = embeddings[np.random.choice(embeddings.shape[0], train_rows, replace=False)]
    else:
        extra = train_rows - embeddings.shape[0]
        sampled = embeddings[np.random.choice(embeddings.shape[0], extra, replace=True)]
        train_set = np.vstack([embeddings, sampled])

    index.train(_as_faiss_array(train_set))
    index.add(_as_faiss_array(embeddings))
    index.nprobe = min(cfg.nprobe, nlist)
    return index


def _build_hnsw(embeddings: np.ndarray, cfg: ANNConfig):
    _ensure_faiss()
    dim = embeddings.shape[1]
    index = faiss.index_factory(dim, f"HNSW{cfg.hnsw_m},Flat", faiss.METRIC_INNER_PRODUCT)
    if not hasattr(index, "hnsw"):
        raise RuntimeError("Failed to build HNSW index.")
    index.hnsw.efConstruction = cfg.ef_construction
    index.hnsw.efSearch = cfg.ef_search
    index.add(_as_faiss_array(embeddings))
    return index


def _build_flat(embeddings: np.ndarray):
    _ensure_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(_as_faiss_array(embeddings))
    return index


def build_index(embeddings: np.ndarray, cfg: ANNConfig):
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must have shape [n, d] with n > 0")

    normalized = _normalize_rows(embeddings)
    index_type = cfg.index_type.lower()
    if index_type == "ivfpq":
        return _build_ivfpq(normalized, cfg), normalized
    if index_type == "hnsw":
        return _build_hnsw(normalized, cfg), normalized
    if index_type == "flat":
        return _build_flat(normalized), normalized
    raise ValueError(f"Unsupported index_type: {cfg.index_type}")


def save_bundle(index_dir: str | Path, index, metadata: list[dict], cfg: ANNConfig) -> None:
    _ensure_faiss()
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(path / "melody.faiss"))
    (path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path / "ann_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_bundle(index_dir: str | Path):
    _ensure_faiss()
    path = Path(index_dir)
    index_path = path / "melody.faiss"
    metadata_path = path / "metadata.json"
    cfg_path = path / "ann_config.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing ANN index file: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing ANN config file: {cfg_path}")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = ANNConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))

    if isinstance(index, faiss.IndexIVF):
        index.nprobe = min(cfg.nprobe, max(1, index.nlist))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = cfg.ef_search
    return index, metadata, cfg


def search(index, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1D")
    q = np.asarray(query_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    scores, ids = index.search(_as_faiss_array(q.reshape(1, -1)), int(top_k))
    return scores[0], ids[0]
