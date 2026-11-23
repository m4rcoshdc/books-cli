"""Embeddings with BAAI/bge-m3 (multilingual, 1024 dims), run locally: on the
GPU in half precision when CUDA is available (~26 chunks/s on an RTX 3050 Ti),
otherwise on the CPU (~1.4 chunks/s).

Vectors are cached by a hash of (model, text) in data/cache/embeddings.sqlite,
so rebuilding the index only embeds chunks whose text actually changed.
"""

import hashlib
import sqlite3
import time
from collections.abc import Callable

import numpy as np

from .config import CACHE_DIR

MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMS = 1024
MAX_TOKENS = 1024  # our chunks are ~400-600 words; the model allows up to 8192
CACHE_FILE = CACHE_DIR / "embeddings.sqlite"

_model = None


def device() -> str:
    """GPU (half precision) when available, else CPU (full precision)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def precision() -> str:
    return "fp16" if device() == "cuda" else "fp32"


def load_model():
    """Load bge-m3 from the local Hugging Face cache (scripts/download_model.py)."""
    global _model
    if _model is None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        # Same file selection as scripts/download_model.py (PyTorch weights, no ONNX copy).
        path = snapshot_download(MODEL, revision=REVISION, local_files_only=True,
                                 allow_patterns=["*.json", "pytorch_model.bin", "sentencepiece.bpe.model", "1_Pooling/*"],
                                 ignore_patterns=["onnx/*"])
        _model = SentenceTransformer(path, device=device())
        if device() == "cuda":
            _model.half()  # 4 GB laptop GPU: full-precision weights alone would take 2.3 GB
        _model.max_seq_length = MAX_TOKENS
    return _model


def encode(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Encode; on a GPU out-of-memory error, retry with half the batch size."""
    import torch

    while True:
        try:
            return load_model().encode(
                texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size //= 2


def text_hash(text: str) -> str:
    # Precision is part of the key so fp16 (GPU) and fp32 (CPU) vectors never mix.
    key = f"{MODEL}@{REVISION}|{MAX_TOKENS}|{precision()}|{text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _cache() -> sqlite3.Connection:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS vectors (hash TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    return db


def cached_vectors(hashes: list[str]) -> dict[str, bytes]:
    db = _cache()
    out = {}
    for i in range(0, len(hashes), 500):
        part = hashes[i : i + 500]
        q = f"SELECT hash, vec FROM vectors WHERE hash IN ({','.join('?' * len(part))})"
        out.update(dict(db.execute(q, part).fetchall()))
    db.close()
    return out


def embed_missing(items: list[tuple[str, str]], progress: Callable[[int, int, float], None] | None = None,
                  batch: int = 32) -> int:
    """items: (hash, text). Embeds those not in the cache, saving every batch so an
    interrupted run resumes where it stopped. Returns how many were embedded."""
    have = cached_vectors([h for h, _ in items])
    todo = [(h, t) for h, t in items if h not in have]
    # Similar lengths in one batch waste less padding work.
    todo.sort(key=lambda x: len(x[1]))
    db = _cache()
    start = time.time()
    for i in range(0, len(todo), batch):
        part = todo[i : i + batch]
        vecs = encode([t for _, t in part], batch_size=16 if device() == "cuda" else 8)
        db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                       [(h, v.tobytes()) for (h, _), v in zip(part, vecs)])
        db.commit()
        if progress:
            progress(i + len(part), len(todo), time.time() - start)
    db.close()
    return len(todo)
