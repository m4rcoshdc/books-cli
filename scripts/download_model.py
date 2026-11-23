"""Download the bge-m3 embedding model (PyTorch weights only, ~2.3 GB) into the
Hugging Face cache, then embed one sentence as a smoke test.

    uv run python scripts/download_model.py
"""

import time

from huggingface_hub import snapshot_download

MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"  # pinned: same weights every time

t = time.time()
path = snapshot_download(
    MODEL,
    revision=REVISION,
    allow_patterns=["*.json", "pytorch_model.bin", "sentencepiece.bpe.model", "1_Pooling/*"],
    ignore_patterns=["onnx/*"],
)
print(f"downloaded to {path} in {time.time() - t:.0f}s")

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(path, device="cpu")
v = model.encode(["Stocks change over time through the actions of a flow."], normalize_embeddings=True)
print("embedding dims:", v.shape[1])
