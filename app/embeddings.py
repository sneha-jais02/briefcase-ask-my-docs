import hashlib
import re
from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def _evaluation_embedding(text: str, dimensions: int) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in re.findall(r"\w+", text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if settings.rag_eval_mode:
        return [_evaluation_embedding(text, settings.embedding_dimensions) for text in texts]
    return _model().encode(texts, normalize_embeddings=True).tolist()
