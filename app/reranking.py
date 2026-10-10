import re
from functools import lru_cache
from typing import Any

from app.config import get_settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not hits:
        return []
    settings = get_settings()
    if settings.rag_eval_mode:
        query_terms = set(re.findall(r"\w+", query.casefold()))

        def score(hit: dict[str, Any]) -> tuple[float, float]:
            text_terms = set(re.findall(r"\w+", hit["content"].casefold()))
            return (len(query_terms & text_terms) / max(len(query_terms), 1), hit["hybrid_score"])

        ordered = sorted(hits, key=score, reverse=True)
        for hit in ordered:
            hit["rerank_score"] = score(hit)[0]
        return ordered
    scores = _model().predict([(query, hit["content"]) for hit in hits])
    ordered = sorted(zip(hits, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    result = []
    for hit, score in ordered:
        hit["rerank_score"] = float(score)
        result.append(hit)
    return result
