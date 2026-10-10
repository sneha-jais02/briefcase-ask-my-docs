import re
from collections.abc import Sequence
from typing import Any

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "had", "has", "have", "how", "in", "into", "is", "it", "its", "may", "must", "of",
    "on", "or", "other", "that", "the", "their", "there", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}
MARKER = re.compile(r"\[(C\d+)\]")


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"\w+", text.casefold()) if term not in STOP_WORDS}


def retrieval_metrics(
    cases: Sequence[dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    k: int = 3,
) -> dict[str, float]:
    if k < 1 or not cases or len(cases) != len(answers):
        raise ValueError("Require k >= 1 and equal, non-empty case and answer collections.")
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    context_precisions: list[float] = []
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []
    for case, answer in zip(cases, answers, strict=True):
        relevant = set(case["expected_titles"])
        citations = answer.get("citations", [])
        source_by_id = {citation["id"]: citation for citation in citations if citation.get("id")}
        ranking = [
            citation["title"]
            for citation in citations
            if isinstance(citation.get("title"), str)
        ]
        top_k = set(ranking[:k])
        recall_hits += bool(relevant & top_k)
        reciprocal_ranks.append(next((1 / rank for rank, title in enumerate(ranking, 1) if title in relevant), 0.0))
        relevant_ranks = [
            rank
            for rank, title in enumerate(ranking[:k], 1)
            if title in relevant
        ]
        context_precisions.append(
            sum(index / rank for index, rank in enumerate(relevant_ranks, 1)) / len(relevant)
            if relevant else 0.0
        )

        factual_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.get("answer", ""))
            if sentence.strip()
        ]
        sentence_scores = []
        for sentence in factual_sentences:
            source_ids = MARKER.findall(sentence)
            sentence_terms = _terms(MARKER.sub("", sentence))
            supported_terms: set[str] = set()
            for source_id in source_ids:
                citation = source_by_id.get(source_id)
                if citation:
                    supported_terms.update(_terms(citation.get("excerpt", "")))
            sentence_scores.append(
                len(sentence_terms & supported_terms) / len(sentence_terms)
                if sentence_terms and source_ids else 0.0
            )
        faithfulness_scores.append(
            sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
        )

        reference_terms = _terms(case.get("reference_answer", ""))
        answer_terms = _terms(MARKER.sub("", answer.get("answer", "")))
        overlap = len(reference_terms & answer_terms)
        relevance_scores.append(
            2 * overlap / (len(reference_terms) + len(answer_terms))
            if reference_terms and answer_terms else 0.0
        )
    count = len(cases)
    return {
        f"recall@{k}": recall_hits / count,
        "mrr": sum(reciprocal_ranks) / count,
        f"context_precision@{k}": sum(context_precisions) / count,
        "faithfulness": sum(faithfulness_scores) / count,
        "answer_relevance": sum(relevance_scores) / count,
    }
