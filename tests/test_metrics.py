import pytest

from eval.metrics import retrieval_metrics


def test_retrieval_metrics_score_precision_faithfulness_and_answer_relevance() -> None:
    cases = [
        {"expected_titles": ["A"], "reference_answer": "Required notice is thirty days."},
        {"expected_titles": ["B"], "reference_answer": "The contract deadline is six years."},
    ]
    answers = [
        {
            "answer": "Required notice is thirty days [C1].",
            "citations": [{"id": "C1", "title": "A", "excerpt": "Required notice is thirty days."}],
        },
        {
            "answer": "The contract deadline is six years [C2].",
            "citations": [
                {"id": "C1", "title": "X", "excerpt": "An unrelated clause applies."},
                {"id": "C2", "title": "B", "excerpt": "The contract deadline is six years."},
            ],
        },
    ]

    assert retrieval_metrics(cases, answers, k=2) == {
        "recall@2": pytest.approx(1),
        "mrr": pytest.approx(0.75),
        "context_precision@2": pytest.approx(0.75),
        "faithfulness": pytest.approx(1),
        "answer_relevance": pytest.approx(1),
    }


def test_metrics_penalize_uncited_and_unsupported_answer_claims() -> None:
    cases = [{"expected_titles": ["A"], "reference_answer": "The required notice is thirty days."}]
    answer = {
        "answer": "The required notice is thirty days [C9].",
        "citations": [{"id": "C1", "title": "A", "excerpt": "The source contains no notice period."}],
    }

    metrics = retrieval_metrics(cases, [answer])

    assert metrics["faithfulness"] == 0
    assert metrics["answer_relevance"] > 0
    assert metrics["context_precision@3"] == 1
    assert metrics["recall@3"] == 1


def test_faithfulness_keeps_citation_after_sentence_punctuation() -> None:
    case = {
        "expected_titles": ["A"],
        "reference_answer": "The notice period is thirty days.",
    }
    answer = {
        "answer": "The notice period is thirty days. [C1]",
        "citations": [
            {
                "id": "C1",
                "title": "A",
                "excerpt": "The notice period is thirty days.",
            }
        ],
    }

    metrics = retrieval_metrics([case], [answer])

    assert metrics["faithfulness"] == 1
