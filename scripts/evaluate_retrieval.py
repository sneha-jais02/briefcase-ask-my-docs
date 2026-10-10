import argparse
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import retrieval_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval, context precision, and answer quality.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--email", default=os.getenv("BOOTSTRAP_ADMIN_EMAIL", "admin@localhost.local"))
    parser.add_argument("--password", default=os.getenv("BOOTSTRAP_ADMIN_PASSWORD", os.getenv("API_KEY", "")))
    parser.add_argument("--dataset", type=Path, default=Path("eval/corpus.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.65)
    parser.add_argument("--min-context-precision", type=float, default=0.65)
    parser.add_argument("--min-faithfulness", type=float, default=0.8)
    parser.add_argument("--min-answer-relevance", type=float, default=0.45)
    args = parser.parse_args()
    if not args.password:
        parser.error("Set BOOTSTRAP_ADMIN_PASSWORD or API_KEY, or pass --password.")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.url, timeout=120) as client:
        login = client.post("/api/auth/login", json={"email": args.email, "password": args.password})
        login.raise_for_status()
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        client.headers.update(headers)
        for document in dataset["documents"]:
            metadata = {key: str(document[key]) for key in (
                "title", "source_url", "jurisdiction", "court", "case_number", "document_date"
            )}
            response = client.post(
                "/api/documents",
                data=metadata,
                files={"file": ("eval.txt", document["text"], "text/plain")},
            )
            response.raise_for_status()
        answers: list[dict] = []
        for case in dataset["questions"]:
            response = client.post("/api/ask", json={
                "question": case["question"],
                "jurisdiction": case["jurisdiction"],
            })
            response.raise_for_status()
            answers.append(response.json())
    metrics = retrieval_metrics(dataset["questions"], answers, args.k)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    failures = []
    if metrics[f"recall@{args.k}"] < args.min_recall:
        failures.append(f"recall@{args.k} {metrics[f'recall@{args.k}']:.3f} < {args.min_recall:.3f}")
    if metrics["mrr"] < args.min_mrr:
        failures.append(f"mrr {metrics['mrr']:.3f} < {args.min_mrr:.3f}")
    for metric, minimum in (
        (f"context_precision@{args.k}", args.min_context_precision),
        ("faithfulness", args.min_faithfulness),
        ("answer_relevance", args.min_answer_relevance),
    ):
        if metrics[metric] < minimum:
            failures.append(f"{metric} {metrics[metric]:.3f} < {minimum:.3f}")
    if failures:
        print("Evaluation gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("Evaluation gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
