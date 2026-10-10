import json
import sys
from types import SimpleNamespace

import pytest

from app import answering
from app.answering import grounded_answer, validate_citations


def make_source() -> dict:
    return {
        "id": "C1",
        "document_id": "doc-1",
        "title": "Fictional Case",
        "source_url": "eval://case",
        "jurisdiction": "Fictional",
        "court": "Demonstration Court",
        "case_number": "DEMO-1",
        "document_date": "2025-01-01",
        "version": 1,
        "section_title": "SECTION 1.1 Notice",
        "content": "Either party must provide thirty days of prior written notice.",
        "excerpt": "Either party must provide thirty days of prior written notice.",
    }


def make_additional_source() -> dict:
    return {
        "id": "C2",
        "document_id": "doc-3",
        "title": "Another Fictional Case",
        "source_url": "eval://another-case",
        "jurisdiction": "Fictional",
        "court": "Demonstration Court",
        "case_number": "DEMO-3",
        "document_date": "2025-02-01",
        "version": 1,
        "section_title": "SECTION 2.1 Deadline",
        "content": "A claim must be filed within six years after it accrues.",
        "excerpt": "A claim must be filed within six years after it accrues.",
    }


def test_citations_require_known_ids_and_verbatim_source_quotes() -> None:
    source = make_source()

    valid = validate_citations(
        "Thirty days' notice is required [C1].",
        [{"id": "C1", "quote": "thirty days of prior written notice."}],
        [source],
    )
    assert valid is not None
    assert valid["citations"][0]["title"] == source["title"]

    assert validate_citations(
        "A different rule applies [C1].",
        [{"id": "C1", "quote": "A different rule applies."}],
        [source],
    ) is None
    assert validate_citations(
        "The rule is stated here. Thirty days' notice is required [C1].",
        [{"id": "C1", "quote": "thirty days of prior written notice."}],
        [source],
    ) is None
    assert validate_citations(
        "The answer is 30 days [C9].",
        [{"id": "C9", "quote": "thirty days"}],
        [source],
    ) is None


def test_missing_citations_falls_back_to_retrieved_excerpts(monkeypatch) -> None:
    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(
        llm_provider="openai",
        openai_api_key="",
        gemini_api_key="",
        llm_model="gpt-4o-mini",
    ))
    result = grounded_answer("How much notice?", [make_source()], [])

    assert "[C1]" in result["answer"]
    assert result["citations"][0]["document_id"] == "doc-1"
    assert result["generation_mode"] == "extractive"
    assert "thirty days" in result["answer"]
    assert "OPENAI_API_KEY" in result["generation_notice"]


def test_openai_response_is_synthesized_and_citations_are_verified(monkeypatch) -> None:
    source = make_source()
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer": "Either party must provide 30 days of written notice [C1].",
                        "citations": [{"id": "C1", "quote": "thirty days of prior written notice"}],
                    })
                )
            )
        ]
    )

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-llm-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: completion)
            )

    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(
        llm_provider="openai",
        openai_api_key="test-llm-key",
        gemini_api_key="",
        llm_model="test-model",
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=RuntimeError, OpenAI=FakeOpenAI))

    result = grounded_answer("How much notice?", [source], [])

    assert result["generation_mode"] == "llm"
    assert result["answer"] == "Either party must provide 30 days of written notice [C1]."
    assert result["citations"][0]["title"] == source["title"]


def test_gemini_key_uses_google_openai_compatible_endpoint(monkeypatch) -> None:
    source = make_source()
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer": "The agreement requires thirty days of written notice [C1].",
                        "citations": [{"id": "C1", "quote": "thirty days of prior written notice"}],
                    })
                )
            )
        ]
    )
    client_options: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: str) -> None:
            client_options.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: completion)
            )

    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(
        llm_provider="gemini",
        openai_api_key="",
        gemini_api_key="test-google-ai-studio-key",
        llm_model="gemini-test-model",
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=RuntimeError, OpenAI=FakeOpenAI))

    result = grounded_answer("How much notice?", [source], [])

    assert result["generation_mode"] == "llm"
    assert client_options == {
        "api_key": "test-google-ai-studio-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }


def test_prompt_lists_exact_retrieved_ids_and_normalizes_equivalent_tags(monkeypatch) -> None:
    sources = [make_source(), make_additional_source()]
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer": (
                            "Thirty days of prior written notice is required 【c1】. "
                            "A claim must be filed within six years [Source C2; c1]."
                        ),
                        "citations": [
                            {"id": "C1", "quote": "thirty days of prior written notice"},
                            {"id": "C2", "quote": "filed within six years after it accrues"},
                        ],
                    })
                )
            )
        ]
    )
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **_: str) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (captured.update(kwargs) or completion)
                )
            )

    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(
        llm_provider="gemini",
        openai_api_key="",
        gemini_api_key="test-google-ai-studio-key",
        llm_model="gemini-test-model",
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=RuntimeError, OpenAI=FakeOpenAI))

    result = grounded_answer("How much notice and what deadline?", [
        {**source, "hybrid_score": 1.0}
        for source in sources
    ], [])

    prompt = captured["messages"][0]["content"]
    assert "[C1], [C2]" in prompt
    assert "Do not renumber IDs" in prompt
    assert result["generation_mode"] == "llm"
    assert result["answer"] == (
        "Thirty days of prior written notice is required [C1]. "
        "A claim must be filed within six years [C2] [C1]."
    )
    assert {citation["id"] for citation in result["citations"]} == {"C1", "C2"}


def test_invalid_llm_citation_is_explicitly_downgraded(monkeypatch) -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer": "The answer is ten days [C1].",
                        "citations": [{"id": "C1", "quote": "The source says ten days."}],
                    })
                )
            )
        ]
    )

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: completion)
            )

    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(
        llm_provider="openai",
        openai_api_key="test-llm-key",
        gemini_api_key="",
        llm_model="test-model",
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=RuntimeError, OpenAI=FakeOpenAI))

    result = grounded_answer("How much notice?", [make_source()], [])

    assert result["generation_mode"] == "extractive_fallback"
    assert "could not be verified" in result["generation_notice"]
    assert "[C1]" in result["answer"]


def test_openai_provider_errors_are_not_hidden_by_extracts(monkeypatch) -> None:
    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(RuntimeError("provider unavailable"))
                )
            )

    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(
        llm_provider="openai",
        openai_api_key="test-llm-key",
        gemini_api_key="",
        llm_model="test-model",
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=RuntimeError, OpenAI=FakeOpenAI))

    with pytest.raises(answering.SynthesisError, match="Check the configured provider key"):
        grounded_answer("How much notice?", [make_source()], [])
