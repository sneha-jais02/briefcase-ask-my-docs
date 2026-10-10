from app.chunking import legal_chunks


def test_chunks_preserve_legal_headings() -> None:
    text = (
        "SECTION 1.1 Scope\nThe first legal paragraph.\n\nAnother paragraph.\n\n"
        "SECTION 2.1 Termination\nA distinct termination provision."
    )
    chunks = legal_chunks(text, max_chars=80)

    assert chunks[0][0] == "SECTION 1.1 Scope"
    assert "first legal paragraph" in chunks[0][1]
    assert any(heading == "SECTION 2.1 Termination" for heading, _ in chunks)


def test_chunks_overlap_paragraphs_and_bound_long_text() -> None:
    text = "first paragraph.\n\nsecond paragraph.\n\nthird paragraph."
    chunks = legal_chunks(text, max_chars=35, overlap_paragraphs=1)

    assert len(chunks) == 2
    assert all(len(content) <= 35 for _, content in chunks)
    assert "second paragraph" in chunks[0][1]
    assert "second paragraph" in chunks[1][1]


def test_empty_input_returns_no_chunks() -> None:
    assert legal_chunks(" \n\n ") == []


def test_oversized_paragraph_prefers_sentence_boundaries_and_preserves_words() -> None:
    text = "A legal sentence ends here. A second sentence ends here. A third sentence ends here."

    chunks = legal_chunks(text, max_chars=35, overlap_paragraphs=0)
    pieces = [content for _, content in chunks]

    assert all(len(content) <= 35 for _, content in chunks)
    assert " ".join(pieces) == text
    assert pieces == [
        "A legal sentence ends here.",
        "A second sentence ends here.",
        "A third sentence ends here.",
    ]
