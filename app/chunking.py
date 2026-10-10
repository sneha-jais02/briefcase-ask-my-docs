import re

HEADING = re.compile(
    r"^\s*(?:(?:SECTION|ARTICLE)\s+[\w.-]+.*|\d+(?:\.\d+)*\s+[A-Z].*|[IVXLC]+\.\s+[A-Z].*)$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    remaining = paragraph.strip()
    pieces: list[str] = []
    while len(remaining) > max_chars:
        lower_bound = max_chars // 2
        sentence_boundaries = [
            match.end()
            for match in re.finditer(r"[.!?;](?:\s+|$)", remaining[:max_chars + 1])
            if match.end() >= lower_bound
        ]
        whitespace_boundaries = [
            match.start()
            for match in re.finditer(r"\s+", remaining[:max_chars + 1])
            if match.start() >= lower_bound
        ]
        boundary = sentence_boundaries[-1] if sentence_boundaries else (
            whitespace_boundaries[-1] if whitespace_boundaries else max_chars
        )
        piece = remaining[:boundary].strip()
        if not piece:
            boundary = max_chars
            piece = remaining[:boundary]
        pieces.append(piece)
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def legal_chunks(text: str, max_chars: int = 1800, overlap_paragraphs: int = 1) -> list[tuple[str, str]]:
    """Split legal text on section headings and paragraph boundaries."""
    if max_chars < 1:
        raise ValueError("max_chars must be a positive integer.")
    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs cannot be negative.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in normalized.splitlines():
        if HEADING.match(line.strip()) and current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.strip()
            current_lines = []
        elif HEADING.match(line.strip()):
            current_heading = line.strip()
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    chunks: list[tuple[str, str]] = []
    for heading, section_text in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        batch: list[str] = []
        for paragraph in paragraphs:
            batch_chars = sum(map(len, batch)) + 2 * max(len(batch) - 1, 0)
            if batch and batch_chars + len(paragraph) + 2 > max_chars:
                chunks.append((heading, "\n\n".join(batch)))
                batch = batch[-overlap_paragraphs:] if overlap_paragraphs else []
                batch_chars = sum(map(len, batch)) + 2 * max(len(batch) - 1, 0)
                if batch and batch_chars + len(paragraph) + 2 > max_chars:
                    batch = []
            if len(paragraph) > max_chars:
                if batch:
                    chunks.append((heading, "\n\n".join(batch)))
                    batch = []
                chunks.extend((heading, piece) for piece in _split_long_paragraph(paragraph, max_chars))
                continue
            batch.append(paragraph)
        if batch:
            chunks.append((heading, "\n\n".join(batch)))
    return chunks
