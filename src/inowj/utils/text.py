"""Text normalization and abstraction utilities."""

from __future__ import annotations

import re

RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
RE_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_term(term: str) -> str:
    """Normalize dictionary terms for matching.

    Uses casefolding and whitespace trimming.

    Args:
        term: Raw term.

    Returns:
        Normalized term.
    """
    return term.casefold().strip()


def abstract_year_num(text: str) -> str:
    """Normalize a text by abstracting years and numbers.

    - Years matching \\b(19|20)\\d{2}\\b -> <YEAR>
    - Numbers -> <NUM>

    Args:
        text: Input string.

    Returns:
        Lowercased, abstracted string.
    """
    if not text:
        return text

    if not re.search(r"\d", text):
        return re.sub(r"\s+", " ", text).strip()

    text = text.casefold()
    text = RE_YEAR.sub("<YEAR>", text)
    text = RE_NUM.sub("<NUM>", text)
    return re.sub(r"\s+", " ", text).strip()


def is_whole_word(sent: str, start: int, end: int) -> bool:
    """Check whether [start, end) is aligned to word boundaries."""
    if start > 0 and sent[start - 1].isalnum():
        return False
    if end < len(sent) and sent[end].isalnum():
        return False
    return True


def abstract_with_offset_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Abstract years/numbers while preserving a char-level offset map.

    Returns:
        abs_text: Abstracted text (lowercased).
        abs_to_raw: For each character position i in abs_text, a tuple
            (raw_start, raw_end) mapping to the original text span.

    This matches the logic from the provided training notebook.
    """
    text = text.lower()
    abs_parts: list[str] = []
    abs_to_raw: list[tuple[int, int]] = []

    last_pos = 0

    matches: list[tuple[str, int, int]] = []
    for m in RE_YEAR.finditer(text):
        matches.append(("YEAR", m.start(), m.end()))
    for m in RE_NUM.finditer(text):
        if not any(m.start() >= ys and m.end() <= ye for _, ys, ye in matches):
            matches.append(("NUM", m.start(), m.end()))

    matches.sort(key=lambda x: x[1])

    for match_type, start, end in matches:
        for i in range(last_pos, start):
            abs_parts.append(text[i])
            abs_to_raw.append((i, i + 1))

        token = f"<{match_type}>"
        for ch in token:
            abs_parts.append(ch)
            abs_to_raw.append((start, end))

        last_pos = end

    for i in range(last_pos, len(text)):
        abs_parts.append(text[i])
        abs_to_raw.append((i, i + 1))

    # Collapse whitespace (single-space)
    collapsed_abs: list[str] = []
    collapsed_map: list[tuple[int, int]] = []
    in_space = False
    for ch, span in zip(abs_parts, abs_to_raw):
        if ch.isspace():
            if not in_space:
                collapsed_abs.append(" ")
                collapsed_map.append(span)
                in_space = True
        else:
            collapsed_abs.append(ch)
            collapsed_map.append(span)
            in_space = False

    abs_joined = "".join(collapsed_abs)
    abs_str = abs_joined.strip()

    trim_start = len(abs_joined) - len(abs_joined.lstrip())
    trim_end = len(abs_joined) - len(abs_joined.rstrip())
    if trim_end > 0:
        final_map = collapsed_map[trim_start : len(collapsed_map) - trim_end]
    else:
        final_map = collapsed_map[trim_start:]

    return abs_str, final_map


def bert_tokens_to_sentence_with_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Reconstruct a sentence from BERT wordpieces and compute token offsets.

    Rules:
      - Tokens starting with '##' are merged into the previous word without space.
      - Non-subword tokens are separated by a single space.

    Special tokens [CLS]/[SEP]/[PAD] are skipped.

    Args:
        tokens: List of BERT tokens (strings).

    Returns:
        sentence: Reconstructed sentence.
        offsets: List of (start, end) for each non-special token in `tokens`.
    """
    sentence_chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0

    for tok in tokens:
        tok = str(tok) if tok is not None else ""
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if tok.startswith("##"):
            tok_text = tok[2:]
        else:
            if pos > 0:
                sentence_chars.append(" ")
                pos += 1
            tok_text = tok

        start = pos
        sentence_chars.append(tok_text)
        pos += len(tok_text)
        end = pos

        offsets.append((start, end))

    return "".join(sentence_chars), offsets
