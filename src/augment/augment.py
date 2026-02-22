"""Context-aware lexical augmentation for clinical NER in Spanish.

This module implements the core augmentation strategy described in the paper:
entity-preserving context substitution using FastText subword embeddings.

"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy
import spacy.tokens
import fasttext  # type: ignore[import-untyped]


@dataclass
class AugConfig:
    """Hyperparameters and paths for one augmentation run.

    Args:
        data_dir: Directory containing ``.txt``, ``_triggers.json``, and
            ``_args.json`` files.
        output_dir: Destination directory for augmented files.
        fasttext_model_path: Path to a pre-trained FastText ``.bin`` file
            (e.g. ``cc.es.300.bin``).
        spacy_model: spaCy model identifier used for tokenisation and POS
            tagging (default: ``es_core_news_sm``).
        top_k: Maximum number of replacement candidates retrieved per token.
        sim_thresh: Minimum cosine similarity for a candidate to be accepted.
        random_seed: Seed for reproducible token replacement sampling.
    """

    data_dir: str
    output_dir: str
    fasttext_model_path: str
    spacy_model: str = "es_core_news_sm"
    top_k: int = 3
    sim_thresh: float = 0.7
    random_seed: int = 42


@dataclass
class TriggerAnnotation:
    """A single trigger mention aligned to an augmented sentence.

    Args:
        doc_id: Source document identifier.
        trigger_id: Corpus-assigned trigger identifier.
        trigger_type: Semantic class of the trigger.
        trigger_text: Surface string of the trigger in the augmented sentence.
        trigger_start_span: Character offset of the first character (inclusive).
        trigger_end_span: Character offset past the last character (exclusive).
    """

    doc_id: str
    trigger_id: str
    trigger_type: str
    trigger_text: str
    trigger_start_span: int
    trigger_end_span: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict compatible with the ToxHabit JSON schema."""
        return {
            "doc_id": self.doc_id,
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type,
            "trigger_text": self.trigger_text,
            "trigger_start_span": self.trigger_start_span,
            "trigger_end_span": self.trigger_end_span,
        }


@dataclass
class ArgumentAnnotation:
    """A single argument mention aligned to an augmented sentence.

    Args:
        doc_id: Source document identifier.
        arg_id: Corpus-assigned argument identifier.
        arg_type: Semantic role of the argument.
        arg_text: Surface string of the argument in the augmented sentence.
        arg_start_span: Character offset of the first character (inclusive).
        arg_end_span: Character offset past the last character (exclusive).
    """

    doc_id: str
    arg_id: str
    arg_type: str
    arg_text: str
    arg_start_span: int
    arg_end_span: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict compatible with the ToxHabit JSON schema."""
        return {
            "doc_id": self.doc_id,
            "arg_id": self.arg_id,
            "arg_type": self.arg_type,
            "arg_text": self.arg_text,
            "arg_start_span": self.arg_start_span,
            "arg_end_span": self.arg_end_span,
        }


_ALPHA_RE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñüÜ]+$")


def _parse_annotation_root(data: Any, path: Path) -> list[dict]:
    """Normalise JSON shapes used in the ToxHabit corpus.

    Args:
        data: Parsed JSON object.
        path: Source path (used only for error messages).

    Returns:
        A list of annotation dicts.

    Raises:
        ValueError: If *data* does not match either accepted shape.
    """
    if isinstance(data, dict) and "root" in data:
        return data["root"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised annotation JSON format in {path}")


def load_annotations(path: str | Path) -> list[dict]:
    """Load trigger or argument annotations.

    Args:
        path: Path to a JSON annotation file.

    Returns:
        List of annotation dicts.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return _parse_annotation_root(json.load(f), p)


def save_annotations(records: list[dict], path: str | Path) -> None:
    """Write annotation records to a JSON file.

    Args:
        records: List of annotation dicts to serialise.
        path: Destination file path (parent directories are created as needed).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({"root": records}, f, ensure_ascii=False, indent=2)


def iter_doc_ids(data_dir: str | Path) -> list[str]:
    """Return a sorted list of document IDs found in *data_dir*.

    A document ID is the stem of each ``.txt`` file.

    Args:
        data_dir: Directory to scan.

    Returns:
        Sorted list of document ID strings.
    """
    return sorted(p.stem for p in Path(data_dir).iterdir() if p.suffix == ".txt")


def char_spans_to_token_spans(
    doc: spacy.tokens.Doc,
    char_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Project character-level spans onto spaCy token indices.

    Args:
        doc: Tokenised spaCy document.
        char_spans: Character-level ``(start, end)`` spans.

    Returns:
        Corresponding token-index spans with exclusive end indices.
    """
    token_spans: list[tuple[int, int]] = []
    for char_start, char_end in char_spans:
        start_tok: int | None = None
        end_tok: int | None = None
        for i, tok in enumerate(doc):
            if tok.idx <= char_start < tok.idx + len(tok.text):
                start_tok = i
            if tok.idx < char_end <= tok.idx + len(tok.text):
                end_tok = i + 1  # exclusive upper bound
        if start_tok is not None:
            token_spans.append((start_tok, end_tok or start_tok + 1))
    return token_spans


def sentence_char_spans(
    triggers: list[dict],
    args: list[dict],
    sent_start: int,
    sent_end: int,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """Filter annotations to those overlapping a sentence window.

    Args:
        triggers: All trigger annotations for the document.
        args: All argument annotations for the document.
        sent_start: Character offset of the sentence start (inclusive).
        sent_end: Character offset of the sentence end (exclusive).

    Returns:
        A three-tuple ``(t_in, a_in, rel_spans)`` where ``t_in`` and ``a_in``
        are the overlapping trigger and argument dicts respectively, and
        ``rel_spans`` is their combined character spans relative to
        *sent_start* — triggers first, then arguments.
    """
    t_in = [
        t for t in triggers
        if not (t["trigger_end_span"] <= sent_start or t["trigger_start_span"] >= sent_end)
    ]
    a_in = [
        a for a in args
        if not (a["arg_end_span"] <= sent_start or a["arg_start_span"] >= sent_end)
    ]
    # Offset spans to be relative to the sentence boundary
    rel_spans: list[tuple[int, int]] = (
        [(t["trigger_start_span"] - sent_start, t["trigger_end_span"] - sent_start) for t in t_in]
        + [(a["arg_start_span"] - sent_start, a["arg_end_span"] - sent_start) for a in a_in]
    )
    return t_in, a_in, rel_spans


def get_replacements(
    ft_model: fasttext.FastText._FastText,
    token: str,
    top_k: int = 3,
    sim_thresh: float = 0.7,
) -> list[str]:
    """Retrieve semantically similar substitutes for *token*.

    Candidates are filtered to retain only pure-alphabetic Spanish words
    whose cosine similarity meets *sim_thresh* and which are not morphological
    prefixes or suffixes of *token* (substring containment check).

    Args:
        ft_model: Loaded FastText model.
        token: Source word to replace.
        top_k: Maximum number of replacements to return.
        sim_thresh: Cosine similarity threshold for candidate acceptance.

    Returns:
        Up to *top_k* replacement strings.
    """
    # Over-fetch by 5× to allow for filtering losses
    neighbors = ft_model.get_nearest_neighbors(token.lower(), k=top_k * 5)
    candidates: list[str] = []
    for score, word in neighbors:
        if score < sim_thresh:
            break
        if not _ALPHA_RE.fullmatch(word):
            continue
        tl, wl = token.lower(), word.lower()
        if tl in wl or wl in tl:  # skip morphological variants
            continue
        candidates.append(word)
        if len(candidates) >= top_k:
            break
    return candidates


def augment_sentence(
    sent_text: str,
    token_spans: list[tuple[int, int]],
    nlp: spacy.Language,
    ft_model: fasttext.FastText._FastText,
    top_k: int = 3,
    sim_thresh: float = 0.7,
) -> str:
    """Produce one augmented copy of *sent_text* via entity-preserving substitution.

    Tokens that fall within any span in *token_spans* (triggers or arguments)
    are kept verbatim to preserve label integrity. All remaining content tokens
    — excluding stop words, proper nouns, numerals, and punctuation — are
    candidates for replacement with a FastText nearest-neighbour.

    Args:
        sent_text: Plain text of the source sentence.
        token_spans: Token-index spans ``(start, end)`` of protected entities
            (exclusive end).
        nlp: Loaded spaCy pipeline for tokenisation and POS tagging.
        ft_model: Loaded FastText model for nearest-neighbour retrieval.
        top_k: Maximum number of replacement candidates per token.
        sim_thresh: Minimum cosine similarity for candidate acceptance.

    Returns:
        The augmented sentence as a whitespace-joined string.
    """
    doc = nlp(sent_text)

    # Mark entity token positions as protected
    protected = [False] * len(doc)
    for start, end in token_spans:
        for i in range(start, min(end, len(doc))):
            protected[i] = True

    stopwords: set[str] = nlp.Defaults.stop_words  # type: ignore[assignment]
    new_tokens: list[str] = []

    for i, tok in enumerate(doc):
        word = tok.text
        # Preserve entities, function words, and morphosyntactically ineligible tokens
        if (
            protected[i]
            or not tok.is_alpha
            or word.lower() in stopwords
            or tok.pos_ in {"PROPN", "NUM", "PUNCT"}
        ):
            new_tokens.append(word)
            continue

        candidates = get_replacements(ft_model, word, top_k=top_k, sim_thresh=sim_thresh)
        if not candidates:
            new_tokens.append(word)
        else:
            choice = random.choice(candidates)
            # Preserve original capitalisation pattern
            if word[0].isupper():
                choice = choice.capitalize()
            new_tokens.append(choice)

    return " ".join(new_tokens)
