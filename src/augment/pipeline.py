"""Corpus-level augmentation pipeline.

Iterates over all annotated documents in the data directory, applies
sentence-level entity-preserving substitution (see :mod:`augment.augment`),
and writes augmented ``.txt``, ``_triggers.json``, and ``_args.json`` files.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import spacy
import spacy.tokens
import fasttext  # type: ignore[import-untyped]

from augment.augment import (
    AugConfig,
    ArgumentAnnotation,
    TriggerAnnotation,
    augment_sentence,
    char_spans_to_token_spans,
    iter_doc_ids,
    load_annotations,
    save_annotations,
    sentence_char_spans,
)

logger = logging.getLogger(__name__)


def _load_models(cfg: AugConfig) -> tuple[spacy.Language, fasttext.FastText._FastText]:
    """Load spaCy and FastText models specified in *cfg*.

    Args:
        cfg: Augmentation configuration holding model identifiers/paths.

    Returns:
        A ``(nlp, ft)`` tuple ready for inference.
    """
    nlp = spacy.load(cfg.spacy_model)
    ft = fasttext.load_model(cfg.fasttext_model_path)
    return nlp, ft


def _build_trigger_record(
    doc_id: str,
    aug_sent: str,
    aug_doc: spacy.tokens.Doc,
    tok_span: tuple[int, int],
    original: dict,
) -> dict:
    """Construct a :class:`TriggerAnnotation` dict for one augmented trigger.

    Args:
        doc_id: Source document identifier.
        aug_sent: Full augmented sentence string.
        aug_doc: spaCy-tokenised augmented sentence.
        tok_span: ``(start_token, end_token)`` in *aug_doc* (exclusive end).
        original: Original trigger annotation dict from the corpus.

    Returns:
        Serialised trigger annotation aligned to *aug_sent*.
    """
    st, ed = tok_span
    char_start = aug_doc[st].idx
    char_end = aug_doc[ed - 1].idx + len(aug_doc[ed - 1].text)
    return TriggerAnnotation(
        doc_id=doc_id,
        trigger_id=original["trigger_id"],
        trigger_type=original.get("trigger_type", ""),
        trigger_text=aug_sent[char_start:char_end].strip(),
        trigger_start_span=char_start,
        trigger_end_span=char_end,
    ).to_dict()


def _build_arg_record(
    doc_id: str,
    aug_sent: str,
    aug_doc: spacy.tokens.Doc,
    tok_span: tuple[int, int],
    original: dict,
) -> dict:
    """Construct an :class:`ArgumentAnnotation` dict for one augmented argument.

    Args:
        doc_id: Source document identifier.
        aug_sent: Full augmented sentence string.
        aug_doc: spaCy-tokenised augmented sentence.
        tok_span: ``(start_token, end_token)`` in *aug_doc* (exclusive end).
        original: Original argument annotation dict from the corpus.

    Returns:
        Serialised argument annotation aligned to *aug_sent*.
    """
    st, ed = tok_span
    char_start = aug_doc[st].idx
    char_end = aug_doc[ed - 1].idx + len(aug_doc[ed - 1].text)
    return ArgumentAnnotation(
        doc_id=doc_id,
        arg_id=original["arg_id"],
        arg_type=original.get("arg_type", ""),
        arg_text=aug_sent[char_start:char_end].strip(),
        arg_start_span=char_start,
        arg_end_span=char_end,
    ).to_dict()


def run_augmentation(cfg: AugConfig) -> None:
    """Execute the full augmentation pipeline for all documents in *cfg*.

    For each document the pipeline:

    1. Sentence-segments the full text with spaCy.
    2. Retains sentences containing at least one annotated trigger or argument.
    3. Applies entity-preserving context substitution via FastText.
    4. Re-aligns annotation spans to the augmented surface form.
    5. Writes ``<doc_id>_<sent_idx>.txt``, ``..._triggers.json``, and
       ``..._args.json`` to *cfg.output_dir*.

    Args:
        cfg: Augmentation configuration.
    """
    random.seed(cfg.random_seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)

    nlp, ft = _load_models(cfg)
    doc_ids = iter_doc_ids(data_dir)
    logger.info("Found %d documents in %s", len(doc_ids), data_dir)

    for doc_id in doc_ids:
        text = (data_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        triggers = load_annotations(data_dir / f"{doc_id}_triggers.json")
        args = load_annotations(data_dir / f"{doc_id}_args.json")

        sent_idx = 1
        for sent in nlp(text).sents:
            s_start, s_end = sent.start_char, sent.end_char
            t_in, a_in, rel_char_spans = sentence_char_spans(
                triggers, args, s_start, s_end
            )
            if not (t_in or a_in):
                continue

            sent_doc = nlp(sent.text)
            token_spans = char_spans_to_token_spans(sent_doc, rel_char_spans)

            aug_sent = augment_sentence(
                sent.text, token_spans, nlp=nlp, ft_model=ft,
                top_k=cfg.top_k, sim_thresh=cfg.sim_thresh,
            )
            aug_doc = nlp(aug_sent)

            # token_spans ordering: triggers first, then arguments
            trig_spans = token_spans[: len(t_in)]
            arg_spans = token_spans[len(t_in):]

            out_trigs = [
                _build_trigger_record(doc_id, aug_sent, aug_doc, span, t)
                for span, t in zip(trig_spans, t_in)
            ]
            out_args = [
                _build_arg_record(doc_id, aug_sent, aug_doc, span, a)
                for span, a in zip(arg_spans, a_in)
            ]

            base = f"{doc_id}_{sent_idx}"
            (out_dir / f"{base}.txt").write_text(aug_sent, encoding="utf-8")
            save_annotations(out_trigs, out_dir / f"{base}_triggers.json")
            save_annotations(out_args, out_dir / f"{base}_args.json")
            sent_idx += 1

        logger.info("  %s → %d augmented sentences", doc_id, sent_idx - 1)

    logger.info("Augmentation complete. Output written to %s", out_dir)
