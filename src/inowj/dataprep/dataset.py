"""PyTorch datasets used by the I-NOWJ training pipeline."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import ahocorasick
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from inowj.resources.dict_builder import ARG2IDX, ARG_TYPES, DICT_FEAT_DIM, TRIG_TYPES
from inowj.utils.text import (
    abstract_with_offset_map,
    bert_tokens_to_sentence_with_offsets,
    is_whole_word,
)

def match_trigger_features_token_level(
    sentence: str,
    token_offsets: Sequence[tuple[int, int]],
    trigger_automaton: ahocorasick.Automaton,
    trig_types: Sequence[str] = TRIG_TYPES,
) -> tuple[list[list[int]], dict[str, list[tuple[int, int]]]]:
    """Match trigger dictionary terms and build per-token trigger features.

    Args:
        sentence: Reconstructed sentence (raw text).
        token_offsets: List of token (start, end) character spans in `sentence`.
        trigger_automaton: Aho-Corasick automaton storing (cat, term) values.
        trig_types: Trigger categories defining feature order.

    Returns:
        token_features: Per-token one-hot features with shape [T, len(trig_types)].
        trigger_spans: Mapping cat -> list of matched character spans (start, end).
    """
    norm_sent = sentence.lower()
    trigger_spans: dict[str, list[tuple[int, int]]] = {t: [] for t in trig_types}

    for end_idx, (cat, term) in trigger_automaton.iter(norm_sent):
        start = end_idx - len(term) + 1
        end = end_idx + 1

        if not is_whole_word(norm_sent, start, end):
            continue

        if cat in trigger_spans:
            trigger_spans[cat].append((start, end))

    token_features: list[list[int]] = []
    for tok_start, tok_end in token_offsets:
        row: list[int] = []
        for cat in trig_types:
            spans = trigger_spans.get(cat, [])
            # Interval overlap: tok overlaps span iff tok_start < span_end AND tok_end > span_start.
            on = any(tok_start < se and tok_end > ss for ss, se in spans)
            row.append(int(on))
        token_features.append(row)

    return token_features, trigger_spans


def match_argument_patterns_token_level(
    sentence: str,
    token_offsets: list[tuple[int, int]],
    arg_automaton: ahocorasick.Automaton,
) -> list[list[int]]:
    """Match argument dictionary patterns and build per-token argument features.

    Args:
        sentence: Reconstructed sentence (raw, not lowercased).
        token_offsets: List of token (start, end) character spans in `sentence`.
        arg_automaton: Aho-Corasick automaton built on *abstracted* patterns.

    Returns:
        token_features: Per-token one-hot features with shape [T, len(ARG_TYPES)].
    """
    abs_text, abs_to_raw_map = abstract_with_offset_map(sentence)

    arg_matches: list[tuple[str, int, int]] = []
    for end_idx, (arg_type, pattern) in arg_automaton.iter(abs_text):
        abs_start = end_idx - len(pattern) + 1
        abs_end = end_idx + 1

        if abs_start > 0 and abs_text[abs_start - 1].isalnum():
            continue
        if abs_end < len(abs_text) and abs_text[abs_end].isalnum():
            continue

        arg_matches.append((arg_type, abs_start, abs_end))

    raw_arg_spans: dict[str, list[tuple[int, int]]] = {t: [] for t in ARG_TYPES}
    for arg_type, abs_start, abs_end in arg_matches:
        if abs_start >= len(abs_to_raw_map) or abs_end > len(abs_to_raw_map):
            continue

        raw_start = abs_to_raw_map[abs_start][0]
        raw_end = abs_to_raw_map[abs_end - 1][1]
        if arg_type in raw_arg_spans:
            raw_arg_spans[arg_type].append((raw_start, raw_end))

    token_features: list[list[int]] = []
    for tok_start, tok_end in token_offsets:
        # Build one-hot vector in ARG_TYPES order; a token is active if
        # its character span overlaps any matched pattern span.
        feat = [0] * len(ARG_TYPES)
        for arg_type in ARG_TYPES:
            spans = raw_arg_spans.get(arg_type, [])
            for span_start, span_end in spans:
                if tok_start < span_end and tok_end > span_start:
                    feat[ARG2IDX[arg_type]] = 1
                    break
        token_features.append(feat)

    return token_features


class MultiTaskDatasetWithDict(Dataset[dict[str, Any]]):
    """Dataset for multi-task Trigger/Argument sequence labeling with dict features."""

    def __init__(
        self,
        grouped_data: Iterable[tuple[Any, pd.DataFrame]],
        tokenizer: PreTrainedTokenizerBase,
        o_trigger_id: int,
        o_arg_id: int,
        trigger_automaton: ahocorasick.Automaton,
        argument_automaton: ahocorasick.Automaton,
    ) -> None:
        """Pre-build all samples at construction time.

        Args:
            grouped_data: Iterable of (key, group_df) from a pandas GroupBy over sentences.
            tokenizer: BERT-compatible tokenizer.
            o_trigger_id: Label id for the trigger outside tag "O".
            o_arg_id: Label id for the argument outside tag "O".
            trigger_automaton: Aho-Corasick automaton for trigger dictionary matching.
            argument_automaton: Aho-Corasick automaton for argument dictionary matching.
        """
        self.samples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer

        for _, group in grouped_data:
            tokens = group["token"].tolist()
            trigger_labels = group["trigger_label_id"].tolist()
            arg_labels = group["arg_label_id"].tolist()
            if not tokens:
                continue

            sentence, token_offsets = bert_tokens_to_sentence_with_offsets(tokens)

            trig_features, _ = match_trigger_features_token_level(
                sentence=sentence,
                token_offsets=token_offsets,
                trigger_automaton=trigger_automaton,
                trig_types=TRIG_TYPES,
            )

            arg_features = match_argument_patterns_token_level(
                sentence=sentence,
                token_offsets=token_offsets,
                arg_automaton=argument_automaton,
            )

            dict_feats: list[list[int]] = []
            for i in range(len(token_offsets)):
                trig_feat = trig_features[i] if i < len(trig_features) else [0] * len(TRIG_TYPES)
                arg_feat = arg_features[i] if i < len(arg_features) else [0] * len(ARG_TYPES)
                dict_feats.append([int(v) for v in (trig_feat + arg_feat)])

            # Prepend [CLS] with dummy labels; dict features are zero at special tokens.
            input_ids: list[int] = [int(tokenizer.cls_token_id)]
            trigger_lbls: list[int] = [int(o_trigger_id)]
            arg_lbls: list[int] = [int(o_arg_id)]
            dict_feats_final: list[list[int]] = [[0] * DICT_FEAT_DIM]

            for i, (tok, t_lab, a_lab) in enumerate(zip(tokens, trigger_labels, arg_labels)):
                token_id = tokenizer.convert_tokens_to_ids(tok)
                if token_id is None:
                    token_id = tokenizer.unk_token_id

                input_ids.append(int(token_id))
                trigger_lbls.append(int(t_lab))
                arg_lbls.append(int(a_lab))
                dict_feats_final.append(
                    dict_feats[i] if i < len(dict_feats) else [0] * DICT_FEAT_DIM
                )

            input_ids.append(int(tokenizer.sep_token_id))
            trigger_lbls.append(int(o_trigger_id))
            arg_lbls.append(int(o_arg_id))
            dict_feats_final.append([0] * DICT_FEAT_DIM)

            if len(input_ids) < 3:
                continue

            if not (len(input_ids) == len(trigger_lbls) == len(arg_lbls) == len(dict_feats_final)):
                raise ValueError(
                    "Length mismatch: "
                    f"{len(input_ids)} vs {len(trigger_lbls)} vs {len(arg_lbls)} vs {len(dict_feats_final)}"
                )

            attention_mask = [1] * len(input_ids)

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "trigger_labels": trigger_lbls,
                    "arg_labels": arg_lbls,
                    "dict_feats": dict_feats_final,
                    "sentence": sentence,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]