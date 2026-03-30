"""Collate function for multi-task batches."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


def multitask_collate_fn_dict(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    o_tr: int,
    o_arg: int,
) -> dict[str, torch.Tensor]:
    """Pad and batch variable-length sequences.

    Args:
        batch: List of samples returned by `MultiTaskDatasetWithDict`.
        tokenizer: Tokenizer providing pad token id and max length.
        o_tr: "O" label id for triggers.
        o_arg: "O" label id for arguments.

    Returns:
        Dictionary of batched tensors.
    """
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    trig_lbls = [torch.tensor(x["trigger_labels"], dtype=torch.long) for x in batch]
    arg_lbls = [torch.tensor(x["arg_labels"], dtype=torch.long) for x in batch]
    dict_feats = [torch.tensor(x["dict_feats"], dtype=torch.long) for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    trig_lbls = pad_sequence(trig_lbls, batch_first=True, padding_value=o_tr)
    arg_lbls = pad_sequence(arg_lbls, batch_first=True, padding_value=o_arg)
    dict_feats = pad_sequence(dict_feats, batch_first=True, padding_value=0)

    max_len = min(int(input_ids.size(1)), int(tokenizer.model_max_length))
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    trig_lbls = trig_lbls[:, :max_len]
    arg_lbls = arg_lbls[:, :max_len]
    dict_feats = dict_feats[:, :max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "trigger_labels": trig_lbls,
        "arg_labels": arg_lbls,
        "dict_feats": dict_feats,
    }
