"""Inference utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from inowj.dataprep.collate import multitask_collate_fn_dict
from inowj.dataprep.dataset import MultiTaskDatasetWithDict
from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict


def infer_token_df_same_logic(
    model: BertMultiTaskLstmCrfDict,
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    id2tr: dict[int, str],
    id2arg: dict[int, str],
    o_tr: int,
    o_arg: int,
    trigger_automaton: Any,
    argument_automaton: Any,
    batch_size: int = 16,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """Run inference on the token-level dataframe with the same logic as training.

    Args:
        model: Trained model.
        df: Token-level dataframe with label ids.
        tokenizer: Tokenizer.
        device: Device.
        id2tr: Trigger id->label mapping.
        id2arg: Argument id->label mapping.
        o_tr: Trigger "O" label id.
        o_arg: Argument "O" label id.
        trigger_automaton: Trigger Aho automaton.
        argument_automaton: Argument Aho automaton.
        batch_size: Batch size.
        out_csv: If set, writes the output CSV.

    Returns:
        DataFrame with `pred_trigger` and `pred_arg` columns.
    """
    df = df.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)
    grouped = df.groupby(["doc_id", "sentence_id"])

    dataset = MultiTaskDatasetWithDict(
        grouped_data=grouped,
        tokenizer=tokenizer,
        o_trigger_id=o_tr,
        o_arg_id=o_arg,
        trigger_automaton=trigger_automaton,
        argument_automaton=argument_automaton,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: multitask_collate_fn_dict(b, tokenizer=tokenizer, o_tr=o_tr, o_arg=o_arg),
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    model.eval()
    flat_tr: list[str] = []
    flat_arg: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dict_feats = batch["dict_feats"].to(device)

            out = model(input_ids, attention_mask, dict_feats=dict_feats)
            batch_trig_preds = out["trigger_preds"]
            batch_arg_preds = out["arg_preds"]

            batch_input_ids = batch["input_ids"].tolist()
            batch_trigger_labels = batch["trigger_labels"].tolist()

            for trig_pred_ids, arg_pred_ids, trig_lbl_ids, seq_input in zip(
                batch_trig_preds, batch_arg_preds, batch_trigger_labels, batch_input_ids
            ):
                for p_tr, p_ar, t_lbl, tok in zip(trig_pred_ids, arg_pred_ids, trig_lbl_ids, seq_input):
                    if t_lbl != -100 and tok not in (cls_id, sep_id):
                        flat_tr.append(id2tr[int(p_tr)])
                        flat_arg.append(id2arg[int(p_ar)])

    ordered_tr_idx: list[int] = []
    ordered_arg_idx: list[int] = []

    for _, g in df.groupby(["doc_id", "sentence_id"]):
        for idx, lbl in zip(g.index, g["trigger_label_id"]):
            if int(lbl) != -100:
                ordered_tr_idx.append(int(idx))
        for idx, lbl in zip(g.index, g["arg_label_id"]):
            if int(lbl) != -100:
                ordered_arg_idx.append(int(idx))

    df_out = df.copy()
    df_out["pred_trigger"] = "O"
    df_out["pred_arg"] = "O"

    n_tr = min(len(ordered_tr_idx), len(flat_tr))
    n_arg = min(len(ordered_arg_idx), len(flat_arg))

    df_out.loc[ordered_tr_idx[:n_tr], "pred_trigger"] = flat_tr[:n_tr]
    df_out.loc[ordered_arg_idx[:n_arg], "pred_arg"] = flat_arg[:n_arg]

    if out_csv is not None:
        df_out.to_csv(out_csv, index=False)

    return df_out
