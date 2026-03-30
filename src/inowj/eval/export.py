"""Prediction post-processing and export utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from inowj.dataprep.collate import multitask_collate_fn_dict
from inowj.dataprep.dataset import MultiTaskDatasetWithDict
from inowj.eval.scorer_exact import evaluate_argument_f1, evaluate_trigger_f1
from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict


def merge_subwords_to_word_df_keep_cols(df_token: pd.DataFrame) -> pd.DataFrame:
    """Merge WordPiece subword rows into whole-word rows, retaining key annotation columns.

    Args:
        df_token: Token-level DataFrame with columns ``doc_id``, ``sentence_id``,
            ``token_id``, ``token``, ``is_subword``, ``start``, ``end``,
            and optional prediction/label columns.

    Returns:
        Word-level DataFrame with one row per head token (non-subword).
    """
    df_token = df_token.sort_values(["doc_id", "sentence_id", "token_id"]).reset_index(drop=True)

    merged_rows: list[dict[str, Any]] = []
    i = 0
    while i < len(df_token):
        row = df_token.iloc[i]
        if bool(row["is_subword"]):
            i += 1
            continue

        merged_token = str(row["token"]) if pd.notna(row["token"]) else ""
        start = int(row["start"])
        end = int(row["end"])

        rec = {
            "doc_id": row["doc_id"],
            "sentence_id": row["sentence_id"],
            "token": merged_token,
            "start": start,
            "end": end,
            "trigger_label": row.get("trigger_label", "O"),
            "arg_label": row.get("arg_label", "O"),
            "pred_trigger": row.get("pred_trigger", "O"),
            "pred_arg": row.get("pred_arg", "O"),
            "pred_trigger_conf": float(row.get("pred_trigger_conf", 0.0)),
            "pred_arg_conf": float(row.get("pred_arg_conf", 0.0)),
        }

        j = i + 1
        while (
            j < len(df_token)
            and bool(df_token.iloc[j]["is_subword"])
            and df_token.iloc[j]["doc_id"] == row["doc_id"]
            and df_token.iloc[j]["sentence_id"] == row["sentence_id"]
        ):
            sub = df_token.iloc[j]["token"]
            if pd.notna(sub):
                rec["token"] += str(sub).replace("##", "")
            rec["end"] = int(df_token.iloc[j]["end"])
            j += 1

        merged_rows.append(rec)
        i = j

    return pd.DataFrame(merged_rows)


def export_doc_jsons_from_word_df(
    df_word: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str]:
    """Export word-level predictions to per-document JSON files.

    Produces two subdirectories: ``triggers_output/`` and ``arguments_output/``,
    each containing one JSON file per document with BIO spans decoded into
    entity records (type, text, start/end character offsets).

    Args:
        df_word: Word-level DataFrame with columns ``doc_id``, ``sentence_id``,
            ``token``, ``start``, ``end``, ``pred_trigger``, and ``pred_arg``.
        output_dir: Root directory under which output subdirectories are created.

    Returns:
        Tuple of (trigger_output_dir, argument_output_dir) paths.
    """
    output_dir = str(output_dir)
    trig_dir = os.path.join(output_dir, "triggers_output")
    arg_dir = os.path.join(output_dir, "arguments_output")
    os.makedirs(trig_dir, exist_ok=True)
    os.makedirs(arg_dir, exist_ok=True)

    for doc_id, group in df_word.groupby("doc_id"):
        group = group.reset_index(drop=True)

        triggers: dict[str, dict[str, Any]] = {}
        trig_idx = 1
        i = 0
        while i < len(group):
            label = group.loc[i, "pred_trigger"]
            if isinstance(label, str) and label.startswith("B-"):
                typ = label[2:]
                text = str(group.loc[i, "token"]) if pd.notna(group.loc[i, "token"]) else ""
                start = int(group.loc[i, "start"])
                end = int(group.loc[i, "end"])

                j = i + 1
                while j < len(group) and group.loc[j, "pred_trigger"] == f"I-{typ}":
                    nxt = group.loc[j, "token"]
                    if pd.notna(nxt):
                        text += " " + str(nxt)
                    end = int(group.loc[j, "end"])
                    j += 1

                triggers[f"E{trig_idx}"] = {
                    "trigger_type": typ,
                    "trigger_text": text,
                    "trigger_start_span": start,
                    "trigger_end_span": end,
                }
                trig_idx += 1
                i = j
            else:
                i += 1

        with open(os.path.join(trig_dir, f"{doc_id}_triggers.json"), "w", encoding="utf-8") as f:
            json.dump(triggers, f, ensure_ascii=False, indent=2)

        arguments: dict[str, dict[str, Any]] = {}
        arg_idx = 1
        i = 0
        while i < len(group):
            label = group.loc[i, "pred_arg"]
            if isinstance(label, str) and label.startswith("B-"):
                typ = label[2:]
                text = str(group.loc[i, "token"]) if pd.notna(group.loc[i, "token"]) else ""
                start = int(group.loc[i, "start"])
                end = int(group.loc[i, "end"])

                j = i + 1
                while j < len(group) and group.loc[j, "pred_arg"] == f"I-{typ}":
                    nxt = group.loc[j, "token"]
                    if pd.notna(nxt):
                        text += " " + str(nxt)
                    end = int(group.loc[j, "end"])
                    j += 1

                arguments[f"A{arg_idx}"] = {
                    "argument_type": typ,
                    "argument_text": text,
                    "argument_start_span": start,
                    "argument_end_span": end,
                }
                arg_idx += 1
                i = j
            else:
                i += 1

        with open(os.path.join(arg_dir, f"{doc_id}_args.json"), "w", encoding="utf-8") as f:
            json.dump(arguments, f, ensure_ascii=False, indent=2)

    return trig_dir, arg_dir


def run_complete_evaluation(
    model: BertMultiTaskLstmCrfDict,
    eval_df: pd.DataFrame,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    id2tr: dict[int, str],
    id2arg: dict[int, str],
    o_tr: int,
    o_arg: int,
    trigger_automaton: Any,
    argument_automaton: Any,
    gold_dir: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run full inference, export predictions to JSON, and compute F1 scores.

    Args:
        model: Trained multi-task model.
        eval_df: Token-level evaluation DataFrame.
        device: Torch device.
        tokenizer: BERT-compatible tokenizer.
        output_dir: Directory to write CSV and JSON prediction files.
        id2tr: Trigger id -> BIO label mapping.
        id2arg: Argument id -> BIO label mapping.
        o_tr: Trigger "O" label id.
        o_arg: Argument "O" label id.
        trigger_automaton: Aho-Corasick automaton for trigger feature extraction.
        argument_automaton: Aho-Corasick automaton for argument feature extraction.
        gold_dir: Directory containing gold-standard JSON annotation files.
        batch_size: Inference batch size.

    Returns:
        Metrics dict with micro/macro precision, recall, and F1 for both
        triggers and arguments, plus per-type breakdowns.
    """
    eval_grouped = eval_df.groupby(["doc_id", "sentence_id"])
    eval_dataset = MultiTaskDatasetWithDict(
        grouped_data=eval_grouped,
        tokenizer=tokenizer,
        o_trigger_id=o_tr,
        o_arg_id=o_arg,
        trigger_automaton=trigger_automaton,
        argument_automaton=argument_automaton,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: multitask_collate_fn_dict(b, tokenizer=tokenizer, o_tr=o_tr, o_arg=o_arg),
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    model.eval()
    flat_trigger_preds: list[str] = []
    flat_arg_preds: list[str] = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dict_feats = batch["dict_feats"].to(device)

            out = model(input_ids, attention_mask, dict_feats=dict_feats)
            batch_trig_preds = out["trigger_preds"]
            batch_arg_preds = out["arg_preds"]

            batch_input_ids = batch["input_ids"].tolist()
            batch_trigger_labels = batch["trigger_labels"].tolist()
            batch_arg_labels = batch["arg_labels"].tolist()

            for trig_pred_ids, arg_pred_ids, trig_label_ids, arg_label_ids, seq_input in zip(
                batch_trig_preds,
                batch_arg_preds,
                batch_trigger_labels,
                batch_arg_labels,
                batch_input_ids,
            ):
                for p_tr, p_ar, t_lbl, a_lbl, tok in zip(
                    trig_pred_ids, arg_pred_ids, trig_label_ids, arg_label_ids, seq_input
                ):
                    if t_lbl != -100 and tok not in (cls_id, sep_id):
                        flat_trigger_preds.append(id2tr[int(p_tr)])
                        flat_arg_preds.append(id2arg[int(p_ar)])

    ordered_trigger_indices: list[int] = []
    ordered_arg_indices: list[int] = []
    for _, group in eval_df.groupby(["doc_id", "sentence_id"]):
        for idx, lbl in zip(group.index, group["trigger_label_id"]):
            if int(lbl) != -100:
                ordered_trigger_indices.append(int(idx))
        for idx, lbl in zip(group.index, group["arg_label_id"]):
            if int(lbl) != -100:
                ordered_arg_indices.append(int(idx))

    df_out = eval_df.copy()
    df_out["pred_trigger"] = "O"
    df_out["pred_arg"] = "O"

    n_trig = min(len(ordered_trigger_indices), len(flat_trigger_preds))
    n_arg = min(len(ordered_arg_indices), len(flat_arg_preds))

    df_out.loc[ordered_trigger_indices[:n_trig], "pred_trigger"] = flat_trigger_preds[:n_trig]
    df_out.loc[ordered_arg_indices[:n_arg], "pred_arg"] = flat_arg_preds[:n_arg]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(os.path.join(output_dir, "token_predictions.csv"), index=False)

    df_word = merge_subwords_to_word_df_keep_cols(df_out)
    df_word.to_csv(os.path.join(output_dir, "word_prediction.csv"), index=False)

    trig_dir, arg_dir = export_doc_jsons_from_word_df(df_word, output_dir=output_dir)

    trigger_results = evaluate_trigger_f1(trig_dir, gold_dir)
    arg_results = evaluate_argument_f1(arg_dir, gold_dir)

    return {
        "trigger_f1": trigger_results["micro_f1"],
        "trigger_precision": trigger_results["micro_p"],
        "trigger_recall": trigger_results["micro_r"],
        "trigger_macro_p": trigger_results["macro_p"],
        "trigger_macro_r": trigger_results["macro_r"],
        "trigger_macro_f1": trigger_results["macro_f1"],
        "trigger_tp": trigger_results["tp"],
        "trigger_fp": trigger_results["fp"],
        "trigger_fn": trigger_results["fn"],
        "trigger_per_type": trigger_results["per_type"],
        "arg_f1": arg_results["micro_f1"],
        "arg_precision": arg_results["micro_p"],
        "arg_recall": arg_results["micro_r"],
        "arg_macro_p": arg_results["macro_p"],
        "arg_macro_r": arg_results["macro_r"],
        "arg_macro_f1": arg_results["macro_f1"],
        "arg_tp": arg_results["tp"],
        "arg_fp": arg_results["fp"],
        "arg_fn": arg_results["fn"],
        "arg_per_type": arg_results["per_type"],
    }
