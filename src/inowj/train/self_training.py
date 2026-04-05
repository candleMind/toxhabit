"""Self-training pipeline (quality-aware filtering by char-span agreement)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from inowj.dataprep.collate import multitask_collate_fn_dict
from inowj.dataprep.dataset import MultiTaskDatasetWithDict
from inowj.eval.export import merge_subwords_to_word_df_keep_cols, run_complete_evaluation
from inowj.eval.report import log_detailed_metrics
from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict
from inowj.model.optim import build_optimizer_and_scheduler
from inowj.train.infer import infer_token_df_same_logic
from inowj.train.trainer import train_model
from inowj.utils.logging import maybe_wandb_log

def extract_entities_charspan_from_bio(
    df_word: pd.DataFrame, label_col: str
) -> set[tuple[str, int, int]]:
    """Extract entity character spans from a word-level BIO-tagged DataFrame.

    Args:
        df_word: Word-level DataFrame with columns ``start``, ``end``,
            and the column named by ``label_col`` containing BIO tags.
        label_col: Column name holding the BIO label strings.

    Returns:
        Set of (entity_type, char_start, char_end) tuples.
    """
    ents: set[tuple[str, int, int]] = set()
    i = 0
    df_word = df_word.reset_index(drop=True)

    while i < len(df_word):
        lab = df_word.loc[i, label_col]
        if isinstance(lab, str) and lab.startswith("B-"):
            typ = lab[2:]
            s = int(df_word.loc[i, "start"])
            e = int(df_word.loc[i, "end"])
            j = i + 1
            while j < len(df_word) and df_word.loc[j, label_col] == f"I-{typ}":
                e = int(df_word.loc[j, "end"])
                j += 1
            ents.add((typ, s, e))
            i = j
        else:
            i += 1

    return ents


def filter_augmented_by_agreement_charspan_only(
    df_aug_pred_token: pd.DataFrame, require_agreement: bool = True
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Filter augmented sentences by exact agreement between gold and predicted spans.

    A sentence passes the quality filter if and only if its character-span sets
    for both triggers and arguments are identical between the gold annotation
    and the model prediction (when ``require_agreement`` is True).

    Args:
        df_aug_pred_token: Token-level DataFrame containing gold labels
            (``trigger_label``, ``arg_label``) and model predictions
            (``pred_trigger``, ``pred_arg``).
        require_agreement: If True, both trigger and argument spans must match
            to retain a sentence. If False, all sentences are kept.

    Returns:
        filtered: Token-level DataFrame with only the accepted sentences.
        stats: Dictionary of filtering statistics.
        df_word: Word-level DataFrame used for span comparison (for debugging).
    """
    df_word = merge_subwords_to_word_df_keep_cols(df_aug_pred_token)

    stats = {
        "total_sentences": 0,
        "pass_sentences": 0,
        "trigger_agree": 0,
        "arg_agree": 0,
        "both_agree": 0,
    }

    keep: list[tuple[str, int]] = []
    grouped = list(df_word.groupby(["doc_id", "sentence_id"]))
    stats["total_sentences"] = len(grouped)

    for (doc_id, sent_id), g in tqdm(grouped, desc="Filtering aug (agreement only)", leave=False):
        gold_tr = extract_entities_charspan_from_bio(g, "trigger_label")
        gold_ar = extract_entities_charspan_from_bio(g, "arg_label")
        pred_tr = extract_entities_charspan_from_bio(g, "pred_trigger")
        pred_ar = extract_entities_charspan_from_bio(g, "pred_arg")

        trig_agree = gold_tr == pred_tr
        arg_agree = gold_ar == pred_ar

        if trig_agree:
            stats["trigger_agree"] += 1
        if arg_agree:
            stats["arg_agree"] += 1
        if trig_agree and arg_agree:
            stats["both_agree"] += 1

        ok = (trig_agree and arg_agree) if require_agreement else True
        if ok:
            keep.append((str(doc_id), int(sent_id)))

    if not keep:
        return df_aug_pred_token.iloc[0:0].copy(), stats, df_word

    keep_df = pd.DataFrame(keep, columns=["doc_id", "sentence_id"]).drop_duplicates()
    filtered = df_aug_pred_token.merge(keep_df, on=["doc_id", "sentence_id"], how="inner")

    stats["pass_sentences"] = len(keep_df)
    return filtered, stats, df_word


def run_self_training(
    model_baseline: BertMultiTaskLstmCrfDict,
    num_iterations: int,
    gold_train_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_base_dir: str,
    tokenizer_name: str,
    label_maps: dict[str, Any],
    dict_bundle: dict[str, Any],
    device: torch.device,
    baseline_epochs: int = 5,
    iter1_epochs: int = 5,
    iter2_epochs: int = 5,
    require_agreement: bool = True,
    log_every: int = 500,
    init_checkpoint_path: str | None = None,
    wandb_run: Any | None = None,
    optimizer_cfg: dict[str, Any] | None = None,
    model_cfg: dict[str, Any] | None = None,
    eval_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run iterative self-training with quality-aware augmentation filtering.

    In each iteration the current teacher model labels the unlabelled augmented
    corpus, retains only sentences where gold annotations and model predictions
    agree (quality filter), concatenates them with the gold training set, and
    trains a new student model from scratch. The test set is evaluated after
    every iteration.

    Args:
        model_baseline: Untrained (or pre-checkpointed) model to use as the
            initial teacher.
        num_iterations: Number of self-training iterations to run.
        gold_train_df: Token-level DataFrame for the gold-labelled training split.
        aug_df: Token-level DataFrame for the unlabelled/augmented corpus.
        test_df: Token-level DataFrame for the held-out test split.
        output_base_dir: Root directory for all iteration outputs.
        tokenizer_name: HuggingFace model name used to load the tokenizer.
        label_maps: Mapping with keys ``tr2id``, ``arg2id``, ``id2tr``, ``id2arg``.
        dict_bundle: Mapping with keys ``A_TR`` and ``A_ARG`` holding the
            trigger and argument Aho-Corasick automata.
        device: Torch device.
        baseline_epochs: Training epochs for the initial baseline model.
        iter1_epochs: Training epochs for the first self-training iteration.
        iter2_epochs: Training epochs for subsequent iterations.
        require_agreement: Quality filter flag passed to
            `filter_augmented_by_agreement_charspan_only`.
        log_every: Log step-level metrics to W&B every N gradient steps.
        init_checkpoint_path: Optional path to a pre-trained baseline checkpoint
            to skip baseline training.
        wandb_run: Optional active W&B run object.
        optimizer_cfg: Optional dict overriding default optimizer hyperparameters.
        model_cfg: Optional dict overriding default model hyperparameters.
        eval_cfg: Config dict; must contain ``gold_dir`` for scorer evaluation.

    Returns:
        List of per-iteration result dictionaries.
    """
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    baseline_dir = os.path.join(output_base_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tr2id = label_maps["tr2id"]
    arg2id = label_maps["arg2id"]
    id2tr = label_maps["id2tr"]
    id2arg = label_maps["id2arg"]
    o_tr = int(tr2id["O"])
    o_arg = int(arg2id["O"])

    trigger_automaton = dict_bundle["A_TR"]
    argument_automaton = dict_bundle["A_ARG"]
    gold_dir = eval_cfg.get("gold_dir") if eval_cfg else None
    if not gold_dir:
        raise ValueError("eval_cfg.gold_dir is required for scoring.")

    # Baseline init / train
    if init_checkpoint_path and os.path.exists(init_checkpoint_path):
        ckpt = torch.load(init_checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model_baseline.load_state_dict(state_dict, strict=True)
    else:
        train_grouped = gold_train_df.groupby(["doc_id", "sentence_id"])
        train_dataset = MultiTaskDatasetWithDict(
            grouped_data=train_grouped,
            tokenizer=tokenizer,
            o_trigger_id=o_tr,
            o_arg_id=o_arg,
            trigger_automaton=trigger_automaton,
            argument_automaton=argument_automaton,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda b: multitask_collate_fn_dict(b, tokenizer=tokenizer, o_tr=o_tr, o_arg=o_arg),
        )

        opt_cfg = optimizer_cfg or {}
        optimizer, scheduler = build_optimizer_and_scheduler(
            model_baseline,
            train_loader=train_loader,
            epochs=baseline_epochs,
            bert_lr=float(opt_cfg.get("bert_lr", 2e-5)),
            lstm_lr=float(opt_cfg.get("lstm_lr", 5e-4)),
            clf_lr=float(opt_cfg.get("clf_lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            warmup_ratio=float(opt_cfg.get("warmup_ratio", 0.1)),
        )

        baseline_ckpt_path = os.path.join(baseline_dir, "best_model.pt")
        _hist, _best = train_model(
            model=model_baseline,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=baseline_epochs,
            save_path=baseline_ckpt_path,
            log_every=log_every,
            max_grad_norm=float(opt_cfg.get("max_grad_norm", 1.0)),
            wandb_run=wandb_run,
            extra_state={"tr2id": tr2id, "arg2id": arg2id},
        )

        ckpt = torch.load(baseline_ckpt_path, map_location=device)
        model_baseline.load_state_dict(ckpt["model_state_dict"])

        baseline_test_dir = os.path.join(baseline_dir, "test_predictions")
        baseline_metrics = run_complete_evaluation(
            model=model_baseline,
            eval_df=test_df,
            device=device,
            tokenizer=tokenizer,
            output_dir=baseline_test_dir,
            id2tr=id2tr,
            id2arg=id2arg,
            o_tr=o_tr,
            o_arg=o_arg,
            trigger_automaton=trigger_automaton,
            argument_automaton=argument_automaton,
            gold_dir=gold_dir,
        )
        log_detailed_metrics(baseline_metrics, "BASELINE TEST")
        with open(os.path.join(baseline_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)

    all_results: list[dict[str, Any]] = []

    for iteration in range(1, num_iterations + 1):
        iter_dir = os.path.join(output_base_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        epochs = iter1_epochs if iteration == 1 else iter2_epochs

        # Step 1: Infer augmented data with teacher
        aug_pred_csv = os.path.join(iter_dir, "aug_token_predictions.csv")
        df_aug_pred_token = infer_token_df_same_logic(
            model=model_baseline,
            df=aug_df,
            tokenizer=tokenizer,
            device=device,
            id2tr=id2tr,
            id2arg=id2arg,
            o_tr=o_tr,
            o_arg=o_arg,
            trigger_automaton=trigger_automaton,
            argument_automaton=argument_automaton,
            batch_size=16,
            out_csv=aug_pred_csv,
        )
        aug_token_count_before = len(df_aug_pred_token)
        aug_sent_count_before = df_aug_pred_token.groupby(["doc_id", "sentence_id"]).ngroups

        maybe_wandb_log(
            wandb_run,
            {
                f"iter{iteration}/aug_before_filter_sentences": aug_sent_count_before,
                f"iter{iteration}/aug_before_filter_tokens": aug_token_count_before,
            },
        )

        # Step 2: Filter by agreement
        filtered_aug_token, filter_stats, df_aug_word_debug = filter_augmented_by_agreement_charspan_only(
            df_aug_pred_token,
            require_agreement=require_agreement,
        )

        aug_token_count_after = len(filtered_aug_token)
        aug_sent_count_after = (
            filtered_aug_token.groupby(["doc_id", "sentence_id"]).ngroups if len(filtered_aug_token) > 0 else 0
        )

        maybe_wandb_log(
            wandb_run,
            {
                f"iter{iteration}/aug_after_filter_sentences": aug_sent_count_after,
                f"iter{iteration}/aug_after_filter_tokens": aug_token_count_after,
            },
        )

        with open(os.path.join(iter_dir, "filter_stats.json"), "w", encoding="utf-8") as f:
            json.dump(filter_stats, f, indent=2)
        df_aug_word_debug.to_csv(os.path.join(iter_dir, "aug_word_debug.csv"), index=False)

        if len(filtered_aug_token) == 0:
            train_df = gold_train_df.copy()
            filtered_count = 0
        else:
            filtered_aug_token = filtered_aug_token.copy()
            filtered_aug_token["doc_id"] = "aug__" + filtered_aug_token["doc_id"].astype(str)
            filtered_aug_token.to_csv(os.path.join(iter_dir, "filtered_aug_token.csv"), index=False)

            train_df = pd.concat([gold_train_df, filtered_aug_token], ignore_index=True)
            filtered_count = int(filtered_aug_token.groupby(["doc_id", "sentence_id"]).ngroups)

        # Step 3: Train student model from scratch on gold + filtered augmented data.
        model_opts = model_cfg or {}
        model_iter = BertMultiTaskLstmCrfDict(
            model_name=tokenizer_name,
            trigger_num_labels=len(tr2id),
            arg_num_labels=len(arg2id),
            o_trigger_id=o_tr,
            o_arg_id=o_arg,
            lstm_hidden_size=int(model_opts.get("lstm_hidden_size", 256)),
            lstm_num_layers=int(model_opts.get("lstm_num_layers", 2)),
            lstm_dropout=float(model_opts.get("lstm_dropout", 0.2)),
            bidirectional=bool(model_opts.get("bidirectional", True)),
            dict_feat_dim=int(model_opts.get("dict_feat_dim", 10)),
            k=int(model_opts.get("dict_mlp_out_dim", 128)),
        ).to(device)

        train_grouped = train_df.groupby(["doc_id", "sentence_id"])
        train_dataset = MultiTaskDatasetWithDict(
            grouped_data=train_grouped,
            tokenizer=tokenizer,
            o_trigger_id=o_tr,
            o_arg_id=o_arg,
            trigger_automaton=trigger_automaton,
            argument_automaton=argument_automaton,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda b: multitask_collate_fn_dict(b, tokenizer=tokenizer, o_tr=o_tr, o_arg=o_arg),
        )

        opt_cfg = optimizer_cfg or {}
        optimizer, scheduler = build_optimizer_and_scheduler(
            model_iter,
            train_loader=train_loader,
            epochs=epochs,
            bert_lr=float(opt_cfg.get("bert_lr", 2e-5)),
            lstm_lr=float(opt_cfg.get("lstm_lr", 5e-4)),
            clf_lr=float(opt_cfg.get("clf_lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            warmup_ratio=float(opt_cfg.get("warmup_ratio", 0.1)),
        )

        save_path = os.path.join(iter_dir, "best_model.pt")
        hist, best_loss = train_model(
            model=model_iter,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            save_path=save_path,
            log_every=log_every,
            max_grad_norm=float(opt_cfg.get("max_grad_norm", 1.0)),
            wandb_run=wandb_run,
            extra_state={"tr2id": tr2id, "arg2id": arg2id},
        )

        ckpt = torch.load(save_path, map_location=device)
        model_iter.load_state_dict(ckpt["model_state_dict"])

        # Step 4: Evaluate
        test_out_dir = os.path.join(iter_dir, "test_predictions")
        test_metrics = run_complete_evaluation(
            model=model_iter,
            eval_df=test_df,
            device=device,
            tokenizer=tokenizer,
            output_dir=test_out_dir,
            id2tr=id2tr,
            id2arg=id2arg,
            o_tr=o_tr,
            o_arg=o_arg,
            trigger_automaton=trigger_automaton,
            argument_automaton=argument_automaton,
            gold_dir=gold_dir,
        )
        log_detailed_metrics(test_metrics, "TEST", iteration=iteration)

        iter_results = {
            "iteration": iteration,
            "require_agreement": require_agreement,
            "filtered_aug_sentences": int(filtered_count),
            "train_tokens": int(len(train_df)),
            "best_loss": float(best_loss),
            "test_metrics": test_metrics,
            "filter_stats": filter_stats,
            "aug_before_filter_sentences": int(aug_sent_count_before),
            "aug_before_filter_tokens": int(aug_token_count_before),
            "aug_after_filter_sentences": int(aug_sent_count_after),
            "aug_after_filter_tokens": int(aug_token_count_after),
        }
        with open(os.path.join(iter_dir, "iteration_results.json"), "w", encoding="utf-8") as f:
            json.dump(iter_results, f, indent=2)

        all_results.append(iter_results)

        maybe_wandb_log(
            wandb_run,
            {
                "iteration": iteration,
                f"iter{iteration}/filtered_aug_sentences": float(filtered_count),
                f"iter{iteration}/best_loss": float(best_loss),
                f"iter{iteration}/trigger_f1": float(test_metrics["trigger_f1"]),
                f"iter{iteration}/arg_f1": float(test_metrics["arg_f1"]),
                f"iter{iteration}/trigger_macro_f1": float(test_metrics["trigger_macro_f1"]),
                f"iter{iteration}/arg_macro_f1": float(test_metrics["arg_macro_f1"]),
            },
        )

    with open(os.path.join(output_base_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results
