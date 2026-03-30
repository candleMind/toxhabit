"""CLI for supervised training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from inowj.dataprep.loader import load_full_dataframe
from inowj.dataprep.dataset import MultiTaskDatasetWithDict
from inowj.dataprep.collate import multitask_collate_fn_dict
from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict
from inowj.model.optim import build_optimizer_and_scheduler
from inowj.resources.dict_builder import build_dicts
from inowj.train.trainer import train_model
from inowj.utils.seed import resolve_device, set_global_seed
from inowj.utils.logging import WandbConfig, maybe_init_wandb


def main() -> None:
    """Entry point: parse config, build data/model, and run supervised training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))

    # Build dicts
    res_cfg = cfg["resources"]
    dict_bundle = build_dicts(
        trigger_dict_json=res_cfg["trigger_dict_json"],
        argument_dict_json=res_cfg["argument_dict_json"]
    )

    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"])

    # Load training split (single split for simplicity; self-training uses a separate CLI)
    data_cfg = cfg["data"]
    train_df, mappings = load_full_dataframe(
        data_cfg["train_trigger_csv"], data_cfg["train_arg_csv"], return_mappings=True
    )
    tr2id = mappings.tr2id
    arg2id = mappings.arg2id
    o_tr = int(tr2id["O"])
    o_arg = int(arg2id["O"])

    train_grouped = train_df.groupby(["doc_id", "sentence_id"])
    train_dataset = MultiTaskDatasetWithDict(
        grouped_data=train_grouped,
        tokenizer=tokenizer,
        o_trigger_id=o_tr,
        o_arg_id=o_arg,
        trigger_automaton=dict_bundle.trigger_automaton,
        argument_automaton=dict_bundle.argument_automaton,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: multitask_collate_fn_dict(b, tokenizer=tokenizer, o_tr=o_tr, o_arg=o_arg),
    )

    model = BertMultiTaskLstmCrfDict(
        model_name=model_cfg["pretrained_name"],
        trigger_num_labels=len(tr2id),
        arg_num_labels=len(arg2id),
        o_trigger_id=o_tr,
        o_arg_id=o_arg,
        lstm_hidden_size=int(model_cfg.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(model_cfg.get("lstm_num_layers", 2)),
        lstm_dropout=float(model_cfg.get("lstm_dropout", 0.2)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
        dict_feat_dim=int(model_cfg.get("dict_feat_dim", 10)),
        k=int(model_cfg.get("dict_mlp_out_dim", 128)),
    ).to(device)

    opt_cfg = cfg["optimizer"]
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        epochs=int(cfg["training"]["epochs"]),
        bert_lr=float(opt_cfg.get("bert_lr", 2e-5)),
        lstm_lr=float(opt_cfg.get("lstm_lr", 5e-4)),
        clf_lr=float(opt_cfg.get("clf_lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(opt_cfg.get("warmup_ratio", 0.1)),
    )

    out_dir = cfg["training"]["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.pt")

    wandb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        wb_cfg = cfg["wandb"]
        wandb_run = maybe_init_wandb(
            WandbConfig(
                enabled=True,
                project=wb_cfg.get("project", ""),
                run_name=wb_cfg.get("run_name", ""),
                config=cfg,
            )
        )

    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(cfg["training"]["epochs"]),
        save_path=ckpt_path,
        log_every=int(cfg["training"].get("log_every_steps", 500)),
        max_grad_norm=float(opt_cfg.get("max_grad_norm", 1.0)),
        wandb_run=wandb_run,
        extra_state={"tr2id": tr2id, "arg2id": arg2id},
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
