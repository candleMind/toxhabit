"""CLI for evaluation + export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from inowj.dataprep.loader import load_full_dataframe
from inowj.eval.export import run_complete_evaluation
from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict
from inowj.resources.dict_builder import build_dicts
from inowj.utils.seed import resolve_device, set_global_seed


def main() -> None:
    """Entry point: parse config, load model checkpoint, and run evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint (overrides config)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Get checkpoint path from config or command line
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = cfg.get("checkpoint", {}).get("model_path")
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided either via --checkpoint argument or in config file under checkpoint.model_path")

    set_global_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))

    res_cfg = cfg["resources"]
    dict_bundle = build_dicts(
        trigger_dict_json=res_cfg["trigger_dict_json"],
        argument_dict_json=res_cfg["argument_dict_json"]
    )

    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"])

    # Load test df and mappings from train (so label ids match)
    data_cfg = cfg["data"]
    train_df, mappings = load_full_dataframe(
        data_cfg["train_trigger_csv"], data_cfg["train_arg_csv"], return_mappings=True
    )
    test_df = load_full_dataframe(data_cfg["test_trigger_csv"], data_cfg["test_arg_csv"], return_mappings=False)

    o_tr = mappings.tr2id["O"]
    o_arg = mappings.arg2id["O"]

    model = BertMultiTaskLstmCrfDict(
        model_name=model_cfg["pretrained_name"],
        trigger_num_labels=len(mappings.tr2id),
        arg_num_labels=len(mappings.arg2id),
        o_trigger_id=o_tr,
        o_arg_id=o_arg,
        lstm_hidden_size=int(model_cfg.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(model_cfg.get("lstm_num_layers", 2)),
        lstm_dropout=float(model_cfg.get("lstm_dropout", 0.2)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
        dict_feat_dim=int(model_cfg.get("dict_feat_dim", 10)),
        k=int(model_cfg.get("dict_mlp_out_dim", 128)),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    out_dir = Path(cfg["training"]["output_dir"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_complete_evaluation(
        model=model,
        eval_df=test_df,
        device=device,
        tokenizer=tokenizer,
        output_dir=str(out_dir),
        id2tr=mappings.id2tr,
        id2arg=mappings.id2arg,
        o_tr=o_tr,
        o_arg=o_arg,
        trigger_automaton=dict_bundle.trigger_automaton,
        argument_automaton=dict_bundle.argument_automaton,
        gold_dir=cfg["evaluation"]["gold_dir"],
    )

    print(metrics)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()