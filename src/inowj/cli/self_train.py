"""CLI for self-training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from inowj.dataprep.loader import load_full_dataframe
from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict
from inowj.resources.dict_builder import build_dicts
from inowj.train.self_training import run_self_training
from inowj.utils.logging import WandbConfig, maybe_init_wandb
from inowj.utils.seed import resolve_device, set_global_seed


def main() -> None:
    """Entry point: parse config, build all components, and run self-training."""
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

    # Load dataframes
    data_cfg = cfg["data"]
    gold_df, mappings = load_full_dataframe(
        data_cfg["gold_train_trigger_csv"],
        data_cfg["gold_train_arg_csv"],
        return_mappings=True,
    )
    test_df = load_full_dataframe(data_cfg["test_trigger_csv"], data_cfg["test_arg_csv"], return_mappings=False)
    aug_df = load_full_dataframe(data_cfg["aug_trigger_csv"], data_cfg["aug_arg_csv"], return_mappings=False)

    model_cfg = cfg["model"]
    tokenizer_name = model_cfg["pretrained_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    o_tr = mappings.tr2id["O"]
    o_arg = mappings.arg2id["O"]

    model_baseline = BertMultiTaskLstmCrfDict(
        model_name=tokenizer_name,
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

    wb_run = None
    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("enabled", False):
        wb_run = maybe_init_wandb(
            WandbConfig(
                enabled=True,
                project=wb_cfg.get("project", ""),
                run_name=wb_cfg.get("run_name", ""),
                config=cfg,
            )
        )

    st_cfg = cfg["self_training"]
    output_dir = st_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    label_maps = {
        "tr2id": mappings.tr2id,
        "arg2id": mappings.arg2id,
        "id2tr": mappings.id2tr,
        "id2arg": mappings.id2arg,
    }
    bundle = {"A_TR": dict_bundle.trigger_automaton, "A_ARG": dict_bundle.argument_automaton}

    run_self_training(
        model_baseline=model_baseline,
        num_iterations=int(st_cfg.get("num_iterations", 1)),
        gold_train_df=gold_df,
        aug_df=aug_df,
        test_df=test_df,
        output_base_dir=output_dir,
        tokenizer_name=tokenizer_name,
        label_maps=label_maps,
        dict_bundle=bundle,
        device=device,
        baseline_epochs=int(st_cfg.get("baseline_epochs", 7)),
        iter1_epochs=int(st_cfg.get("iter1_epochs", 7)),
        iter2_epochs=int(st_cfg.get("iter2_epochs", 7)),
        require_agreement=bool(st_cfg.get("require_agreement", True)),
        log_every=int(cfg.get("training", {}).get("log_every_steps", 500)) if "training" in cfg else 500,
        init_checkpoint_path=st_cfg.get("init_checkpoint_path"),
        wandb_run=wb_run,
        optimizer_cfg=cfg.get("optimizer", {}),
        model_cfg=model_cfg,
        eval_cfg=cfg.get("evaluation", {}),
    )

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
