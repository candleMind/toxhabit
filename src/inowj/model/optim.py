"""Optimizer and scheduler builders."""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict


def build_optimizer_and_scheduler(
    model: BertMultiTaskLstmCrfDict,
    train_loader: DataLoader,
    epochs: int,
    bert_lr: float = 2e-5,
    lstm_lr: float = 5e-4,
    clf_lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create AdamW parameter groups with per-component learning rates and a
    linear warmup + linear decay schedule.

    Args:
        model: The multi-task model instance.
        train_loader: Training DataLoader (used to compute total training steps).
        epochs: Number of training epochs.
        bert_lr: Learning rate for BERT encoder parameters.
        lstm_lr: Learning rate for LSTM, dict MLP, and CRF parameters.
        clf_lr: Learning rate for linear classifier heads.
        weight_decay: AdamW weight decay coefficient.
        warmup_ratio: Fraction of total steps used for linear LR warmup.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    bert_params = list(model.bert.parameters())
    lstm_params = list(model.lstm.parameters())
    classifier_params = list(model.trigger_classifier.parameters()) + list(model.arg_classifier.parameters())
    crf_params = list(model.trigger_crf.parameters()) + list(model.arg_crf.parameters())
    dict_mlp_params = list(model.dict_mlp.parameters())
    fuse_norm_params = list(model.fuse_norm.parameters())

    def param_group(params: list, lr: float, wd: float) -> dict:
        return {"params": [p for p in params if p.requires_grad], "lr": lr, "weight_decay": wd}

    optimizer = AdamW(
        [
            param_group(bert_params, lr=bert_lr, wd=weight_decay),
            param_group(lstm_params, lr=lstm_lr, wd=weight_decay),
            param_group(dict_mlp_params, lr=lstm_lr, wd=weight_decay),
            param_group(fuse_norm_params, lr=lstm_lr, wd=0.0),  # LayerNorm: no weight decay
            param_group(classifier_params, lr=clf_lr, wd=weight_decay),
            param_group(crf_params, lr=lstm_lr, wd=weight_decay),
        ]
    )

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler
