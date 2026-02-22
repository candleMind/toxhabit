"""Training loop utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inowj.model.multitask_crf import BertMultiTaskLstmCrfDict
from inowj.utils.logging import maybe_wandb_log


def train_model(
    model: BertMultiTaskLstmCrfDict,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    save_path: str,
    log_every: int = 500,
    max_grad_norm: float = 1.0,
    wandb_run: Any | None = None,
    extra_state: dict[str, Any] | None = None,
) -> tuple[dict[str, list[float]], float]:
    """Train the model and save the best checkpoint by average epoch loss.

    Args:
        model: Model instance.
        train_loader: Training dataloader.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        device: Device.
        epochs: Number of epochs.
        save_path: Path to save the best checkpoint.
        log_every: Log step metrics every N steps.
        max_grad_norm: Gradient clipping norm.
        wandb_run: Optional W&B run.
        extra_state: Extra dict to store in the checkpoint (e.g., label mappings).

    Returns:
        Tuple of (training_history, best_loss) where training_history is a dict
        mapping ``epoch_losses``, ``trigger_losses``, and ``arg_losses`` to
        their per-epoch float values, and best_loss is the lowest average
        epoch loss observed during training.
    """
    best_loss = float("inf")
    training_history: dict[str, list[float]] = {
        "epoch_losses": [],
        "trigger_losses": [],
        "arg_losses": [],
    }
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_trig_sum = 0.0
        epoch_arg_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            trigger_labels = batch["trigger_labels"].to(device)
            arg_labels = batch["arg_labels"].to(device)
            dict_feats = batch["dict_feats"].to(device)

            optimizer.zero_grad()

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_feats=dict_feats,
                trigger_labels=trigger_labels,
                arg_labels=arg_labels,
            )

            loss = out["total_loss"]
            trig_loss = out.get("trigger_loss", torch.tensor(0.0, device=device))
            arg_loss = out.get("arg_loss", torch.tensor(0.0, device=device))

            if not torch.isfinite(loss):
                # Skip batches with NaN/Inf loss to prevent gradient corruption.
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += float(loss.item())
            epoch_trig_sum += float(trig_loss.item())
            epoch_arg_sum += float(arg_loss.item())
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{epoch_loss_sum/step:.4f}",
                    "trig": f"{epoch_trig_sum/step:.4f}",
                    "arg": f"{epoch_arg_sum/step:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if log_every > 0 and global_step % log_every == 0:
                maybe_wandb_log(
                    wandb_run,
                    {
                        "train/step_loss": float(loss.item()),
                        "train/step_trigger_loss": float(trig_loss.item()),
                        "train/step_arg_loss": float(arg_loss.item()),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "epoch": epoch,
                        "step": global_step,
                    },
                )

        n_batches = max(1, len(train_loader))
        epoch_avg = epoch_loss_sum / n_batches
        epoch_trig_avg = epoch_trig_sum / n_batches
        epoch_arg_avg = epoch_arg_sum / n_batches

        training_history["epoch_losses"].append(epoch_avg)
        training_history["trigger_losses"].append(epoch_trig_avg)
        training_history["arg_losses"].append(epoch_arg_avg)

        maybe_wandb_log(
            wandb_run,
            {
                "train/epoch_loss": epoch_avg,
                "train/epoch_trigger_loss": epoch_trig_avg,
                "train/epoch_arg_loss": epoch_arg_avg,
                "epoch": epoch,
            },
        )

        if epoch_avg < best_loss:
            best_loss = epoch_avg
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_loss,
                "training_history": training_history,
            }
            if extra_state:
                ckpt.update(extra_state)

            torch.save(ckpt, save_path)

    return training_history, best_loss
