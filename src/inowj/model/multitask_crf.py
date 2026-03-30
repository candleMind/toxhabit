"""Multi-task BERT + BiLSTM + CRF model with dictionary feature fusion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel


class BertMultiTaskLstmCrfDict(nn.Module):
    """BERT encoder with dictionary-feature MLP fusion, BiLSTM contextualization,
    and two independent CRF decoding heads for trigger and argument labeling."""

    def __init__(
        self,
        model_name: str,
        trigger_num_labels: int,
        arg_num_labels: int,
        o_trigger_id: int,
        o_arg_id: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        bidirectional: bool = True,
        dict_feat_dim: int = 10,
        k: int = 32,
    ) -> None:
        """Instantiate model components.

        Args:
            model_name: HuggingFace pretrained model identifier for the BERT encoder.
            trigger_num_labels: Number of BIO labels for the trigger head.
            arg_num_labels: Number of BIO labels for the argument head.
            o_trigger_id: Index of the outside (``O``) class in the trigger label set.
            o_arg_id: Index of the outside (``O``) class in the argument label set.
            lstm_hidden_size: Hidden size per direction of the BiLSTM.
            lstm_num_layers: Number of stacked LSTM layers.
            lstm_dropout: Dropout applied between LSTM layers (inactive if layers == 1).
            bidirectional: Whether to use a bidirectional LSTM.
            dict_feat_dim: Dimensionality of the concatenated dictionary feature vector.
            k: Output size of the dictionary MLP; controls the fused representation width.
        """
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional

        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size

        # Dictionary branch: project sparse one-hot dict features into a dense k-dim space
        # before concatenation with BERT hidden states.
        self.k = k
        # Use min(k, 32) as intermediate MLP dimension to avoid over-parameterizing small k.
        first_dim = 32 if k >= 32 else k
        self.dict_mlp = nn.Sequential(
            nn.Linear(dict_feat_dim, first_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(first_dim, k),
        )
        self.fuse_norm = nn.LayerNorm(self.bert.config.hidden_size + k)

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size + k,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.lstm_dropout = nn.Dropout(lstm_dropout)

        self.trigger_classifier = nn.Linear(lstm_output_size, trigger_num_labels)
        self.trigger_crf = CRF(trigger_num_labels, batch_first=True)
        self.o_trigger_id = o_trigger_id

        self.arg_classifier = nn.Linear(lstm_output_size, arg_num_labels)
        self.arg_crf = CRF(arg_num_labels, batch_first=True)
        self.o_arg_id = o_arg_id

    def init_hidden_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state (h0, c0)."""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.lstm_num_layers * num_directions,
            batch_size,
            self.lstm_hidden_size,
            device=device,
        )
        c0 = torch.zeros(
            self.lstm_num_layers * num_directions,
            batch_size,
            self.lstm_hidden_size,
            device=device,
        )
        return h0, c0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dict_feats: torch.Tensor | None = None,
        trigger_labels: torch.Tensor | None = None,
        arg_labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Train mode: provide `trigger_labels` and/or `arg_labels` -> returns losses.
        Inference mode: omit labels -> returns decoded predictions.

        Args:
            input_ids: [B, T]
            attention_mask: [B, T]
            dict_feats: [B, T, dict_feat_dim]
            trigger_labels: [B, T] with -100 for ignored positions.
            arg_labels: [B, T] with -100 for ignored positions.

        Returns:
            Dict containing losses or predictions.
        """
        batch_size = int(input_ids.size(0))
        device = input_ids.device

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden = self.dropout(outputs.last_hidden_state)  # [B, T, H]

        if dict_feats is not None:
            dfeat = self.dict_mlp(dict_feats.float())  # [B, T, k]
        else:
            t_len = int(bert_hidden.size(1))
            dfeat = torch.zeros(batch_size, t_len, self.k, device=device)

        fused = torch.cat([bert_hidden, dfeat], dim=-1)  # [B, T, H+k]
        fused = self.fuse_norm(fused)  # LayerNorm stabilizes the concatenated representation.

        h0, c0 = self.init_hidden_state(batch_size, device)
        lengths = attention_mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        try:
            # Use packed sequences to avoid computing over padding positions.
            packed = nn.utils.rnn.pack_padded_sequence(
                fused, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed, (h0, c0))
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        except Exception:
            # Fallback for edge cases (e.g. single-token sequences).
            lstm_out, _ = self.lstm(fused, (h0, c0))

        lstm_hidden = self.lstm_dropout(lstm_out)

        trigger_emissions = self.trigger_classifier(lstm_hidden)
        arg_emissions = self.arg_classifier(lstm_hidden)

        if trigger_labels is not None or arg_labels is not None:
            losses: dict[str, torch.Tensor] = {}
            valid_mask = attention_mask.bool()

            if trigger_labels is not None:
                trig_valid = valid_mask & (trigger_labels != -100)
                trig_tags = trigger_labels.clone()
                # Replace the ignore index (-100) with "O" so CRF sees a valid label.
                trig_tags[trig_tags == -100] = self.o_trigger_id
                valid_seq = trig_valid.sum(dim=1) > 0
                if valid_seq.any():
                    # CRF returns log-likelihood; negate for minimization.
                    trig_loss = -self.trigger_crf(
                        trigger_emissions[valid_seq],
                        trig_tags[valid_seq],
                        mask=trig_valid[valid_seq],
                        reduction="mean",
                    )
                else:
                    trig_loss = torch.tensor(0.0, device=device, requires_grad=True)
                losses["trigger_loss"] = trig_loss

            if arg_labels is not None:
                arg_valid = valid_mask & (arg_labels != -100)
                arg_tags = arg_labels.clone()
                # Replace the ignore index (-100) with "O" so CRF sees a valid label.
                arg_tags[arg_tags == -100] = self.o_arg_id
                valid_seq = arg_valid.sum(dim=1) > 0
                if valid_seq.any():
                    # CRF returns log-likelihood; negate for minimization.
                    arg_loss = -self.arg_crf(
                        arg_emissions[valid_seq],
                        arg_tags[valid_seq],
                        mask=arg_valid[valid_seq],
                        reduction="mean",
                    )
                else:
                    arg_loss = torch.tensor(0.0, device=device, requires_grad=True)
                losses["arg_loss"] = arg_loss

            # Joint loss: sum of the two CRF NLL terms.
            losses["total_loss"] = sum(losses.values())
            return losses

        trigger_preds = self.trigger_crf.decode(trigger_emissions, mask=attention_mask.bool())
        arg_preds = self.arg_crf.decode(arg_emissions, mask=attention_mask.bool())
        return {
            "trigger_preds": trigger_preds,
            "arg_preds": arg_preds,
            "trigger_emissions": trigger_emissions,
            "arg_emissions": arg_emissions,
        }
