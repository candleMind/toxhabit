"""Data loading and label mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from inowj.dataprep.schema import validate_columns
from inowj.utils.io import read_csv


@dataclass(frozen=True)
class LabelMappings:
    """Label mappings for triggers and arguments."""

    tr2id: dict[str, int]
    arg2id: dict[str, int]
    id2tr: dict[int, str]
    id2arg: dict[int, str]


def load_full_dataframe(
    trigger_path: str,
    arg_path: str,
    return_mappings: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, LabelMappings]:
    """Load and merge trigger + argument CSVs into a single DataFrame.

    Args:
        trigger_path: Path to trigger CSV.
        arg_path: Path to argument CSV.
        return_mappings: If True, also returns LabelMappings.

    Returns:
        DataFrame or (DataFrame, LabelMappings).
    """
    df_tr = read_csv(trigger_path, keep_default_na=False)
    validate_columns(df_tr.columns)

    if "label" in df_tr.columns:
        df_tr = df_tr.rename(columns={"label": "trigger_label"})

    unique_tr = sorted(lbl for lbl in df_tr["trigger_label"].unique() if lbl)
    tr2id = {lbl: i for i, lbl in enumerate(unique_tr)}
    id2tr = {i: lbl for lbl, i in tr2id.items()}

    df_tr["trigger_label_id"] = df_tr.apply(
        lambda r: -100 if bool(r["is_subword"]) else tr2id[r["trigger_label"]],
        axis=1,
    )
    df_tr["token_id"] = df_tr.groupby(["doc_id", "sentence_id"]).cumcount()

    df_tr = df_tr[
        [
            "doc_id",
            "sentence_id",
            "token_id",
            "token",
            "is_subword",
            "trigger_label",
            "trigger_label_id",
            "start",
            "end",
        ]
    ]

    df_arg = read_csv(arg_path, keep_default_na=False)
    validate_columns(df_arg.columns)

    if "label" in df_arg.columns:
        df_arg = df_arg.rename(columns={"label": "arg_label"})

    unique_arg = sorted(lbl for lbl in df_arg["arg_label"].unique() if lbl)
    arg2id = {lbl: i for i, lbl in enumerate(unique_arg)}
    id2arg = {i: lbl for lbl, i in arg2id.items()}

    df_arg["arg_label_id"] = df_arg.apply(
        lambda r: -100 if bool(r["is_subword"]) else arg2id[r["arg_label"]],
        axis=1,
    )
    df_arg["token_id"] = df_arg.groupby(["doc_id", "sentence_id"]).cumcount()

    df_arg = df_arg[
        [
            "doc_id",
            "sentence_id",
            "token_id",
            "token",
            "is_subword",
            "arg_label",
            "arg_label_id",
            "start",
            "end",
        ]
    ]

    df = pd.merge(
        df_tr,
        df_arg,
        on=["doc_id", "sentence_id", "token_id", "token", "is_subword", "start", "end"],
        how="outer",
    )

    df["trigger_label_id"] = df["trigger_label_id"].fillna(-100).astype(int)
    df["arg_label_id"] = df["arg_label_id"].fillna(-100).astype(int)

    df["trigger_label"] = df["trigger_label"].fillna("O")
    df["arg_label"] = df["arg_label"].fillna("O")

    if not return_mappings:
        return df

    mappings = LabelMappings(
        tr2id=tr2id,
        arg2id=arg2id,
        id2tr=id2tr,
        id2arg=id2arg,
    )
    return df, mappings
