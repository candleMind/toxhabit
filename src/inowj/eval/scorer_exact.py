"""Exact-match scorers matching the notebook evaluation logic."""

from __future__ import annotations

import json
import os
from typing import Any


def load_triggers(
    file_path: str,
) -> tuple[set[tuple[str, int, int]], dict[tuple[str, int, int], dict[str, Any]]]:
    """Load a trigger JSON file into a span set and a detail mapping.

    Args:
        file_path: Path to a JSON file with trigger annotations.

    Returns:
        triggers: Set of (type, start, end) span tuples.
        details: Mapping from span tuple to annotation fields.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    trigger_list = list(data.values()) if isinstance(data, dict) else data
    triggers: set[tuple[str, int, int]] = set()
    details: dict[tuple[str, int, int], dict[str, Any]] = {}

    for trig in trigger_list:
        typ = trig["trigger_type"]
        start = int(trig["trigger_start_span"])
        end = int(trig["trigger_end_span"])
        key = (typ, start, end)
        triggers.add(key)
        details[key] = {
            "trigger_text": trig.get("trigger_text", ""),
            "start": start,
            "end": end,
            "trigger_type": typ,
        }
    return triggers, details


def load_arguments(
    file_path: str,
) -> tuple[set[tuple[str, int, int]], dict[tuple[str, int, int], dict[str, Any]]]:
    """Load an argument JSON file into a span set and a detail mapping.

    Args:
        file_path: Path to a JSON file with argument annotations.

    Returns:
        arguments: Set of (type, start, end) span tuples.
        details: Mapping from span tuple to annotation fields.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    arg_list = list(data.values()) if isinstance(data, dict) else data
    arguments: set[tuple[str, int, int]] = set()
    details: dict[tuple[str, int, int], dict[str, Any]] = {}

    for arg in arg_list:
        if "argument_type" in arg:
            typ = arg.get("argument_type")
            raw = arg.get("argument_text")
            start = arg.get("argument_start_span")
            end = arg.get("argument_end_span")
        else:
            typ = arg.get("arg_type")
            raw = arg.get("arg_text")
            start = arg.get("arg_start_span")
            end = arg.get("arg_end_span")

        if None in (typ, raw, start, end):
            continue

        try:
            start_i = int(start)
            end_i = int(end)
        except (ValueError, TypeError):
            continue

        key = (str(typ), start_i, end_i)
        arguments.add(key)
        details[key] = {"arg_text": raw, "start": start_i, "end": end_i, "arg_type": str(typ)}

    return arguments, details


def evaluate_trigger_f1(predict_dir: str, gold_dir: str) -> dict[str, Any]:
    """Compute exact-match micro and macro F1 for trigger spans.

    Matches predicted and gold trigger JSON files by filename, then evaluates
    span-level precision, recall, and F1 for each trigger type and overall.

    Args:
        predict_dir: Directory containing predicted ``*_triggers.json`` files.
        gold_dir: Directory containing gold-standard ``*_triggers.json`` files.

    Returns:
        Dictionary with keys: ``micro_p``, ``micro_r``, ``micro_f1``,
        ``macro_p``, ``macro_r``, ``macro_f1``, ``tp``, ``fp``, ``fn``,
        and ``per_type`` (per-category P/R/F1 breakdown).
    """
    all_files = sorted(
        [
            f
            for f in os.listdir(predict_dir)
            if f.endswith("_triggers.json") and os.path.exists(os.path.join(gold_dir, f))
        ]
    )

    total_tp = total_fp = total_fn = 0
    type_stats: dict[str, dict[str, int]] = {}

    for filename in all_files:
        pred_path = os.path.join(predict_dir, filename)
        gold_path = os.path.join(gold_dir, filename)

        pred_set, pred_info = load_triggers(pred_path)
        gold_set, gold_info = load_triggers(gold_path)

        tp = pred_set & gold_set
        fp = pred_set - gold_set
        fn_set = gold_set - pred_set

        for key in tp:
            t = key[0]
            type_stats.setdefault(t, {"TP": 0, "FP": 0, "FN": 0})["TP"] += 1
        for key in fp:
            t = pred_info[key]["trigger_type"]
            type_stats.setdefault(t, {"TP": 0, "FP": 0, "FN": 0})["FP"] += 1
        for key in fn_set:
            t = gold_info[key]["trigger_type"]
            type_stats.setdefault(t, {"TP": 0, "FP": 0, "FN": 0})["FN"] += 1

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn_set)

    micro_p = round(total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0, 4)
    micro_r = round(total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0, 4)
    micro_f = round((2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0, 4)

    per_type: dict[str, dict[str, Any]] = {}
    for t, s in type_stats.items():
        tp = s["TP"]
        fp = s["FP"]
        fn = s["FN"]
        p = round(tp / (tp + fp) if (tp + fp) else 0.0, 4)
        r = round(tp / (tp + fn) if (tp + fn) else 0.0, 4)
        f1 = round((2 * p * r / (p + r)) if (p + r) else 0.0, 4)
        per_type[t] = {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    if per_type:
        macro_p = round(sum(per_type[t]["p"] for t in per_type) / len(per_type), 4)
        macro_r = round(sum(per_type[t]["r"] for t in per_type) / len(per_type), 4)
        macro_f1 = round(sum(per_type[t]["f1"] for t in per_type) / len(per_type), 4)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_type": per_type,
    }


def evaluate_argument_f1(predict_dir: str, gold_dir: str) -> dict[str, Any]:
    """Compute exact-match micro and macro F1 for argument spans.

    Matches predicted and gold argument JSON files by filename, then evaluates
    span-level precision, recall, and F1 for each argument type and overall.

    Args:
        predict_dir: Directory containing predicted ``*_args.json`` files.
        gold_dir: Directory containing gold-standard ``*_args.json`` files.

    Returns:
        Dictionary with keys: ``micro_p``, ``micro_r``, ``micro_f1``,
        ``macro_p``, ``macro_r``, ``macro_f1``, ``tp``, ``fp``, ``fn``,
        and ``per_type`` (per-category P/R/F1 breakdown).
    """
    all_files = sorted(
        [
            f
            for f in os.listdir(predict_dir)
            if f.endswith("_args.json") and os.path.exists(os.path.join(gold_dir, f))
        ]
    )

    total_tp = total_fp = total_fn = 0
    type_stats: dict[str, dict[str, int]] = {}

    for filename in all_files:
        pred_path = os.path.join(predict_dir, filename)
        gold_path = os.path.join(gold_dir, filename)

        pred_set, pred_info = load_arguments(pred_path)
        gold_set, gold_info = load_arguments(gold_path)

        tp = pred_set & gold_set
        fp = pred_set - gold_set
        fn_set = gold_set - pred_set

        for k in tp:
            t = k[0]
            type_stats.setdefault(t, {"TP": 0, "FP": 0, "FN": 0})["TP"] += 1
        for k in fp:
            t = pred_info[k]["arg_type"]
            type_stats.setdefault(t, {"TP": 0, "FP": 0, "FN": 0})["FP"] += 1
        for k in fn_set:
            t = gold_info[k]["arg_type"]
            type_stats.setdefault(t, {"TP": 0, "FP": 0, "FN": 0})["FN"] += 1

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn_set)

    micro_p = round(total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0, 4)
    micro_r = round(total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0, 4)
    micro_f = round((2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0, 4)

    per_type: dict[str, dict[str, Any]] = {}
    for t, s in type_stats.items():
        tp = s["TP"]
        fp = s["FP"]
        fn = s["FN"]
        p = round(tp / (tp + fp) if (tp + fp) else 0.0, 4)
        r = round(tp / (tp + fn) if (tp + fn) else 0.0, 4)
        f1 = round((2 * p * r / (p + r)) if (p + r) else 0.0, 4)
        per_type[t] = {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    if per_type:
        macro_p = round(sum(per_type[t]["p"] for t in per_type) / len(per_type), 4)
        macro_r = round(sum(per_type[t]["r"] for t in per_type) / len(per_type), 4)
        macro_f1 = round(sum(per_type[t]["f1"] for t in per_type) / len(per_type), 4)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_type": per_type,
    }
