"""Human-readable reporting utilities."""

from __future__ import annotations

from typing import Any


def log_detailed_metrics(metrics: dict[str, Any], stage_name: str, iteration: int | None = None) -> None:
    """Print per-type + micro/macro metrics in a compact table."""
    prefix = f"[ITERATION {iteration}] " if iteration is not None else ""

    print("\n" + "=" * 80)
    print(f"{prefix}{stage_name} - TRIGGER RESULTS")
    print("=" * 80)
    print(f"{'Type':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)

    for type_name, stats in metrics.get("trigger_per_type", {}).items():
        print(
            f"{type_name:<12} {stats['p']:>10.4f} {stats['r']:>10.4f} {stats['f1']:>10.4f} "
            f"{stats['tp']:>6d} {stats['fp']:>6d} {stats['fn']:>6d}"
        )

    print("-" * 80)
    print(
        f"{'MICRO':<12} {metrics.get('trigger_precision', 0):>10.4f} "
        f"{metrics.get('trigger_recall', 0):>10.4f} "
        f"{metrics.get('trigger_f1', 0):>10.4f} "
        f"{metrics.get('trigger_tp', 0):>6} "
        f"{metrics.get('trigger_fp', 0):>6} "
        f"{metrics.get('trigger_fn', 0):>6}"
    )
    print(
        f"{'MACRO':<12} {metrics.get('trigger_macro_p', 0):>10.4f} "
        f"{metrics.get('trigger_macro_r', 0):>10.4f} "
        f"{metrics.get('trigger_macro_f1', 0):>10.4f}"
    )

    print("\n" + "=" * 80)
    print(f"{prefix}{stage_name} - ARGUMENT RESULTS")
    print("=" * 80)
    print(f"{'Type':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)

    for type_name, stats in metrics.get("arg_per_type", {}).items():
        print(
            f"{type_name:<12} {stats['p']:>10.4f} {stats['r']:>10.4f} {stats['f1']:>10.4f} "
            f"{stats['tp']:>6d} {stats['fp']:>6d} {stats['fn']:>6d}"
        )

    print("-" * 80)
    print(
        f"{'MICRO':<12} {metrics.get('arg_precision', 0):>10.4f} "
        f"{metrics.get('arg_recall', 0):>10.4f} "
        f"{metrics.get('arg_f1', 0):>10.4f} "
        f"{metrics.get('arg_tp', 0):>6} "
        f"{metrics.get('arg_fp', 0):>6} "
        f"{metrics.get('arg_fn', 0):>6}"
    )
    print(
        f"{'MACRO':<12} {metrics.get('arg_macro_p', 0):>10.4f} "
        f"{metrics.get('arg_macro_r', 0):>10.4f} "
        f"{metrics.get('arg_macro_f1', 0):>10.4f}"
    )
    print("=" * 80)
