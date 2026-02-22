"""Command-line interface for the ToxHabit augmentation pipeline.

Usage::

    uv run python -m augment.cli \\
        --data-dir  path/to/annotated-docs \\
        --output-dir path/to/augmented-output \\
        --fasttext-model path/to/cc.es.300.bin

All flags correspond directly to fields of :class:`augment.augment.AugConfig`.
"""

from __future__ import annotations

import argparse
import logging

from augment.augment import AugConfig
from augment.pipeline import run_augmentation


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="augment",
        description="Context-aware data augmentation for ToxHabit NER.",
    )
    p.add_argument("--data-dir", required=True, help="Directory with .txt and _triggers/_args JSON files.")
    p.add_argument("--output-dir", required=True, help="Directory to write augmented files.")
    p.add_argument("--fasttext-model", required=True, help="Path to fasttext .bin model (e.g. cc.es.300.bin).")
    p.add_argument("--spacy-model", default="es_core_news_sm", help="spaCy model name (default: es_core_news_sm).")
    p.add_argument("--top-k", type=int, default=3, help="Max replacement candidates per token (default: 3).")
    p.add_argument("--sim-thresh", type=float, default=0.7, help="Min cosine similarity for a candidate (default: 0.7).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    """Parse CLI arguments and run the corpus augmentation pipeline."""
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    cfg = AugConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fasttext_model_path=args.fasttext_model,
        spacy_model=args.spacy_model,
        top_k=args.top_k,
        sim_thresh=args.sim_thresh,
        random_seed=args.seed,
    )

    run_augmentation(cfg)


if __name__ == "__main__":
    main()
