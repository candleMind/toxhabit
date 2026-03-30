"""
Data loading utilities for ToxHabit corpus preprocessing.

This module loads clinical case documents with annotated triggers and arguments
from JSON format into structured Python dictionaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_samples(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load clinical case samples from directory containing text and annotation files.
    
    Expected file structure per document:
    - {doc_id}.txt: raw clinical text
    - {doc_id}_triggers.json: trigger annotations
    - {doc_id}_args.json: argument annotations
    
    Args:
        data_dir: Path to directory containing document files
        
    Returns:
        List of dictionaries, each containing:
            - doc_id: document identifier
            - text: raw clinical text
            - triggers: list of trigger annotations
            - arguments: list of argument annotations
    """
    data_dir = Path(data_dir)
    txt_files = list(data_dir.glob("*.txt"))
    samples = []
    
    for txt_file in txt_files:
        doc_id = txt_file.stem
        trigger_path = data_dir / f"{doc_id}_triggers.json"
        arg_path = data_dir / f"{doc_id}_args.json"
        
        if not (trigger_path.exists() and arg_path.exists()):
            continue
        
        with open(txt_file, encoding='utf-8') as f:
            text = f.read()
        with open(trigger_path, encoding='utf-8') as f:
            triggers = json.load(f)
        with open(arg_path, encoding='utf-8') as f:
            args = json.load(f)
        
        samples.append({
            "doc_id": doc_id,
            "text": text,
            "triggers": triggers,
            "arguments": args
        })
    
    return samples
