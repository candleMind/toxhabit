"""
Tokenization and CSV export utilities for sequence labeling datasets.

This module handles subword tokenization with label alignment for trigger
and argument detection tasks in clinical NLP.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import spacy
from transformers import AutoTokenizer

from acs_utils import process_document_sentences

# Configure logger
logger = logging.getLogger(__name__)

# Valid argument types for clinical substance use events
VALID_ARG_TYPES: Set[str] = {
    "Type", "Method", "Amount", "Frequency", "Duration", "History"
}


def tokenize_and_align_labels(
    text: str,
    trigger_spans: List[Dict[str, Any]],
    tokenizer: AutoTokenizer
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Tokenize text and align trigger labels at subword level.
    
    Uses BIO tagging scheme where:
    - B-{type}: Beginning of trigger
    - I-{type}: Inside trigger
    - O: Outside any trigger
    
    Args:
        text: Input text to tokenize
        trigger_spans: List of trigger annotations with start/end spans and types
        tokenizer: Pre-loaded HuggingFace tokenizer
        
    Returns:
        Tuple of (tokens, aligned_labels, token_spans) where:
            - tokens: subword tokens
            - aligned_labels: BIO labels for each token
            - token_spans: (start, end) character positions for each token
    """
    # Initialize character-level labels
    labels = ['O'] * len(text)
    
    for trig in trigger_spans:
        start = trig.get('trigger_start_span')
        end = trig.get('trigger_end_span')
        trig_type = trig.get('trigger_type')
        
        if start is None or end is None or trig_type is None:
            continue
        
        if 0 <= start < end <= len(text):
            labels[start] = f'B-{trig_type}'
            for i in range(start + 1, end):
                labels[i] = f'I-{trig_type}'
    
    # Tokenize without special tokens
    encoding = tokenizer(
        text, 
        return_offsets_mapping=True, 
        truncation=False, 
        add_special_tokens=False
    )
    token_spans = encoding['offset_mapping']
    token_ids = encoding['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Align labels to subword tokens using majority voting from character spans
    aligned_labels = []
    for (start, end) in token_spans:
        if start == end:  # Empty token
            aligned_labels.append('O')
        else:
            span_labels = labels[start:end]
            b_labels = set(label for label in span_labels if label.startswith("B-"))
            i_labels = set(label for label in span_labels if label.startswith("I-"))
            
            # Priority: B- tags > I- tags > O
            if b_labels:
                aligned_labels.append(list(b_labels)[0])
            elif i_labels:
                aligned_labels.append(list(i_labels)[0])
            else:
                aligned_labels.append('O')
    
    return tokens, aligned_labels, token_spans


def tokenize_and_align_argument_labels(
    text: str,
    argument_spans: List[Dict[str, Any]],
    tokenizer: AutoTokenizer
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Tokenize text and align argument labels at subword level.
    
    Uses BIO tagging for argument role labeling (Type, Method, Amount, etc.).
    
    Args:
        text: Input text to tokenize
        argument_spans: List of argument annotations with start/end spans and types
        tokenizer: Pre-loaded HuggingFace tokenizer
        
    Returns:
        Tuple of (tokens, aligned_labels, token_spans)
    """
    labels = ['O'] * len(text)
    
    for arg in argument_spans:
        start = arg.get('arg_start_span')
        end = arg.get('arg_end_span')
        atype = arg.get('arg_type')
        
        if atype not in VALID_ARG_TYPES:
            continue
        
        if start is None or end is None or atype is None:
            continue
        
        if 0 <= start < end <= len(text):
            labels[start] = f'B-{atype}'
            for i in range(start + 1, end):
                labels[i] = f'I-{atype}'
    
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False
    )
    token_spans = encoding['offset_mapping']
    token_ids = encoding['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    aligned = []
    for (s, e) in token_spans:
        if s == e:
            aligned.append('O')
        else:
            span_lbls = labels[s:e]
            b_lbls = {label for label in span_lbls if label.startswith("B-")}
            i_lbls = {label for label in span_lbls if label.startswith("I-")}
            
            if b_lbls:
                aligned.append(b_lbls.pop())
            elif i_lbls:
                aligned.append(i_lbls.pop())
            else:
                aligned.append('O')
    
    return tokens, aligned, token_spans


def export_with_config(
    samples: List[Dict[str, Any]],
    output_dir: str,
    nlp: spacy.Language,
    tokenizer: AutoTokenizer,
    is_train: bool = True
) -> None:
    """
    Export processed samples to CSV files for model training/evaluation.
    
    Applies ACS (clause validity check) with semicolon-based subsegmentation.
    
    Generates three CSV files:
    - {prefix}_triggers_tagging.csv: trigger detection labels
    - {prefix}_args_tagging.csv: argument role labeling
    - {prefix}_sentences.csv: sentence segmentation metadata
    
    Args:
        samples: List of document samples with text and annotations
        output_dir: Directory to write CSV files
        nlp: Pre-loaded spaCy language model
        tokenizer: Pre-loaded HuggingFace tokenizer
        is_train: If True, prefix files with 'train', else 'test'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trigger_rows = []
    arg_rows = []
    sentence_rows = []
    
    for sample in samples:
        doc_id = sample.get('doc_id')
        text = sample.get('text', '')
        triggers = sample.get('triggers', [])
        arguments = sample.get('arguments', [])
        
        # Parse JSON strings if needed
        if isinstance(triggers, str):
            try:
                triggers = json.loads(triggers)
            except json.JSONDecodeError:
                triggers = []
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = []
        
        # Apply ACS for sentence segmentation
        sentence_spans = process_document_sentences(text, nlp)
        
        # Build character position to sentence mapping
        local_sentence_id = 0
        char_to_sentence = {}
        
        for s_start, s_end, s_text, is_segmented in sentence_spans:
            local_sentence_id += 1
            
            sentence_rows.append({
                "sentence_id": local_sentence_id,
                "doc_id": doc_id,
                "sentence": s_text,
                "segment_check": is_segmented
            })
            
            for char_pos in range(s_start, s_end):
                char_to_sentence[char_pos] = (local_sentence_id, is_segmented)
        
        # Tokenize and tag triggers
        try:
            tokens, labels, spans = tokenize_and_align_labels(
                text, triggers, tokenizer
            )
            
            for tok, label, (start, end) in zip(tokens, labels, spans):
                is_subword = tok.startswith("##")
                
                # Find sentence assignment for this token
                sid = -1
                segment_check = False
                for char_pos in range(start, end):
                    if char_pos in char_to_sentence:
                        sid, segment_check = char_to_sentence[char_pos]
                        break
                
                trigger_rows.append({
                    "doc_id": doc_id,
                    "token": tok,
                    "start": start,
                    "end": end,
                    "trigger_label": label,
                    "is_subword": is_subword,
                    "sentence_id": sid,
                    "segment_check": segment_check
                })
        except Exception as e:
            logger.error(f"Trigger tagging failed for doc_id={doc_id}: {e}")
        
        # Tokenize and tag arguments
        try:
            tokens, labels, spans = tokenize_and_align_argument_labels(
                text, arguments, tokenizer
            )
            
            for tok, label, (start, end) in zip(tokens, labels, spans):
                is_subword = tok.startswith("##")
                
                sid = -1
                segment_check = False
                for char_pos in range(start, end):
                    if char_pos in char_to_sentence:
                        sid, segment_check = char_to_sentence[char_pos]
                        break
                
                arg_rows.append({
                    "doc_id": doc_id,
                    "token": tok,
                    "start": start,
                    "end": end,
                    "arg_label": label,
                    "is_subword": is_subword,
                    "sentence_id": sid,
                    "segment_check": segment_check
                })
        except Exception as e:
            logger.error(f"Argument tagging failed for doc_id={doc_id}: {e}")
    
    # Write CSV files
    prefix = "train" if is_train else "test"
    
    with open(output_dir / f"{prefix}_triggers_tagging.csv", 'w', 
              encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "doc_id", "token", "start", "end", "trigger_label", 
            "is_subword", "sentence_id", "segment_check"
        ])
        writer.writeheader()
        writer.writerows(trigger_rows)
    
    with open(output_dir / f"{prefix}_args_tagging.csv", 'w', 
              encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "doc_id", "token", "start", "end", "arg_label", 
            "is_subword", "sentence_id", "segment_check"
        ])
        writer.writeheader()
        writer.writerows(arg_rows)
    
    with open(output_dir / f"{prefix}_sentences.csv", 'w', 
              encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentence_id", "doc_id", "sentence", "segment_check"
        ])
        writer.writeheader()
        writer.writerows(sentence_rows)
    
    logger.info(f"Exported {len(trigger_rows)} trigger annotations, "
                f"{len(arg_rows)} argument annotations, "
                f"{len(sentence_rows)} sentences to {output_dir}")
