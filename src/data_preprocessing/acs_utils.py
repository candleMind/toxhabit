"""
Adaptive Context Selection (ACS) utilities for clinical text segmentation.

This module implements ACS, ensuring syntactically valid clause segmentation.
"""

from typing import List, Tuple
import spacy


def spacy_split_sentences(text: str, nlp: spacy.Language) -> List[Tuple[int, int]]:
    """
    Split text into sentences using spaCy's sentence boundary detection.
    
    Args:
        text: Input text to segment
        nlp: Pre-loaded spaCy language model
        
    Returns:
        List of (start_char, end_char) tuples for each sentence
    """
    doc = nlp(text)
    return [(sent.start_char, sent.end_char) for sent in doc.sents]


def is_valid_clause(text_segment: str, nlp: spacy.Language) -> bool:
    """
    Verify if text segment forms a valid independent clause.
    
    A valid clause must contain:
    - A ROOT verb (main predicate)
    - A subject (nominal or clausal)
    
    Args:
        text_segment: Text to validate
        nlp: Pre-loaded spaCy language model
        
    Returns:
        True if segment is a valid independent clause
    """
    doc = nlp(text_segment.strip())
    has_root_verb = any(
        token.dep_ == "ROOT" and token.pos_ == "VERB" 
        for token in doc
    )
    has_subject = any(
        token.dep_.startswith("nsubj") or token.dep_.startswith("csubj") 
        for token in doc
    )
    return has_root_verb and has_subject


def split_by_semicolon(text_segment: str) -> List[Tuple[int, int, str]]:
    """
    Split text segment at semicolon and comma boundaries.
    
    Args:
        text_segment: Text to split
        
    Returns:
        List of (relative_start, relative_end, text) tuples for each subsegment
    """
    semicolons = [i for i, char in enumerate(text_segment) if char in {',', ';'}]
    
    if not semicolons:
        return [(0, len(text_segment), text_segment)]
    
    splits = []
    prev = 0
    
    for sc_pos in semicolons:
        if prev < sc_pos:
            segment = text_segment[prev:sc_pos].strip()
            if segment:
                splits.append((prev, sc_pos, segment))
        prev = sc_pos + 1
    
    # Handle remaining text after last delimiter
    if prev < len(text_segment):
        segment = text_segment[prev:].strip()
        if segment:
            splits.append((prev, len(text_segment), segment))
    
    return splits if splits else [(0, len(text_segment), text_segment)]


def check_all_valid_clauses(
    subsegments: List[Tuple[int, int, str]], 
    nlp: spacy.Language
) -> bool:
    """
    ACS Rule 3: Verify all subsegments form valid independent clauses.
    
    This rule ensures that splitting produces syntactically complete segments,
    each with a main verb and subject.
    
    Args:
        subsegments: List of (rel_start, rel_end, text) tuples
        nlp: Pre-loaded spaCy language model
        
    Returns:
        True if all subsegments are valid clauses (split allowed)
    """
    if len(subsegments) == 1:
        return True
    
    for _, _, text in subsegments:
        if not is_valid_clause(text, nlp):
            return False
    
    return True


def process_document_sentences(
    text: str,
    nlp: spacy.Language
) -> List[Tuple[int, int, str, bool]]:
    """
    Process document with ACS (clause validity check).
    
    Pipeline:
    1. Initial sentence splitting using spaCy
    2. Semicolon-based subsegmentation
    3. Verify all segments are valid independent clauses
    
    Args:
        text: Full document text
        nlp: Pre-loaded spaCy language model
        
    Returns:
        List of (start, end, text, is_segmented) tuples where:
            - start/end are absolute character positions
            - text is the sentence/segment content
            - is_segmented indicates if semicolon split was applied
    """
    spacy_sentences = spacy_split_sentences(text, nlp)
    final_sentences = []
    
    for sent_start, sent_end in spacy_sentences:
        sentence_text = text[sent_start:sent_end]
        subsegments = split_by_semicolon(sentence_text)
        
        if len(subsegments) == 1:
            final_sentences.append((sent_start, sent_end, sentence_text.strip(), False))
            continue
        
        # Apply ACS Rule: check if all subsegments are valid clauses
        if check_all_valid_clauses(subsegments, nlp):
            # All segments are valid clauses - apply split
            for rel_s, rel_e, subtext in subsegments:
                abs_start = sent_start + rel_s
                abs_end = sent_start + rel_e
                final_sentences.append((abs_start, abs_end, subtext, True))
        else:
            # Contains invalid clause(s) - keep original sentence
            final_sentences.append((sent_start, sent_end, sentence_text.strip(), False))
    
    return final_sentences
