"""
Convert BRAT annotation files (.ann) to JSON format for ToxHabit corpus.

This module processes trigger (ToxNER) and argument (ToxUse) annotations
from BRAT format to structured JSON files compatible with dataset loaders.
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_ann_file(
    ann_path: str,
    doc_id: str,
    entry_type: str
) -> List[Dict[str, Any]]:
    """
    Parse BRAT annotation file and extract trigger or argument annotations.
    
    BRAT format: Each line contains tab-separated fields:
        - Column 0: Annotation ID (e.g., T1, R2)
        - Column 1: Label with character spans (e.g., "Alcohol 45 52")
        - Column 2: Annotated text span
    
    Args:
        ann_path: Path to .ann annotation file
        doc_id: Document identifier (filename without extension)
        entry_type: Either 'trigger' for event detection or 'arg' for role labeling
        
    Returns:
        List of annotation dictionaries with document metadata and span information
    """
    entries = []
    
    if not Path(ann_path).exists():
        return entries
    
    with open(ann_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            
            if len(parts) < 3:
                continue  # Skip malformed lines
            
            ann_id = parts[0]
            type_parts = parts[1].split()
            
            if len(type_parts) < 3:
                continue  # Skip incomplete annotations
            
            label = type_parts[0]
            start = int(type_parts[1])
            end = int(type_parts[2])
            text = parts[2]
            
            # Convert BRAT annotation ID (T#/R#) to event/relation ID (E#/R#)
            entry_id = ('E' if ann_id.startswith('T') else 'R') + ann_id[1:]
            
            entry = {'doc_id': doc_id}
            
            if entry_type == 'trigger':
                entry.update({
                    'trigger_id': entry_id,
                    'trigger_type': label,
                    'trigger_text': text,
                    'trigger_start_span': start,
                    'trigger_end_span': end
                })
            else:  # argument
                entry.update({
                    'arg_id': entry_id,
                    'arg_type': label,
                    'arg_text': text,
                    'arg_start_span': start,
                    'arg_end_span': end
                })
            
            entries.append(entry)
    
    return entries


def convert_brat_to_json(
    triggers_folder: str,
    args_folder: str,
    output_dir: str,
    copy_text_files: bool = True
) -> Tuple[int, int]:
    """
    Convert BRAT annotation files to JSON format for all documents.
    
    For each document, generates:
    - {doc_id}.txt: Clinical text (copied from source)
    - {doc_id}_triggers.json: Trigger/event annotations
    - {doc_id}_args.json: Argument/role annotations
    
    Args:
        triggers_folder: Directory containing trigger annotations (.ann files)
        args_folder: Directory containing argument annotations (.ann files)
        output_dir: Target directory for JSON output
        copy_text_files: If True, copy .txt files to output directory
        
    Returns:
        Tuple of (num_documents_processed, num_files_created)
    """
    triggers_folder = Path(triggers_folder)
    args_folder = Path(args_folder)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(triggers_folder.glob('*.txt'))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {triggers_folder}")
    
    # Copy text files to output directory
    if copy_text_files:
        for txt_file in txt_files:
            shutil.copy(txt_file, output_dir / txt_file.name)
    
    num_documents = 0
    num_files = 0
    
    # Process each document
    for txt_file in txt_files:
        doc_id = txt_file.stem
        trigger_ann = triggers_folder / f'{doc_id}.ann'
        arg_ann = args_folder / f'{doc_id}.ann'
        
        # Parse annotations
        triggers = parse_ann_file(str(trigger_ann), doc_id, 'trigger')
        arguments = parse_ann_file(str(arg_ann), doc_id, 'arg')
        
        # Write trigger annotations
        trigger_json = output_dir / f'{doc_id}_triggers.json'
        with open(trigger_json, 'w', encoding='utf-8') as f:
            json.dump(triggers, f, ensure_ascii=False, indent=2)
        num_files += 1
        
        # Write argument annotations
        arg_json = output_dir / f'{doc_id}_args.json'
        with open(arg_json, 'w', encoding='utf-8') as f:
            json.dump(arguments, f, ensure_ascii=False, indent=2)
        num_files += 1
        
        num_documents += 1
    
    return num_documents, num_files


def main() -> None:
    """
    Main function for standalone execution.
    
    Update the paths below for your environment.
    """
    logger = logging.getLogger(__name__)
    
    # Input directories
    triggers_folder = '/ToxHabits(ToxNER)_Train_ANNFiles/train_annotations'
    args_folder = '/ToxHabits(ToxUse)_Train_ANNFiles/train_annotations_task_2'
    
    # Output directory
    output_dir = Path('/kaggle/working/')
    
    logger.info("Starting BRAT to JSON conversion")
    logger.info(f"  Trigger folder: {triggers_folder}")
    logger.info(f"  Argument folder: {args_folder}")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        num_docs, num_files = convert_brat_to_json(
            triggers_folder=triggers_folder,
            args_folder=args_folder,
            output_dir=str(output_dir),
            copy_text_files=True
        )
        
        logger.info(f"Conversion successful: {num_docs} documents processed")
        logger.info(f"Generated {num_files} JSON files")
        logger.info(f"Output saved to: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
