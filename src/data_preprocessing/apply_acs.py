"""
Main script for applying Adaptive Context Selection (ACS) to ToxHabit corpus.

This script processes clinical case documents from BRAT annotation format through
the complete pipeline: BRAT → JSON → ACS → Tokenized CSV.

Pipeline:
1. Convert BRAT annotations (.ann) to JSON format
2. Apply ACS (clause validity check) for sentence segmentation
3. Tokenize and export to CSV files for model training

ACS implemented:
- Ensure all segments form valid independent clauses with ROOT verb and subject

Usage:
    python apply_acs.py
"""

import logging
from pathlib import Path
import spacy
from transformers import AutoTokenizer

from convert_ann_to_json import convert_brat_to_json
from load_samples import load_samples
from tokenize_export import export_with_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main() -> None:
    """
    Execute complete ACS preprocessing pipeline on ToxHabit datasets.
    
    Full pipeline: BRAT annotations → JSON → ACS segmentation → Tokenized CSV
    
    Processes multiple dataset splits and exports all intermediate and final
    outputs to organized subdirectories.
    """

    # Define datasets with BRAT annotation paths
    datasets_config = {
        "train": {
            "triggers_folder": "../data/train/ToxHabits(ToxNER)_Train_ANNFiles/train_annotations",
            "args_folder": "../data/train/ToxHabits(ToxUse)_Train_ANNFiles/train_annotations_task_2",
            "is_train": True
        },
        "test": {
            "triggers_folder": "../data/test/ToxHabits(ToxNER)_Test_ANNFiles/test_annotations",
            "args_folder": "../data/test/ToxHabits(ToxUse)_Test_ANNFiles/test_annotations_task_2",
            "is_train": False
        }
    }
    
    # Base output directory
    output_base = Path("../data/processed/")
    
    # Configure logger
    logger = logging.getLogger(__name__)
    
    # Load pre-trained models
    logger.info("Initializing ToxHabit ACS preprocessing pipeline")
    logger.info("Loading pre-trained models...")
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    nlp = spacy.load("es_core_news_sm")
    logger.info("Model loading complete")
    
    # Process each dataset
    for dataset_name, config in datasets_config.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        triggers_folder = config["triggers_folder"]
        args_folder = config["args_folder"]
        is_train = config["is_train"]
        
        # Create output directories
        dataset_output = output_base / dataset_name
        json_output = dataset_output / "json"
        csv_output = dataset_output / "csv"
        
        # Step 1: Convert BRAT to JSON
        logger.info("Step 1/3: Converting BRAT annotations to JSON format")
        logger.info(f"  Trigger annotations: {triggers_folder}")
        logger.info(f"  Argument annotations: {args_folder}")
        
        try:
            num_docs, num_files = convert_brat_to_json(
                triggers_folder=triggers_folder,
                args_folder=args_folder,
                output_dir=str(json_output),
                copy_text_files=True
            )
            logger.info(f"  Converted {num_docs} documents, generated {num_files} JSON files")
            logger.info(f"  Output: {json_output}")
        except Exception as e:
            logger.error(f"  BRAT conversion failed: {e}")
            logger.warning(f"  Skipping dataset: {dataset_name}")
            continue
        
        # Step 2: Load JSON samples
        logger.info("Step 2/3: Loading JSON samples")
        try:
            samples = load_samples(str(json_output))
            logger.info(f"  Loaded {len(samples)} document samples")
        except Exception as e:
            logger.error(f"  Sample loading failed: {e}")
            continue
        
        # Step 3: Apply ACS and export to CSV
        logger.info("Step 3/3: Applying ACS and tokenizing")
        logger.info("  Sentence splitting: spaCy sentence boundary detection")
        logger.info("  Subsegmentation: semicolon and comma boundaries")
        logger.info("  ACS Rule: Merge segments containing invalid clauses")
        
        try:
            export_with_config(
                samples=samples,
                output_dir=str(csv_output),
                nlp=nlp,
                tokenizer=tokenizer,
                is_train=is_train
            )
            logger.info(f"  CSV export complete: {csv_output}")
        except Exception as e:
            logger.error(f"  Export failed: {e}")
            continue
        
        logger.info(f"Dataset {dataset_name} processed successfully")
    
    # Summary
    logger.info("Pipeline execution completed")
    logger.info(f"All outputs saved to: {output_base.absolute()}")
    logger.info("Directory structure: {dataset_name}/json/ and {dataset_name}/csv/")
    logger.info("Generated files: triggers_tagging.csv, args_tagging.csv, sentences.csv")


if __name__ == "__main__":
    main()
