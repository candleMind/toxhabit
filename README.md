# I-NOWJ: Adaptive Context Selection and Knowledge-Guided Learning for Substance Use NER

Official implementation of I-NOWJ framework for Named Entity Recognition in Spanish clinical substance use texts.

## Requirements

- Python 3.12+
- Package manager: uv
- (Optional) CUDA-enabled GPU recommended for training

## Quickstart

Reproduce the main results with our pre-trained checkpoint:

This quickstart assumes you have:
1) obtained the ToxHabits dataset under `data/` (see **Data**), and  
2) generated the processed CSV files (see **Preprocessing**).

```bash
# Install dependencies
uv sync

# Download spaCy Spanish model
uv run python -m spacy download es_core_news_sm

# Evaluate on test set
cd src
uv run python -m inowj.cli.evaluate --config ../configs/toxhabits_inowj_eval.yaml
```

Results will be saved to `outputs/toxhabits_inowj_eval/evaluation/`.

## Pre-trained artifacts

This repository includes pre-trained artifacts required to reproduce the reported results:

- **Model checkpoint**: `checkpoints/inowj_model.pt`  
  Pre-trained I-NOWJ model weights used in the Quickstart evaluation.

- **Knowledge dictionaries**: `resources/dictionaries/`  
  - `trigger_dict.json`: trigger lexicon (substance mentions)  
  - `argument_dict.json`: argument lexicon (usage characteristics)

These files are committed in this repository.

## Data

### How to obtain ToxHabits

- Shared task homepage (task description, guidelines, announcements):
  [ToxHabits](https://temu.bsc.es/toxhabits/)

- Direct dataset download (Zenodo, latest version):
  [ToxHabits-NER (Zenodo record)](https://zenodo.org/records/17566029)
  (DOI: 10.5281/zenodo.15304758)

### Expected directory layout

After obtaining the dataset, organize files as follows:

```
data/
├── train/
│   ├── ToxHabits(ToxNER)_Train_ANNFiles/
│   │   └── train_annotations/          
│   └── ToxHabits(ToxUse)_Train_ANNFiles/
│       └── train_annotations_task_2/   
└── test/
    ├── ToxHabits(ToxNER)_Test_ANNFiles/
    │   └── test_annotations/
    └── ToxHabits(ToxUse)_Test_ANNFiles/
        └── test_annotations_task_2/
```

### Data licensing

Please follow the license stated on the Zenodo release and any shared task terms associated with the dataset distribution.

## Preprocessing

Convert BRAT annotations to tokenized CSV files with Adaptive Context Selection (ACS):

```bash
cd src
uv run python data_preprocessing/apply_acs.py
```

**Outputs:** Processed CSV files saved to `data/processed/{train,test}/csv/`:
- `{train|test}_triggers_tagging.csv`: Token-level trigger labels
- `{train|test}_args_tagging.csv`: Token-level argument labels
- `{train|test}_sentences.csv`: Source text metadata

## Training

Train the I-NOWJ model from scratch on preprocessed data:

```bash
cd src
uv run python -m inowj.cli.train --config ../configs/toxhabits_inowj_eval.yaml
```

Best model checkpoint saved to `outputs/toxhabits_inowj_eval/best_model.pt`.

### Self-training

Run self-training with augmented data:

```bash
cd src
uv run python -m inowj.cli.self_train --config ../configs/self_train.yaml
```

## Evaluation

Evaluate model performance on test data:

```bash
cd src
uv run python -m inowj.cli.evaluate --config ../configs/toxhabits_inowj_eval.yaml --checkpoint ../path/to/model.pt
```
## Project Structure

```
├── configs/              YAML configuration files for training/evaluation
├── src/
│   ├── inowj/           Core model implementation (architecture, training, evaluation)
│   ├── data_preprocessing/  BRAT to CSV pipeline with ACS
│   └── augment/         Data augmentation utilities
├── resources/
│   └── dictionaries/    External knowledge dictionaries (trigger and argument terms)
├── checkpoints/         Pre-trained model weights
├── data/                ToxHabits corpus (raw BRAT and processed CSV)
└── outputs/             Experiment results and logs
```

**Key modules:**
- `src/inowj/model/`: BERT-BiLSTM-CRF architecture with dictionary features
- `src/inowj/cli/`: Command-line interfaces for train/eval/self-train
- `src/inowj/eval/`: Evaluation metrics and export utilities
- `src/data_preprocessing/`: ACS preprocessing pipeline

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: 
