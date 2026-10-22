# CPC353 Natural Language Processing: Assignment 2

This repository contains a notebook-based speaker-change detection project for CPC353 Assignment 2. The workflow uses the IEMOCAP corpus to extract acoustic features, create frame-level speaker-change labels, prepare session-based data splits, and train/evaluate LSTM classifiers.

## Repository Contents

| Notebook | Purpose |
| --- | --- |
| `CPC353 ASSG2 - TASK 2 (FEATURES EXTRACTION).ipynb` | Extracts fundamental frequency (F0) and 13 MFCC coefficients from audio. Compares 20 ms and 80 ms analysis windows with a 10 ms frame shift, and includes feature visualizations. |
| `CPC353 ASSG2 - TASK 3 (DATA PREPROCESSING).ipynb` | Reads RTTM annotations, labels frames at speaker boundaries, applies Z-score normalization, and combines sessions into training, validation, and test CSVs. |
| `CPC353 ASSG2 - TASK 4 (MODEL DEVELOPMEMENT).ipynb` | Trains LSTM binary classifiers for speaker-change detection, compares window-size experiments, handles class imbalance, and reports classification metrics and plots. |

## Dataset

The notebooks expect the IEMOCAP dataset, which is not included in this repository. Arrange the audio and annotations so the dataset contains the five session directories, with audio under `SessionN/dialog/wav` and the corresponding RTTM annotation files somewhere under the dataset root.

The feature-extraction notebook currently constructs an IEMOCAP path in code. The preprocessing notebook asks for the dataset root when run. Before running the model notebook, update its train, validation, and test CSV paths to point to the files generated in your working directory.

## Setup

Use Python with Jupyter Notebook or JupyterLab. Install the libraries imported by the notebooks:

```bash
python -m pip install jupyter numpy pandas librosa praat-parselmouth tqdm scikit-learn matplotlib seaborn tensorflow
```

## Workflow

Run the notebooks in this order:

1. **Task 2: Feature extraction** processes the five IEMOCAP sessions and writes per-session feature CSVs. The two experiments use 20 ms and 80 ms windows, respectively, with a 10 ms frame shift. Features include file name, frame index and times, F0, and `mfcc_1` through `mfcc_13`.
2. **Task 3: Data preprocessing** matches audio files to RTTM annotations, adds a `speaker_change` label, and writes labeled feature CSVs. It then combines sessions 1–3 for training, session 4 for validation, and session 5 for testing. The experiments produce `train_data1.csv`, `validation_data1.csv`, `test_data1.csv` and corresponding `*data2.csv` files.
3. **Task 4: Model development** loads the prepared CSVs, trains an LSTM using the 14 acoustic features, and evaluates predictions with a classification report and confusion matrix. It includes experiments on imbalanced data, undersampling, and the two window sizes.

Generated CSVs and plots are written by the notebooks and are not currently tracked in this repository.

## Important Notes

- Several notebook cells contain absolute paths tied to the original Windows machine. Update those paths for your environment before running the cells; in particular, the feature extraction and model development notebooks use hard-coded paths.
- Task 3 normalizes each per-session feature file before combining the session splits. Review the notebook implementation if you change the split or normalization strategy.
- The notebooks contain the assignment work for Tasks 2–4; no Task 1 as it consist of assignment and context understanding.

## Authors

Nadiatul Fakhira binti Zakpar (160307) and Norita binti Muin (160453).
