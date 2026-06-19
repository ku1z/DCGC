# DCGC

[![PyTorch](https://img.shields.io/badge/PyTorch-implementation-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **DCGC**, a tensor factorization model that combines convolutional feature extraction, dual attention, personalized gating, and an auxiliary contrastive learning objective for contextual recommendation prediction.

## Highlights

- Learns user, item, and context embeddings from sparse tensor observations.
- Models pairwise interactions with convolutional expert channels.
- Uses dual attention over channels and latent features.
- Adds a personalized gating module to refine interaction representations.
- Optimizes the main prediction task together with an interval-based contrastive learning loss.

## Repository Structure

```text
.
├── data/                  # Dataset folders, e.g. data/mov100k
├── results/               # Training logs, checkpoints, and final metrics
├── data_generator.py      # Sparse tensor batching and chunk-wise sorting
├── main.py                # Training, validation, testing, and result saving
├── metrics.py             # MAE/RMSE evaluation utilities
├── model.py               # DCGC model and contrastive learning loss
└── utils.py               # Regularization helpers
```

## Data Format

Each dataset should be placed under `data/<dataset_name>/`. For example, the default dataset path is:

```text
data/mov100k/
```

A dataset folder should contain three files:

| File | Description |
| --- | --- |
| `all_indices.txt` | Tensor coordinates in the form `(user_id, item_id, context_id)`. |
| `all_values.txt` | Observed tensor values corresponding to `all_indices.txt`. |
| `tensor_shape.txt` | Overall tensor shape, such as the number of users, items, and contexts. |

The training script randomly splits all observations into train, validation, and test sets with an `8:1:1` ratio.

## Quick Start

### 1. Install dependencies

This repository does not pin a dedicated environment file yet. Install the core dependencies first:

```bash
pip install torch numpy scikit-learn
```

### 2. Prepare data

Place the processed tensor files under `data/<dataset_name>/`. For the default configuration, use:

```text
data/mov100k/all_indices.txt
data/mov100k/all_values.txt
data/mov100k/tensor_shape.txt
```

### 3. Train and evaluate

Run DCGC with the default settings:

```bash
python main.py --dataset mov100k
```




## Outputs

Training artifacts are saved to:

```text
results/<dataset>/<timestamp>_conv_channel_<conv_channels>/
```

Each run stores:

- `hyperparameters.json`: command-line configuration for the run.
- `best_model.pth`: checkpoint selected by the best validation MAE.
- `final_results.txt`: validation and test MAE/RMSE summary.

## Citation

If you use this code in your research, please cite the corresponding paper after the citation information is released.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
