# DCGC
This repository contains the official PyTorch implementation of the paper

Using mov100k as an example, you should obtain three files:

all_indices.txt: stores the tensor indices in the form (user_id, item_id, context_id).
all_values.txt: stores the corresponding tensor values.
tensor_shape.txt: stores the overall tensor shape.

main.py: The main entry point of the project. It handles the complete training loop, validation, and testing phases.

model.py: Contains the core architecture of the DCGC model. This includes the embedding layer, the mixture of convolutional experts, the dual-attention mechanism, the personalized gating module, and the implementation of the contrastive learning loss for the auxiliary task.

data_generator.py: Utilities for tensor data processing. 

metrics.py: Contains the implementation of evaluation metrics (e.g., RMSE, MAE) used to assess the performance of the tensor factorization task.
