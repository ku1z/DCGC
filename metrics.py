import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any

# ===================================================================
# Part 1: Metric Functions
# These now operate on either PyTorch tensors or NumPy arrays.
# =odes==================================================================


def rmse_torch(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) calculated using PyTorch tensors.
    """
    # Ensure tensors are flat
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return torch.sqrt(torch.mean(torch.square(y_pred - y_true)))

# --- The following functions are already pure NumPy and need no changes ---

def mae(y_true, y_pred) :
    """Mean Absolute Error (NumPy version)."""
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    """Root Mean Squared Error (NumPy version)."""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (NumPy version).
    This version is cleaner than the original mixed Keras/NumPy one.
    """
    epsilon = 1e-8
    diff = np.abs((y_true - y_pred) / (y_true + epsilon))
    return 100.0 * np.mean(diff)


# ===================================================================
# Part 2: Utility Functions
# ===================================================================

def set_seed(seed= 0):
    """
    Sets the random seed for reproducibility in PyTorch, NumPy, and random.
    This replaces the TensorFlow-specific `set_session`.
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # The following two lines are often recommended for deterministic results
    # but can impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def transform(idxs) :
    """
    Splits an index array into a list of column vectors.
    This function works for NumPy and is kept as is. It's useful for preparing
    data for the model input format.
    """
    return [idxs[:, i] for i in range(idxs.shape[1])]

def get_metrics(model, x, y, batch_size= 1024):
    """
    Calculates evaluation metrics for a PyTorch model.
    This function replaces the Keras `model.predict`-based version with a
    standard PyTorch evaluation loop.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        x (List): A list containing the model inputs. For HOCTC_SP, this should be
                  `[y_true_numpy, indices_numpy]`.
        y (np.ndarray): The ground truth labels (N,).
        batch_size (int): The batch size for prediction.

    Returns:
        Dict[str, float]: A dictionary containing 'rmse', 'mape', and 'mae'.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Determine the device the model is on
    device = next(model.parameters()).device
    
    x_values, x_indices = x[0], x[1]
    
    y_predictions = []
    num_samples = len(y)
    
    # Wrap the loop in `torch.no_grad()` for efficiency
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = i + batch_size
            
            # --- Prepare batch data ---
            batch_y_true_np = x_values[i:end]
            batch_indices_np = x_indices[i:end]
            
            # Convert to tensors and move to the correct device
            batch_y_true = torch.from_numpy(batch_y_true_np).float().to(device)
            # The `transform` function splits the index array into a list
            batch_indices_list = [
                torch.from_numpy(col).long().to(device) for col in transform(batch_indices_np)
            ]
            
            # --- Perform forward pass ---
            final_prediction,_ = model(batch_y_true, batch_indices_list)
            # Move predictions to CPU and store as NumPy array
            y_predictions.append(final_prediction.cpu().numpy())

    # Concatenate all batch predictions and flatten
    yp = np.concatenate(y_predictions).flatten()
    
    # Ensure the true labels `y` are also flat
    y = y.flatten()

    # Calculate metrics using the pure NumPy functions
    return {
        "rmse": float(rmse(y, yp)),
        "mae": float(mae(y, yp))
    }
