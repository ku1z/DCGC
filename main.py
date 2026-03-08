import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pprint import pprint
import numpy as np
import math
import time
from typing import List, Dict, Any
# form sklearn.model_selection import KFold # 不再需要
from torch.utils.data import Dataset, DataLoader
from data_generator import data_generator
import json
import argparse


from model import cl_loss, DCGC
from metrics import set_seed, get_metrics, transform
from utils import RegLoss, get_l1_l2_regularization_loss


class SparseTensorDataset(Dataset):
    """Custom PyTorch Dataset for our sparse tensor data."""
    def __init__(self, indices, values):
        self.indices = torch.from_numpy(indices).long()
        self.values = torch.from_numpy(values).float()

    def __len__(self):
        return len(self.indices) 

    def __getitem__(self, idx):
        return self.indices[idx], self.values[idx]

def main(args):
    # --- GPU Setup ---
    gpu_id = 3

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print("Using device: {}".format(device))

    meta_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = meta_path + '/data/' + args.dataset

    results_dir = os.path.join(meta_path, 'results', args.dataset, f"{time.strftime('%Y%m%d-%H%M%S')}_conv_channel_{args.conv_channels}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    args_dict = vars(args)
    with open(os.path.join(results_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    print("Hyperparameters saved to hyperparameters.json")

    # --- Set Global Seed for Reproducibility ---
    set_seed(args.seed)

    # --- Load the ENTIRE Dataset ---
    print("Loading data...")
    try:
        shape = np.loadtxt(os.path.join(data_folder, 'tensor_shape.txt')).astype(int).tolist()
        all_indices = np.loadtxt(os.path.join(data_folder, 'all_indices.txt')).astype(int)
        all_values = np.loadtxt(os.path.join(data_folder, 'all_values.txt')).astype(np.float32).reshape(-1, 1)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print("Error: Data file not found. Details: {}".format(e))
        exit()



    total_samples = len(all_indices)
    indices_perm = np.random.permutation(total_samples) 
    
    all_indices = all_indices[indices_perm]
    all_values = all_values[indices_perm]
    
    n_train = int(total_samples * 0.8)
    n_val = int(total_samples * 0.1)

    tr_idxs = all_indices[:n_train]
    tr_vals = all_values[:n_train]
    
    val_idxs = all_indices[n_train : n_train + n_val]
    val_vals = all_values[n_train : n_train + n_val]
    
    test_idxs = all_indices[n_train + n_val :]
    test_vals = all_values[n_train + n_val :]
    
    print(f"Data Split Summary: Train={len(tr_idxs)}, Val={len(val_idxs)}, Test={len(test_idxs)}")


    model = DCGC(shape=shape, rank=args.rank, nc=args.conv_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)
    mse_loss_fn = nn.MSELoss()

    train_gen = data_generator(tr_idxs, tr_vals, args.batch_size, args.chunk_num)
    steps_per_epoch = math.ceil(len(tr_vals) / args.batch_size)


    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    best_model_path = os.path.join(results_dir, "best_model.pth")
    patience_counter = 0

    print("\nStart Training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()

        for step in range(steps_per_epoch):
            try:
                inputs, targets = next(train_gen)
            except StopIteration:
                train_gen = data_generator(tr_idxs, tr_vals, args.batch_size, args.chunk_num)
                inputs, targets = next(train_gen)

            value_tensor, index_list = inputs
            value_tensor, targets = value_tensor.to(device), targets.to(device)
            index_list = [idx.to(device) for idx in index_list]

            optimizer.zero_grad()
            final_prediction, con_puts = model(value_tensor, index_list)
            
            indices_flat = torch.stack(index_list, dim=1)
            pred_squeezed = final_prediction.squeeze()
            targets_squeezed = targets.squeeze()
            con_puts_squeezed = con_puts.squeeze()

            mse_loss = mse_loss_fn(pred_squeezed, targets_squeezed) 
            

            ssl_loss = cl_loss(con_puts_squeezed, indices_flat, targets_squeezed)

            l1_l2_loss = get_l1_l2_regularization_loss(model, l1_lambda=args.l1_lambda, l2_lambda=args.l2_lambda)
            
            total_loss = mse_loss + l1_l2_loss + args.tau * ssl_loss
            
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()
        
        epoch_duration = time.time() - start_time
        avg_train_loss = total_train_loss / steps_per_epoch
        
        # --- Evaluation on Validation Set ---
        model.eval()

        val_metrics = get_metrics(model, x=[val_vals, val_idxs], y=val_vals, batch_size=args.batch_size)
        val_mae, val_rmse = val_metrics['mae'], val_metrics['rmse']
        
        print("Epoch {}/{} - {:.2f}s - loss: {:.4f} - val_mae: {:.4f} - val_rmse: {:.4f}".format(
            epoch + 1, args.epochs, epoch_duration, avg_train_loss, val_mae, val_rmse
        ))

        # Checkpoint based on MAE (or RMSE, depending on preference)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            # print(f"  Best model saved! (MAE: {best_val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at Epoch {epoch + 1}.")
                break
    
    print("\nTraining Finished.")
    print(f"Best Validation MAE: {best_val_mae:.4f}")
    print(f"Best Validation RMSE: {best_val_rmse:.4f}")


    print("\n" + "="*50)
    print(" FINAL TEST EVALUATION ".center(50, "="))
    print("="*50)

    # Load the best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model from validation phase.")
    else:
        print("Warning: Best model file not found, using current model state.")

    model.eval()
    test_metrics = get_metrics(model, x=[test_vals, test_idxs], y=test_vals, batch_size=args.batch_size)
    test_mae, test_rmse = test_metrics['mae'], test_metrics['rmse']

    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Save Results
    with open(os.path.join(results_dir, "final_results.txt"), "w") as f_results:
        f_results.write("=== Final Results (8:1:1 Split) ===\n")
        f_results.write(f"Validation Best MAE: {best_val_mae:.4f}\n")
        f_results.write(f"Validation Best RMSE: {best_val_rmse:.4f}\n")
        f_results.write("-" * 30 + "\n")
        f_results.write(f"Test MAE:  {test_mae:.4f}\n")
        f_results.write(f"Test RMSE: {test_rmse:.4f}\n")
    
    print(f"Final results saved to {os.path.join(results_dir, 'final_results.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=6e-5, help="learn rate")
    parser.add_argument("--rank", type=int, default=30, help="tensor rank")
    parser.add_argument("--tau", type=float, default=0.2, help="ssl loss weight")
    parser.add_argument("--epochs", type=int, default=1000, help="training epoches")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for training")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--chunk_num", type=int, default=4, help="chunk num")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--l1_lambda", type=float, default=0, help="L1 regularization lambda")
    parser.add_argument("--l2_lambda", type=float, default=1e-2, help="L2 regularization lambda")
    parser.add_argument("--dataset", type=str, default='mov100k', help="dataset name")
    parser.add_argument("--conv_channels", type=int, default=32, help="number of convolutional channels")
    args = parser.parse_args()
    main(args)
