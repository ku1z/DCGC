# 文件名: main_kfold.py

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
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from data_generator import data_generator
import json
import argparse

# Note: The model and metrics functions are simplified based on your new request
from model import  cl_loss,DCGC
from metrics import set_seed, get_metrics, transform


from utils import RegLoss,get_l1_l2_regularization_loss


# --- Custom Dataset Class ---
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

    gpu_id = 3
    torch.cuda.set_device(gpu_id)

    meta_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = meta_path + '/data/' + args.dataset
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    print("Using device: {}".format(device))

    results_dir = os.path.join(meta_path, 'results_dual_attention_2level', args.dataset, f"{time.strftime('%Y%m%d-%H%M%S')}_conv_channel_80")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")


    args_dict = vars(args)
    with open(os.path.join(results_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    print("Hyperparameters saved to hyperparameters.json")

    # --- Set Global Seed for Reproducibility ---
    set_seed(args.seed)

    # --- Load the ENTIRE Dataset ---
    print("Loading preprocessed data for K-Fold Cross-Validation...")
    try:
        shape = np.loadtxt(os.path.join(data_folder, 'tensor_shape.txt')).astype(int).tolist()
        all_indices = np.loadtxt(os.path.join(data_folder, 'all_indices.txt')).astype(int)
        all_values = np.loadtxt(os.path.join(data_folder, 'all_values.txt')).astype(np.float32).reshape(-1, 1)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print("Error: Data file not found. Details: {}".format(e))
        exit()

    # ===================================================================
    # 2. K-Fold Cross-Validation Loop
    # ===================================================================

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    all_fold_results = []

    best_val_metrics_fold = {'mae':[], 'rmse': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):


        print("\n" + "="*50)
        print(" FOLD {}/{} ".format(fold + 1, args.kfold).center(50, "="))
        print("="*50)


        with open(os.path.join(results_dir, "fold_results.txt"), "a") as f_results:
            f_results.write(f"\n=== Fold {fold+1}/{args.kfold} ===\n")
        best_model_state = None # To store the best model's state_dict for this fold
        best_val_mae_fold = float('inf')
        best_val_rmse_fold = float('inf')
        # --- 2.1. Prepare data for the current fold ---
        tr_idxs, val_idxs = all_indices[train_idx], all_indices[val_idx]
        tr_vals, val_vals = all_values[train_idx], all_values[val_idx]

        # --- 2.2. Re-initialize Model and Optimizer ---
        # The 'nc' parameter is part of the model, assuming a default or removing if not needed
        model = DCGC(shape=shape, rank=args.rank, nc=args.conv_channels).to(gpu_id)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)
        mse_loss_fn = nn.MSELoss()

        train_gen = data_generator(tr_idxs, tr_vals, args.batch_size, args.chunk_num)
        steps_per_epoch = math.ceil(len(tr_vals) / args.batch_size)

        # --- 2.3. Inner Training Loop (using DataLoader) ---

        patience_counter = 0
        best_model_path_fold = "best_model_fold_{}.pth".format(fold + 1)

        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0
            start_time = time.time()

            for step in range(steps_per_epoch):
                inputs,targets = next(train_gen)
                value_tensor, index_list = inputs
                value_tensor, targets = value_tensor.to(device), targets.to(device)
                index_list = [idx.to(device) for idx in index_list]

                optimizer.zero_grad()
                final_prediction,con_puts = model(value_tensor, index_list)
                
                indices_flat = torch.stack(index_list, dim=1)
                pred_squeezed = final_prediction.squeeze()
                targets_squeezed = targets.squeeze()
                con_puts_squeezed = con_puts.squeeze()

                mse_loss = mse_loss_fn(pred_squeezed, targets_squeezed) 
                ssl_loss = cl_loss(con_puts_squeezed, indices_flat, targets_squeezed)


                l1_l2_loss = get_l1_l2_regularization_loss(model, l1_lambda=args.l1_lambda, l2_lambda=args.l2_lambda)
                # l2_loss = l2_lambda*RegLoss(model)
                total_loss = mse_loss + l1_l2_loss + args.tau * ssl_loss
                
                total_loss.backward()
                optimizer.step()
                total_train_loss += total_loss.item()
            
            epoch_duration = time.time() - start_time
            avg_train_loss = total_train_loss / steps_per_epoch
            
            # --- Evaluation using the new get_metrics with DataLoader ---
            model.eval()
            val_metrics = get_metrics(model, x=[val_vals, val_idxs], y=val_vals, batch_size=args.batch_size)
            val_mae, val_rmse = val_metrics['mae'], val_metrics['rmse']
            
            print("Fold {} | Epoch {}/{} - {:.2f}s - loss: {:.4f} - val_mae: {:.4f} - val_rmse: {:.4f}".format(
                fold + 1, epoch + 1, args.epochs, epoch_duration, avg_train_loss, val_mae, val_rmse
            ))
            if val_mae < best_val_mae_fold:
                best_val_mae_fold = val_mae
                best_val_rmse_fold = val_rmse
                best_model_state = model.state_dict()
                patience_counter = 0
                # torch.save(model.state_dict(), best_model_path_fold)

            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered for Fold {} at Epoch {}.".format(fold + 1, epoch + 1))
                    break
            

                
        best_val_metrics_fold['mae'].append(best_val_mae_fold)
        best_val_metrics_fold['rmse'].append(best_val_rmse_fold)
        print("\nBest metrics for Fold {}: MAE={:.4f}, RMSE={:.4f}".format(
            fold + 1, best_val_mae_fold, best_val_rmse_fold
        ))

        with open(os.path.join(results_dir, "fold_results.txt"), "a") as f_results:
            f_results.write(f"Best MAE: {best_val_mae_fold:.4f}, Best RMSE: {best_val_rmse_fold:.4f}\n")

        if best_model_state is not None:
            model_save_path = os.path.join(results_dir, f"model_fold_{fold+1}_best_rmse_{best_val_rmse_fold:.4f}.pth")
            torch.save(best_model_state, model_save_path)
            print(f"Best model for Fold {fold+1} saved to {model_save_path}")

    # ... Final Aggregation part remains the same ...
    print("\n" + "="*50)
    print(" K-FOLD CROSS-VALIDATION SUMMARY ".center(50, "="))
    print("="*50)
    if best_val_metrics_fold:
        print(f"MAE: {np.mean(best_val_metrics_fold['mae']):.4f} ± {np.std(best_val_metrics_fold['mae']):.4f}")
        print(f"RMSE: {np.mean(best_val_metrics_fold['rmse']):.4f} ± {np.std(best_val_metrics_fold['rmse']):.4f}")
    else:
        print("No results were recorded from any fold.")


    with open(os.path.join(results_dir, "fold_results.txt"), "a") as f_results:
        f_results.write("\n=== Cross Validation Final Results ===\n")
        f_results.write(f"Mean MAE: {np.mean(best_val_metrics_fold['mae']):.4f} ± {np.std(best_val_metrics_fold['mae']):.4f}\n")
        f_results.write(f"Mean RMSE: {np.mean(best_val_metrics_fold['rmse']):.4f} ± {np.std(best_val_metrics_fold['rmse']):.4f}\n")
    print(f"Final results saved to {os.path.join(results_dir, 'fold_results.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=6e-5, help="learn rate")
    parser.add_argument("--rank", type=int, default=30, help="tensor rank")
    parser.add_argument("--tau", type=float, default=0.2, help="ssl loss weight")
    parser.add_argument("--epochs", type=int, default=1000, help="training epoches")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for training")
    parser.add_argument("--kfold", type=int, default=10, help="kfold")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--chunk_num", type=int, default=4, help="chunk num")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--l1_lambda", type=float, default=0, help="L1 regularization lambda")
    parser.add_argument("--l2_lambda", type=float, default=1e-2, help="L2 regularization lambda")
    parser.add_argument("--dataset", type=str, default='mov100k', help="dataset name")
    parser.add_argument("--conv_channels", type=int, default=32, help="number of convolutional channels")
    args = parser.parse_args()
    main(args)


 