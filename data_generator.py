import numpy as np
import random
import torch

def data_generator(idxs, vals, batch_size, chunk_num):

    data = np.column_stack((idxs, vals.reshape(-1, 1)))
    chunk_size = len(data) // chunk_num
    
    while True:

        np.random.shuffle(data)
        
        chunk_order = np.arange(chunk_num)
        all_sorted_data = []

        for i in chunk_order:
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size if i != chunk_num - 1 else len(data)
            chunk = data[chunk_start:chunk_end]

            sort_orders = [
                (chunk[:, 0], chunk[:, 1], chunk[:, 2]),
                (chunk[:, 0], chunk[:, 2], chunk[:, 1]),
                (chunk[:, 1], chunk[:, 2], chunk[:, 0]),
                (chunk[:, 1], chunk[:, 0], chunk[:, 2]),               
            ]

            sorted_order = random.choice(sort_orders)
            sorted_chunk = chunk[np.lexsort(sorted_order)]
            all_sorted_data.extend(sorted_chunk)
            

        for start in range(0, len(all_sorted_data), batch_size):
            end = start + batch_size
            if end > len(all_sorted_data):
                continue

            batch = np.array(all_sorted_data[start:end])
            

            batch_idxs_np = batch[:, :3].astype(np.int64) 
            batch_vals_np = batch[:, 3].astype(np.float32) 


            batch_idxs_torch = torch.from_numpy(batch_idxs_np) 
            batch_vals_torch = torch.from_numpy(batch_vals_np)

            inputs_torch = [batch_vals_torch, [batch_idxs_torch[:, i] for i in range(3)]]

            targets_torch = batch_vals_torch
            
            yield inputs_torch, targets_torch