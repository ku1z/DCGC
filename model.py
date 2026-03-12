# Helper Functions (PyTorch Version)
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import torch.nn as nn


def cl_loss(con_puts, indices, y_true, min_v=1.0, max_v=5.0):
    """
    Contrastive Loss function with Interval-based Negative Sampling.
    
    Args:
        con_puts: Predicted scores (Tensor).
        indices: Sample indices for grouping (Tensor).
        y_true: Ground truth scores (Tensor).
        min_v: Minimum possible score for interval calculation.
        max_v: Maximum possible score for interval calculation.
    """
    device = con_puts.device
    con_puts = con_puts.squeeze()  # Shape: (N,)
    y_true = y_true.squeeze()      # Shape: (N,)
    

    tensor_min = torch.min(con_puts)
    tensor_max = torch.max(con_puts)

    normalized_tensor = (con_puts - tensor_min + 1e-7) / (tensor_max - tensor_min + 1e-7)
    
    # Scale and exponentiate (Temperature scaling implicit in scale_factor)
    scale_factor = 0.1
    scaled_tensor = normalized_tensor / scale_factor
    exp_tensor = torch.exp(scaled_tensor) # Shape: (N,)
    
    # -------------------------------------------------------------------------

    mask_dim0 = torch.eq(indices[:, 0].unsqueeze(1), indices[:, 0].unsqueeze(0))
    mask_dim1 = torch.eq(indices[:, 1].unsqueeze(1), indices[:, 1].unsqueeze(0))
    mask_dim2 = torch.eq(indices[:, 2].unsqueeze(1), indices[:, 2].unsqueeze(0))
    mask_indice = mask_dim0 | mask_dim1 | mask_dim2 # Shape: (N, N)
    

    interval_range = max_v - min_v
    step = interval_range / 3.0
    
    threshold_low = min_v + step
    threshold_high = min_v + 2 * step
    
    levels = torch.zeros_like(y_true, dtype=torch.long)
    
    # 区间 1 (Low): [min, threshold_low)
    levels[y_true < threshold_low] = 1
    
    # 区间 2 (Mid): [threshold_low, threshold_high)
    levels[(y_true >= threshold_low) & (y_true < threshold_high)] = 2
    
    # 区间 3 (High): [threshold_high, max]
    levels[y_true >= threshold_high] = 3
    

    levels[y_true >= max_v] = 3 
    levels[levels == 0] = 1 
    
    levels_i = levels.unsqueeze(1) # (N, 1)
    levels_j = levels.unsqueeze(0) # (1, N)
    mask_level_lower = torch.lt(levels_j, levels_i) # Level_j < Level_i
    
    mask = mask_indice & mask_level_lower # Shape: (N, N)
    
    replicated_exp_tensor = exp_tensor.unsqueeze(1).repeat(1, exp_tensor.shape[0]) # Shape: (N, N)
    
    candidate_vals = torch.where(mask, replicated_exp_tensor.T, torch.tensor(0.0, device=device))
    
    replicated_tensor_true = y_true.unsqueeze(1).repeat(1, y_true.shape[0])
    transposed_tensor_true = replicated_tensor_true.T
    
    weight_m = replicated_tensor_true - transposed_tensor_true
    weight_non = torch.where(mask, weight_m, torch.tensor(0.0, device=device))
    
    weight_sum = torch.sum(weight_non, dim=1, keepdim=True)
    mask_true_count = torch.sum(mask.float(), dim=1, keepdim=True)
    
    weight = (weight_non / (weight_sum + 1e-7)) * mask_true_count
    

    weighted_denominator = torch.sum(weight * candidate_vals, dim=1) # Shape: (N,)
    
    final_denominator = exp_tensor + weighted_denominator
    
    individual_scores = exp_tensor / (final_denominator + 1e-10) 
    

    log_scores = -torch.log(individual_scores + 1e-10) + torch.log(torch.tensor(2.0, device=device))
    
    average_score = torch.mean(log_scores)
    
    return average_score

def restore(data):
    """Stacks a list of index tensors. PyTorch version."""
    x = torch.stack(data, dim=-1) # Stacks along a new last dimension -> (N, 3)
    # The original has a squeeze, but the input shape is (N,), so stacking gives (N, 3)
    # and no squeeze is needed. If input was (N, 1), this would be (N, 1, 3)
    # and `y = torch.squeeze(x, dim=1)` would be correct. We assume input indices are 1D.
    return x



class DCGC(nn.Module):
    ''' add U(x) as gating  final model'''
    def __init__(self, shape, rank, nc=30):
        super().__init__()
        self.shape = shape
        self.rank = rank
        
        # --- Layer Definitions ---
        
        # Embedding layers
        self.embeds = nn.ModuleList([
            nn.Embedding(num_embeddings=s, embedding_dim=rank) for s in shape
        ])
        

        self.convl1 = nn.Conv2d(
            in_channels=1, out_channels=nc, kernel_size=(1, 6)
        )
        self.convl2 = nn.Conv2d(
            in_channels=nc, out_channels=nc, kernel_size=(rank, 1)
        )
        self.convl3 = nn.Conv2d(
            in_channels=nc, out_channels=nc, kernel_size=(rank, 1)
        )

        self.norm1 = nn.LayerNorm(rank)
        self.key_emb = nn.Linear(rank, rank)
        self.val_emb = nn.Linear(rank, rank)
        self.u_emb = nn.Linear(rank, rank)
        self.lo = nn.Linear(rank, rank)
        self.dropout = nn.Dropout(0.3)

        # --- MLP for 'y' branch (contrastive loss) ---

        self.mlp1 = nn.Sequential(
            nn.Linear(nc, 1),
            nn.ReLU()
        )
    
        self.mlp2 = nn.Sequential(
            nn.Linear(nc, 1),
            nn.ReLU()
        )


        self._init_weights()
        

    def _init_weights(self):
        # Initialize weights for all modules
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='relu')


    def forward(self,y_true,  indices):
        embed_list = [
            self.embeds[i](indices[i]) for i in range(len(self.shape))
        ] # List of 3 tensors of shape (N, rank)

        user_embed, item_embed, context_embed = embed_list[0], embed_list[1], embed_list[2]

        # 2. pariwese interaction (element-wise product)
        user_item_interaction = user_embed * item_embed      # Shape: (N, rank)
        user_context_interaction = user_embed * context_embed  # Shape: (N, rank)
        item_context_interaction = item_embed * context_embed  # Shape: (N, rank)

        all_embeddings = [
            user_embed, 
            item_embed, 
            context_embed,
            user_item_interaction,
            user_context_interaction,
            item_context_interaction
        ]

        x = torch.stack(all_embeddings, dim=1) #  Shape: (N, 6, rank)

        x = x.view(x.shape[0], 1, self.rank, 6)  #torch.Size([N, 1, rank, 3])

        # --- Feature Engineering ---
        x = F.relu(self.convl1(x)) # Shape: (N, nc, rank, 1)

        # --- attention mechanism ---
        u = self.u_emb(x.squeeze(3))  # Shape: (N, nc, rank)
        u = nn.SiLU()(u)  # activation

        key = self.key_emb(x.squeeze(3))  # Shape: (N, nc, rank)
        scaling_factior = np.sqrt(self.rank)
        key = key / scaling_factior

        # --- attention score ---

        key = nn.SiLU()(key)  # activation
        # key = nn.SiLU()(key)  # activation

        channel_attention = F.softmax(key, dim=1)  # Shape: (N, nc, rank)  select feature
        feature_attention = F.softmax(key, dim=2)  # Shape: (N, nc, rank)  select channel

        '''dropout'''
        # channel_attention = self.dropout(channel_attention)
        # feature_attention = self.dropout(feature_attention)

        
        # --- mixing module ---
        val = self.val_emb(x.squeeze(3))  # Shape: (N, nc, rank)

        val = nn.SiLU()(val)  # activation
        # val = nn.SiLU()(val)  # activation

        x_att = channel_attention * val + feature_attention * val  # Shape: (N, nc, rank)

        x_att = nn.SiLU()(self.lo(x_att * u))  # gating
        # x_att = x_att * torch.sigmoid(u)  # gating

        '''connection'''
        # x = self.norm1(key+x_att)  # KFconnection
        x = self.norm1(x.squeeze(3)+x_att)  # IFconnection

        '''no connection'''
        # x = x_att


        x = x.view(x.shape[0], x.shape[1], x.shape[2], 1) # Reshape to (N, nc, 1, rank)
        x_pre = F.relu(self.convl2(x)) # Shape: (N, nc, 1, 1)
        x_con = F.relu(self.convl3(x))
        x_pre = x_pre.view(x.shape[0], x.shape[1]) 
        x_con = x_con.view(x.shape[0], x.shape[1])

        
        # --- 'x' branch for final prediction ---
        # att1_out,_ = self.att_1(x_pre, x_pre)
        prediction = self.mlp1(x_pre)

        # att2_out,_ = self.att_2(x_con, x_con)
        cons_output = self.mlp2(x_con)

        return prediction, cons_output
    
