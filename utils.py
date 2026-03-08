import numpy as np
import torch
from collections import Counter
import numpy as np
from typing import Tuple, Union, List
import random,copy



def RegLoss(model):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.norm(param, p=2)
    return reg_loss


def get_l1_l2_regularization_loss(model, l1_lambda, l2_lambda):
    """Calculates L1 and L2 regularization penalties for all Linear and Embedding layers."""
    l1_loss = torch.tensor(0., device=next(model.parameters()).device)
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)

    # for name, param in model.named_parameters():
    #     # We only apply regularization to weights of Linear and Embedding layers, similar to kernel_regularizer
    #     if 'weight' in name and ('mlp' in name or 'embeds' in name):
    #         if l1_lambda > 0:
    #             l1_loss += torch.norm(param, 1)
    #         if l2_lambda > 0:
    #             l2_loss += torch.norm(param, 2).pow(2)  # L2 norm squared
    for param in model.parameters():
        l1_loss += torch.norm(param,1)
        l2_loss += torch.norm(param,2)

    return l1_lambda * l1_loss + l2_lambda * l2_loss

