import os
import torch
import torch.nn.functional as F

from torch import nn

class IDSModel(nn.Module):

    def __init__(self, num_features):

        # Eredito il costruttore della classe base
        super(IDSModel, self).__init__()

        # Creiamo un contenitore sequenziale il quale conterrà una serie di livelli neurali definiti in ordine
        self.linear_relu_stack = nn.Sequential (
            
            nn.Linear(num_features, 128), # Layer 1: Input -> 128 neuroni
            nn.LayerNorm(128),
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 64), # Layer 2: 128 -> 64 neuroni
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(64, 32), # Layer 3: 64 -> 32 neuroni
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1) # Layer di Output: 64 -> 1 neurone 
        )

    # Passiamo l'input attraverso l'intera sequenza
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    