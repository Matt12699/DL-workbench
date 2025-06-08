import os
import torch
import torch.nn.functional as F

from torch import nn
from utilities.config_manager import ConfigManager

class IDSModel(nn.Module):

    def __init__(self, dropout, hidden_layers, input_dim, output_dim):

        # Eredito il costruttore della classe base
        super(IDSModel, self).__init__()

        # Creo i layer in base a quello che ho impostato nel file json
        layers=[]
        for i, unit in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.LayerNorm(unit))
            layers.append(nn.ReLU())
            if i < len(hidden_layers) -1 :
                layers.append(nn.Dropout(dropout))
            input_dim=unit
        layers.append(nn.Linear(unit, output_dim))

        # Creiamo un contenitore sequenziale il quale conterrà una serie di livelli neurali definiti in ordine
        # L'asterisco serve a spacchettare la lista in singoli elementi
        self.linear_relu_stack = nn.Sequential (*layers)

    # Passiamo l'input attraverso l'intera sequenza
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    