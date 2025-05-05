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
            
            nn.Linear(num_features, 128),
            # Funzione di attivazione ReLU applicata all'output del livello precedente
            nn.ReLU(),
            # Contiene un output a due neuroni
            nn.Linear(128, 2)
        )

    # Passiamo l'input attraverso l'intera sequenza
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    