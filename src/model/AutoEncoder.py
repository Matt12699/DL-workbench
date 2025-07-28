import os
import torch
import torch.nn.functional as F

from torch import nn
from utilities.config_manager import ConfigManager
from model.IDSModel import IDSModel

class AutoEncoder(nn.Module):

    def __init__(self, encoder_dropout, decoder_dropout, encoder_layers, decoder_layers, input_dim, latent_dim):

        # Eredito il costruttore della classe base
        super(AutoEncoder, self).__init__()

        self.encoder=IDSModel(
            dropout=encoder_dropout,
            hidden_layers=encoder_layers,
            input_dim=input_dim,
            output_dim=latent_dim
        )

        self.decoder=IDSModel(
            dropout=decoder_dropout,
            hidden_layers=decoder_layers,
            input_dim=latent_dim,
            output_dim=input_dim
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    