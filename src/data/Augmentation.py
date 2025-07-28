import torch
import torch.nn as nn
import numpy as np

class Augmentation:
    def __init__(self, noise_std=0.01, mask_prob=0.1):
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    def __call__(self, x):
        x = x.clone()

        # Aggiunta di rumore gaussiano
        noise = torch.randn_like(x) * self.noise_std
        x += noise

        # Mascheratura casuale (mette a 0 alcune feature)
        mask = torch.rand_like(x) < self.mask_prob
        x[mask] = 0

        return x
