import torch
import torch.nn.functional as F

from torch import nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0] // 2  # invece di usare labels.shape[0]

        labels = labels.contiguous().view(-1, 1)  # [2B, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [2B, 2B]

        contrast_features = F.normalize(features, dim=1)  # [2B, D]
        similarity_matrix = torch.matmul(contrast_features, contrast_features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size).to(device)  # ora batch_size è la metà

        mask = mask * logits_mask

        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

