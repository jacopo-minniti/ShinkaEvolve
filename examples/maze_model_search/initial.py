# EVOLVE-BLOCK-START

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolvedModel(nn.Module):
    """
    The model to be evolved. 
    It must implement __init__, forward, and compute_loss (or return loss in forward).
    """
    def __init__(self):
        super().__init__()
        # Extremely simple dummy model
        # Input size: 2 * 7 * 7 = 98 (assuming obs_size=7)
        self.dummy_layer = nn.Linear(98, 4) 

    def forward(self, x):
        # x is (B, 2, obs_size, obs_size)
        # Flatten and project
        return self.dummy_layer(x.view(x.size(0), -1))

    def compute_loss(self, batch, outputs):
        """
        Computes the training loss.
        batch: dict containing 'obs', 'action', 'target', etc.
        outputs: result of forward(batch['obs'])
        """
        return F.cross_entropy(outputs, batch['target'])

# EVOLVE-BLOCK-END
