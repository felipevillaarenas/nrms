"""**Additive attention layer**.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import torch
import torch.nn as nn

from typing import Tuple, Optional


class AdditiveAttention(torch.nn.Module):
    """Additive word attention layer."""
    def __init__(self, in_dim=100, v_size=200):
        """Initialization parameters.

        Args:
            in_dim (int): Input dimension.
            v_size (int): Projection size.
        """
        super(AdditiveAttention, self).__init__()
        self.in_dim = in_dim
        self.v_size = v_size
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, in_dim]
        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        weights = self.proj_v(self.proj(context)).squeeze(-1)
        weights = torch.softmax(weights, dim=-1)
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights
