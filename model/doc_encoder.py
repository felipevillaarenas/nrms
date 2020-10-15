"""**News Encoder**.

The news encoder module is used to learn news representations from news titles.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import  AdditiveAttention


class DocEncoder(nn.Module):
    """ News Encoder implementation (Wu et al., 2019):
        - Neural News Recommendation with Multi-Head Self-Attention.
    """
    def __init__(self, hparams, weight=None) -> None:
        """Init News Encoder.

        Args:
            hparams (dict): Configuration parameters.
            weight (tensor): Embeding weight.
        """
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(100, 300)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.mha = nn.MultiheadAttention(hparams['embed_size'], hparams['nhead'], 0.1)
        self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])

    def forward(self, x):
        """Forward.

        Args:
            input (tensor): Input indexed words.
            output (tensor): News encoded.
        
        Returns:
            tensor: News encoded representation.
        """
        embed = F.dropout(self.embedding(x), 0.2)
        embed = embed.permute(1, 0, 2)
        output, _ = self.mha(query=embed, key=embed, value=embed)
        output = F.dropout(output.permute(1, 0, 2))
        output = self.proj(output)
        output, _ = self.additive_attn(output)
        return output