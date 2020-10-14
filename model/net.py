"""**Neural news recommendation approach with multi-head self- attention (NRMS)**.

The core of this approach is a news encoder and a user encoder. In the news encoder,
we use multi-head self-attentions to learn news representations from news titles by
modeling the interactions between words. In the user encoder, we learn representations
of users from their browsed news and use multi- head self-attention to capture the
relatedness between the news.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from newsrecom.model.doc_encoder import DocEncoder
from newsrecom.model.attention import  AdditiveAttention


class NRMS(nn.Module):
    """ NRMS implementation (Wu et al., 2019):
        - Neural News Recommendation with Multi-Head Self-Attention.
    """
    def __init__(self, hparams, weight=None):
        """Init News Encoder.

        Args:
            hparams (dict): Configuration parameters.
            weight (tensor): Embeding weight.
        """
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1)
        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, clicks, cands, labels=None):
        """Forward.

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        
        Returns:
            tensor: News Score.
        """

        num_click_docs = clicks.shape[1]
        num_cand_docs = cands.shape[1]
        num_user = clicks.shape[0]
        seq_len = clicks.shape[2]
        clicks = clicks.reshape(-1, seq_len)
        cands = cands.reshape(-1, seq_len)
        click_embed = self.doc_encoder(clicks)
        cand_embed = self.doc_encoder(cands)
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        click_embed = click_embed.permute(1, 0, 2)
        click_output, _ = self.mha(click_embed, click_embed, click_embed)
        click_output = F.dropout(click_output.permute(1, 0, 2), 0.2)

        click_repr = self.proj(click_output)
        click_repr, _ = self.additive_attn(click_output)
        logits = torch.bmm( click_repr.unsqueeze(1),
                            cand_embed.permute(0, 2, 1)).squeeze(1)
        if labels is not None:
            _, targets = labels.max(dim=1)
            loss = self.criterion(logits, targets)
            return loss, logits
        return torch.sigmoid(logits)
        