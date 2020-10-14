# Neural news recommendation approach with multi-head self- attention (NRMS)

NRMS implementation (Wu et al., 2019): - Neural News Recommendation with Multi-Head Self-Attention.

## Description

The core of this approach is a news encoder and a user encoder. In the news encoder,
we use multi-head self-attentions to learn news representations from news titles by
modeling the interactions between words. In the user encoder, we learn representations
of users from their browsed news and use multi- head self-attention to capture the
relatedness between the news.

## Installation

In order to set up the necessary environment:

1. create an environment `nrms` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate newsrecom
   ```
