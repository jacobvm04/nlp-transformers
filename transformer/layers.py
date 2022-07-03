import torch
from torch import nn

from transformer.modules import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, dim_embeddings, num_heads, masked=False):
        super(TransformerBlock, self).__init__()

        self.dim_embeddings = dim_embeddings
        self.num_heads = num_heads

        self.norm_layer1 = nn.LayerNorm(dim_embeddings)
        self.norm_layer2 = nn.LayerNorm(dim_embeddings)

        self.self_attention = MultiHeadSelfAttention(dim_embeddings, num_heads, masked)

        self.feedforward = nn.Sequential(
            nn.Linear(dim_embeddings, dim_embeddings * 4),
            nn.ReLU(),
            nn.Linear(dim_embeddings * 4, dim_embeddings)
        )

    def forward(self, inputs):
        attention = self.self_attention(inputs)
        attention = self.norm_layer1(attention + inputs)

        feedforward = self.feedforward(attention)
        feedforward = self.norm_layer2(feedforward + attention)

        return feedforward
