import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_embeddings, num_heads, masked):
        super(MultiHeadSelfAttention, self).__init__()

        self.dim_embeddings = dim_embeddings
        self.num_heads = num_heads
        self.masked = masked

        self.query_weights = nn.Linear(dim_embeddings, dim_embeddings * num_heads, bias=False)
        self.key_weights = nn.Linear(dim_embeddings, dim_embeddings * num_heads, bias=False)
        self.value_weights = nn.Linear(dim_embeddings, dim_embeddings * num_heads, bias=False)

        self.reduce_heads = nn.Linear(dim_embeddings * num_heads, dim_embeddings, bias=False)

    def forward(self, inputs):
        num_batches, num_tokens, dim_embeddings = inputs.size()

        query = self.query_weights(inputs).view(num_batches, num_tokens, self.num_heads, dim_embeddings)
        key = self.key_weights(inputs).view(num_batches, num_tokens, self.num_heads, dim_embeddings)
        value = self.value_weights(inputs).view(num_batches, num_tokens, self.num_heads, dim_embeddings)

        query = query.transpose(1, 2).contiguous().view(num_batches * self.num_heads, num_tokens, dim_embeddings)
        key = key.transpose(1, 2).contiguous().view(num_batches * self.num_heads, num_tokens, dim_embeddings)
        value = value.transpose(1, 2).contiguous().view(num_batches * self.num_heads, num_tokens, dim_embeddings)

        attention = torch.bmm(query, key.transpose(1, 2))
        attention = attention / (dim_embeddings ** 0.5)

        if self.masked:
            masked_indices = torch.triu_indices(num_tokens, num_tokens, offset=1)
            attention[:, masked_indices[0], masked_indices[1]] = float('-inf')

        attention = F.softmax(attention, dim=2)
        
        attention = torch.bmm(attention, value).view(num_batches, self.num_heads, num_tokens, dim_embeddings)
        attention = attention.transpose(1, 2).contiguous().view(num_batches, num_tokens, dim_embeddings * self.num_heads)

        attention = self.reduce_heads(attention)

        return attention
