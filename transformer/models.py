import torch
from torch import nn
import torch.nn.functional as F

from transformer.layers import TransformerBlock

class AutoregressiveDecoder(nn.Module):
    def __init__(self, dim_embeddings, num_heads, num_layers, num_tokens, seq_length):
        super(AutoregressiveDecoder, self).__init__()

        self.dim_embeddings = dim_embeddings
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.seq_length = seq_length

        self.token_embeddings = nn.Embedding(embedding_dim=dim_embeddings, num_embeddings=num_tokens)
        self.positional_embeddings = nn.Embedding(embedding_dim=dim_embeddings, num_embeddings=seq_length)
        self.combine_embeddings = nn.Linear(dim_embeddings * 2, dim_embeddings)

        transformer_layers = [TransformerBlock(dim_embeddings, num_heads, masked=True) for _ in range(num_layers)]
        self.transformer_layers = nn.Sequential(*transformer_layers)

        self.output_layer = nn.Linear(dim_embeddings, num_tokens)

    def forward(self, inputs):
        token_embeds = self.token_embeddings(inputs)
        batch_size, seq_length, _ = token_embeds.size()
        
        position_embeds = torch.arange(seq_length).cuda()
        position_embeds = self.positional_embeddings(position_embeds)[None, :, :].expand(batch_size, seq_length, -1)

        embeddings = token_embeds + position_embeds

        outputs = self.transformer_layers(embeddings)
        outputs = self.output_layer(outputs.view(batch_size * seq_length, self.dim_embeddings)).view(batch_size, seq_length, self.num_tokens)

        outputs = F.log_softmax(outputs, dim=2)

        return outputs
