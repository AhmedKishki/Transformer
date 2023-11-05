import torch
from torch import nn

class Embeddings(nn.Module):
    
    def __init__(self, d_model, vocab, dropout, max_len=5000):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings