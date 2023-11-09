import torch
from torch import nn
from transformer_backend import TransformerBaseClass, Encoder

class SequenceClassifier(TransformerBaseClass):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm_f = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select hidden state of [CLS] token
        x = self.layer_norm_f(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def loss(self, input, ground_truth):
        pass
    
class Config:
    def __init__(self, num_labels, vocab_size):
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.max_position_embeddings = 5000
        self.hidden_size = 512
        self.intermediate_size = 512
        self.num_attn_heads = 6
        self.hidden_dropout_prob = 0.1
    
    @property
    def head_dim(self):
        return self.hidden_size // self.num_attn_heads
    
    @property
    def embed_dim(self):
        return self.hidden_size
    
    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'