from torch import nn

class SequenceClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
class Config:
    def __init__(self, num_labels, vocab_size):
        self.device = 'cuda'
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.max_position_embeddings
        self.hidden_size = 512
        self.intermediate_size = 512
        self.num_attn_heads = 6
        self.hidden_dropout_prob = 0.1
        self.head_dim = self.embed_dim // self.num_attn_heads