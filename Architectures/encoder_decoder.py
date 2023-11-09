import torch
from torch import nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    """A standard Encoder-Decoder architecture"""
    def __init__(self, encoder, decoder, projection, src_embed, tgt_embed):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.proj = projection
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask=None)
        return self.decode(memory, tgt, src_mask, tgt_mask)

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, tgt, src_mask=None, tgt_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def generate(self, src, src_mask, max_len, start_symbol):
        """greedy decoder"""
        memory = self.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
            out = self.decode(memory, ys, src_mask, tgt_mask)
            prob = F.log_softmax(self.proj(out[:, -1]), dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

class Config:
    """
    Hyperparameters:
    
    batch_size: int = 100
        number of independent sequences to be processed in parallel
        
    src_vocab: int = 11
        input vocab size
    
    tgt_vocab: int = 11
        output vocab size
    
    N: int = 6
       model consists of a stack of N identical layers. This hyperparameter determines how deep the model is and affects 
       its capacity to learn complex patterns
    
    d_model: int = 8
        determines the dimensionality of the model's internal representations at each layer
    
    h: int = 8
        number of attention heads, denoted as 'h,' is a hyperparameter that influences the diversity and quality of learned 
        representations
    
    d_ff: int = 2048
        determines the dimension of the intermediate representations in the feed-forward neural networks within the Transformer 
        layers
    
    dropout: float = 0.1
        dropout is applied within the model for regularization. It helps prevent overfitting by randomly dropping out a fraction 
        of neurons during training.
    
    max_len: int = 5000 
        The model's architecture limits the maximum sequence length it can handle. For very long sequences, you may need to truncate 
        or split them.
    
    lr: float = 1e-3
        initial learning rate
    """    
    def __init__(self, src_vocab=11, tgt_vocab=11):
        self.batch_size = 32
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.N = 6
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.max_len = 5000
        self.base_lr = 1.0
        self.lr = 1e-3
        self.warmup = 3000
        self.max_padding = 72
        self.distributed = False
        self.num_epochs = 8
        self.accum_iter = 10
        self.file_prefix = "model_"
        self.device = 'cuda'