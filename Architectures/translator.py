import torch
import torch.nn.functional as F
from torch import nn
from transformer_backend import Encoder, Decoder, TransformerBaseClass

class Translator(TransformerBaseClass):
    def __init__(self, enc_config, dec_config):
        self.encoder = Encoder(enc_config)
        self.decoder = Decoder(dec_config)
        self.dropout = nn.Dropout(dec_config.hidden_dropout_prob)
        self.layer_norm_f = nn.LayerNorm(dec_config.hidden_size)
        self.projection = nn.Linear(dec_config.hidden_size, dec_config.vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        mem = self.encoder(src, src_mask)
        x = self.decoder(tgt, mem, tgt_mask, src_mask)
        x = self.layer_norm_f(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x
    
    def generate(self, idx, src_mask, start_mask, max_new_tokens):
        pass
    
    def loss(self, input, ground_truth):
        pass