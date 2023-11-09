import copy
from torch import nn
from Layers import *
from Architectures import *

c = copy.deepcopy

def make_model(config):
    "Helper: Construct a model from hyperparameters."
    attn = MultiHeadedAttention(config.h, config.d_model)
    ff = FeedForward(config.d_model, config.d_ff, config.dropout)
    src_embed = Embeddings(config.d_model, config.src_vocab, config.dropout, config.max_len)
    tgt_embed = Embeddings(config.d_model, config.tgt_vocab, config.dropout, config.max_len)
    projection = nn.Linear(config.d_model, config.tgt_vocab)
    encoder = Encoder(EncoderLayer(config.d_model, c(attn), c(ff), config.dropout), config.N)
    decoder = Decoder(DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout), config.N)
    
    model = EncoderDecoder(encoder, decoder, projection, src_embed, tgt_embed).to(config.device)
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model