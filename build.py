import copy
from torch import nn
from helpers import *
from attention import MultiHeadedAttention
from embeddings import Embeddings, PositionalEncoding
from feedforward import FeedForward
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from architecture import Transformer, Generator

def make_model(
    src_vocab, 
    tgt_vocab, 
    N=6, 
    d_model=512, 
    d_ff=2048, 
    h=8,
    dropout=0.1,
    use_encoder=True,
    use_decoder=True):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    
    encoder = None
    decoder = None
    attn = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    generator=Generator(d_model, tgt_vocab)
    if use_encoder:
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    if use_decoder:
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    
    model = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        generator)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model