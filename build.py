import copy
from torch import nn
from helpers import *
from attention import MultiHeadedAttention
from embeddings import Embeddings, PositionalEncoding
from feedforward import FeedForward
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from architecture import EncoderDecoder, Generator


def make_model(config):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(config['h'], config['d_model'])
    ff = FeedForward(config['d_model'], config['d_ff'], config['dropout'])
    position = PositionalEncoding(config['d_model'], config['dropout'])
    src_embed = nn.Sequential(Embeddings(config['d_model'], config['src_vocab']), c(position))
    tgt_embed = nn.Sequential(Embeddings(config['d_model'], config['tgt_vocab']), c(position))
    generator = Generator(config['d_model'], config['tgt_vocab'])
    encoder = Encoder(EncoderLayer(config['d_model'], c(attn), c(ff), config['dropout']), config['N'])
    decoder = Decoder(DecoderLayer(config['d_model'], c(attn), c(attn), c(ff), config['dropout']), config['N'])
    
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model