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
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def generate(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        
class BigramLanguageModel(nn.Module):
    """Decoder Only. Ref: nanoGPT"""
    def __init__(self, decoder, embed, generator):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, idx, mask):
        return self.decoder(self.embed(idx), self.embed(idx), mask, mask)
    
    def generate(idx, max_):
        pass