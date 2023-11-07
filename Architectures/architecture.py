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

class BigramLanguageModel(nn.Module):
    """Decoder Only. Ref: nanoGPT"""
    def __init__(self, decoder, projection, embedding):
        super().__init__()
        self.decoder = decoder
        self.embed = embedding
        self.proj = projection

    def forward(self, idx):
        return self.decoder(self.embed(idx), self.embed(idx), None, subsequent_mask(idx.size(1)).type_as(idx.data))

    def generate(self, idx, max_len):
        for _ in range(max_len):
            logits = self.proj(self(idx))
            # focus only on the last time step
            logits = logits[:, -1] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
          
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0