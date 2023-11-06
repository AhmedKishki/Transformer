from Layers.layer import LayerNorm, SublayerConnection
from torch import nn
import copy

c = copy.deepcopy
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.s_attn = self_attn
        self.x_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([c(SublayerConnection(size, dropout)) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.s_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.x_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([c(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)