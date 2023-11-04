import torch

class Mask:
    def __init__(self, size):
        self.size = size
            
    def subsequent_mask(self):
        "Mask out subsequent positions."
        attn_shape = (1, self.size, self.size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

