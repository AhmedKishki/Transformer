import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'src_vocab': 11, 
    'tgt_vocab': 11, 
    'N': 1, # Number of encoder/decoder blocks
    'd_model': 512, # 
    'd_ff': 2048, #
    'h': 8, # Number of attention heads
    'dropout': 0.1,
    'model': 'EncoderDecoder',
    'device': device, # default device
    'init': None,
}