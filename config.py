import torch

class Config:
    def __init__(self):
        self.N = 6 # The Transformer model consists of a stack of N identical layers. This hyperparameter determines how deep the model is and affects its capacity to learn complex patterns
        self.d_model = 512 # determines the dimensionality of the model's internal representations at each layer
        self.h = 8 # The number of attention heads, denoted as 'h,' is a hyperparameter that influences the diversity and quality of learned representations
        self.d_ff = 2048 # determines the dimension of the intermediate representations in the feed-forward neural networks within the Transformer layers
        self.dropout = 0.1 # Dropout is applied within the model for regularization. It helps prevent overfitting by randomly dropping out a fraction of neurons during training.
        self.src_vocab = 11 # input vocab size
        self.tgt_vocab = 11 # output vocab size
        self.max_len = 5000 # The model's architecture limits the maximum sequence length it can handle. For very long sequences, you may need to truncate or split them.
        self.lr = 1e-3 # initial learning rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'