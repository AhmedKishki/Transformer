class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, enc_in, mask=None):
        x = self.embeddings(enc_in)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.self_attn(hidden_state, mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_config, enc_config):
        super().__init__()
        self.dec_emmbeddings = Embeddings(dec_config)
        self.enc_embeddings = Embeddings(enc_config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, dec_in, memory, dec_mask=None, enc_mask=None):
        x = self.dec_embeddings(dec_in)
        for layer in self.layers:
            x = layer(x, memory, dec_mask, enc_mask)
        return x
        
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = CrossAttentionHead(config)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x, memory, dec_mask=None, enc_mask=None):
        x = self.layer_norm_1(x)
        x = x + self.self_attn(x, dec_mask)
        x = self.layer_norm_2(x)
        x = x + self.cross_attn(x, memory, dec_mask, enc_mask)
        x = self.layer_norm_3(x)
        x = x + self.feed_forward(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state, mask):
        x = torch.cat([h(hidden_state, mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
    
class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.w_q = nn.Linear(embed_dim, head_dim)
        self.w_k = nn.Linear(embed_dim, head_dim)
        self.w_v = nn.Linear(embed_dim, head_dim)
    
    def forward(self, hidden_state, mask=None):
        attn_outputs = scaled_dot_product_attention(self.w_q(hidden_state), self.w_k(hidden_state), self.w_v(hidden_state), mask)
        return attn_outputs
    
class CrossHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([CrossAttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, dec_hidden_state, enc_hidden_state, mask):
        x = torch.cat([h(enc_hidden_state, dec_hidden_state, mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
    
class CrossAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.w_q = nn.Linear(embed_dim, head_dim)
        self.w_k = nn.Linear(embed_dim, head_dim)
        self.w_v = nn.Linear(embed_dim, head_dim)
    
    def forward(self, dec_hidden_state, enc_hidden_state, mask):
        attn_outputs = scaled_dot_product_attention(self.w_q(dec_hidden_state), self.w_k(enc_hidden_state), self.w_v(enc_hidden_state), mask)
        return attn_outputs
    
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)
    