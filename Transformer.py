import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        
        # Ensure d_model is a positive integer
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        
        # Initialize model dimensions and vocabulary size
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Define an embedding layer with vocab_size entries, each with d_model dimensions
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Ensure input is on the same device as the embedding layer
        x = x.to(self.embeddings.weight.device)
        
        # Compute embeddings and scale by sqrt(d_model)
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncodings(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super(PositionalEncodings, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape (seq_len, 1)

        # Correct the calculation of div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sin and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape (1, seq_len, d_model)

        # Register as a buffer to avoid updating during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to input tensor x
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNormalisation(nn.Module):

    def __init__(self, eta: float = 1e-6) -> None:
        super(LayerNormalisation, self).__init__()
        
        # Small epsilon value to avoid division by zero
        self.eta = eta
        
        # Initialize alpha and bias as learnable parameters, matching input dimension
        self.alpha = nn.Parameter(torch.ones(1))  # Can modify shape based on input size if needed
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Calculate mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize input with learnable scaling (alpha) and shifting (bias) terms
        return self.alpha * (x - mean) / (std + self.eta) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardBlock, self).__init__()
        
        # Define the first linear layer, dropout, and second linear layer
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply the first linear layer, ReLU activation, dropout, then the second linear layer
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        return self.linear2(x)
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout):
        super(MultiHeadAttentionBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        # Dimension of each head
        self.d_k = d_model // num_heads

        # Linear layers for query, key, value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Linear layer for the output projection
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]

        # Compute scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if any) by setting masked positions to a large negative value
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Weighted sum of values
        output = torch.matmul(attention, value)
        return output, attention

    def forward(self, q, k, v, mask=None):
        # Compute query, key, and value projections
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        batch_size = query.shape[0]

        # Reshape and transpose for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention on all heads
        x, self.attention_scores = self.attention(query, key, value, mask)

        # Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(x)
        return out

class EncoderResidualConnectionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout, d_ff=2048):
        super(EncoderResidualConnectionBlock, self).__init__()

        # Initialize Multi-Head Attention, Layer Normalization, FeedForward, and Dropout
        self.multihead_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.layer_norm = LayerNormalisation()
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # First sublayer: Multi-head Attention with Residual Connection
        out = self.multihead_attention(x, x, x, src_mask)  # Attention with query, key, value as x
        x = self.layer_norm(out + x)  # Residual Connection
        x = self.dropout(x)

        # Second sublayer: Feed-Forward with Residual Connection
        out2 = self.ff(x)
        out2 = self.dropout(out2)  # Apply dropout to output of FeedForward
        x = self.layer_norm(out2 + x)  # Second Residual Connection
        return x


class EncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout, num_layers):
        super(EncoderBlock, self).__init__()

        # Create a list of EncoderResidualConnectionBlock layers
        self.layers = nn.ModuleList([
            EncoderResidualConnectionBlock(d_model, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)  # No src_mask passed here

        return self.layer_norm(x)

    
class DecoderResidualConnectionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout, d_ff=2048):
        super(DecoderResidualConnectionBlock, self).__init__()

        # Initialize the multi-head attention blocks, layer normalization, dropout, and feedforward block
        self.multihead_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.layer_norm = LayerNormalisation()
        self.multihead_attention2 = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: Target input (decoder input)
        enc_output: Encoder output (for cross-attention)
        tgt_mask: Look-ahead mask for the target sequence (self-attention mask)
        src_mask: Padding mask for the encoder output (encoder-decoder attention mask)
        """
        
        # First multi-head attention (self-attention)
        attn = self.multihead_attention(x, x, x, tgt_mask)  # Query, Key, Value, and Target Mask
        x = self.layer_norm(attn + x)  # Apply residual connection and normalization
        x = self.dropout(x)

        # Second multi-head attention (cross-attention)
        attn2 = self.multihead_attention2(x, encoder_output, encoder_output, src_mask)  # Target to Encoder Attention
        x = self.layer_norm(attn2 + x)  # Apply residual connection and normalization
        
        # Feed-forward network
        ffn = self.ff(x)
        x = self.layer_norm(ffn + x)  # Apply residual connection and normalization

        return x

    
class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout, num_layers, d_ff=2048):
        super(DecoderBlock, self).__init__()

        # Initialize the layers of the decoder block by repeating DecoderResidualConnectionBlock
        self.layers = nn.ModuleList([DecoderResidualConnectionBlock(d_model, num_heads, dropout, d_ff) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: Target input (decoder input)
        encoder_output: Encoder output (for cross-attention)
        tgt_mask: Look-ahead mask for target sequence (to prevent attending to future tokens)
        src_mask: Padding mask for the encoder output (to prevent attention to padding tokens)
        """
        
        # Process through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply final layer normalization
        return self.layer_norm(x)

    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()
        # A fully connected linear layer that projects the model's output to vocab size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Apply the linear transformation and then log softmax for output probabilities
        return torch.log_softmax(self.linear(x), dim=-1)

    
class Transformer(nn.Module):
    def __init__(self, encoder: EncoderBlock, decoder: DecoderBlock, src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncodings, tgt_pos: PositionalEncodings, projection_layer=ProjectionLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoding
        encoder_output = self.encode(src, src_mask)

        # Decoding
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)

        # Final projection
        output = self.project(decoder_output)
        
        return output

    def encode(self, src, src_mask):
        src = self.src_embed(src)  # Embedding for the source
        src = self.src_pos(src)    # Positional encoding for the source
        src = self.encoder(src, src_mask)  # Pass through encoder
        return src
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)  # Embedding for the target
        tgt = self.tgt_pos(tgt)    # Positional encoding for the target
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # Pass through decoder
        return tgt

    def project(self, x):
        return self.projection_layer(x)  # Projection layer for output

def build_transformer_model(d_model, vocab_size, seq_len, num_heads, N, d_ff=2048, dropout=0.1):
    # Input Embeddings and Positional Encodings for source and target
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
    src_pos = PositionalEncodings(d_model=d_model, seq_len=seq_len, dropout=dropout)
    tgt_pos = PositionalEncodings(d_model=d_model, seq_len=seq_len, dropout=dropout)
    
    # Encoder and Decoder Blocks
    encoder = EncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout, num_layers=N)
    decoder = DecoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout, num_layers=N, d_ff=d_ff)
    
    # Projection Layer with correct initialization
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
    
    # Instantiate the Transformer model with the components
    model = Transformer(
        encoder=encoder, 
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer
    )
    
    return model

# Usage example
vocab_size = 1000
seq_len = 20
d_model = 512
num_heads = 8
N = 6


# Build the Transformer model
model = build_transformer_model(d_model=d_model, vocab_size=vocab_size, seq_len=seq_len, num_heads=num_heads, N=N)

# Sample input tensorss
src = torch.randint(0, vocab_size, (1, seq_len))  # Batch size of 1, sequence length of seq_len
tgt = torch.randint(0, vocab_size, (1, seq_len))  # Batch size of 1, sequence length of seq_len

# Forward pass
output = model(src, tgt)
print(output.shape)  # Expected output shape: (1, seq_len, vocab_size)

