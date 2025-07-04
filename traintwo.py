import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.cuda.amp import autocast, GradScaler

# Device setup (Apple Silicon MPS backend)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1
vocab_size = 10000
seq_length = 64
batch_size = 32
num_epochs = 10
learning_rate = 0.0001


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x * self.scale + self.pe[:x.size(1)]
        return self.dropout(x)


# Language Model using PyTorch's TransformerDecoder
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Create decoder layers
        decoder_layers = TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        # Create transformer decoder
        self.decoder = TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, mask=None):
        # src shape: (batch_size, seq_length)
        embedded = self.embed(src)
        embedded = self.pos_encoder(embedded)

        # For decoder-only model, we use the same input for memory
        # PyTorch requires memory input even for decoder-only models
        output = self.decoder(
            tgt=embedded,
            memory=embedded,  # Use same as tgt for decoder-only
            tgt_mask=mask,  # Causal mask
            memory_mask=mask  # Typically None, but required here
        )

        return self.fc_out(output)


# Initialize model
model = TransformerLM().to(device)
scaler = GradScaler()  # For mixed precision gradient scaling

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Create causal mask function
def create_causal_mask(size):
    # Generate upper triangular matrix (True values are masked)
    return torch.triu(torch.ones(size, size), diagonal=1).bool().to(device)


# Synthetic data generator
def generate_batch():
    src = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    # Shifted right target (next token prediction)
    target = src[:, 1:].contiguous()
    return src, target


# Training loop
model.train()
mask = create_causal_mask(seq_length)  # (S, S)

for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()

    # Generate synthetic batch
    inputs, targets = generate_batch()

    # Mixed precision forward pass
    with autocast(dtype=torch.float16, device_type=device.type):
        outputs = model(inputs, mask)
        # Only compute loss on next-token predictions
        loss = criterion(outputs[:, :-1].contiguous().view(-1, vocab_size),
                         targets.view(-1))

    # Backpropagation with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

print("Training complete!")