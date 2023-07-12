import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, max_seq_length, hidden_dim):
        super(PositionalEncoder, self).__init__()

        # Define the positional encoding layer
        self.positional_encoding = nn.Sequential(
            nn.Linear(max_seq_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, seq_length, height, width, input_dim = x.shape

        # Generate the positional encoding based on sequence length
        positional_encoding = self.positional_encoding(torch.arange(seq_length).float())
        positional_encoding = positional_encoding.unsqueeze(-2).unsqueeze(-2)

        # Expand the positional encoding to match the input tensor shape
        positional_encoding = positional_encoding.expand(batch_size, seq_length, height, width, input_dim)

        # Add the positional encoding to the input tensor
        x = x + positional_encoding

        return x