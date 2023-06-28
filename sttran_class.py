# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

# patch embed --> custom class
# position encoder --> custom class from nn.Module
# encoder --> nn.TransformerEncoder()
    # encoder input layer
    # self-attention 
    # add + normalize
    # feed forward
    # add + normalize
# decoder --> nn.TransformerDecoder()
    # decoder input layer
    # self-attention
    # add + normalize
    # encoder-decoder attention
    # add + normalize
    # feed-forward
    # add + normalize 
# linear mapping --> nn.Linear()

import torch
import torch.nn as nn

class SpatiotemporalTransformer(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_dim, num_heads, num_layers):
        super(SpatiotemporalTransformer, self).__init__()

        # Spatial Transformer
        self.spatial_transformer = nn.TransformerEncoderLayer(d_model=num_channels, nhead=num_heads)

        # Temporal Transformer
        self.temporal_transformer = nn.TransformerEncoderLayer(d_model=num_channels, nhead=num_heads)

        # Positional encodings for spatial and temporal dimensions
        self.spatial_pos_enc = nn.Parameter(torch.zeros(num_frames, num_channels))
        self.temporal_pos_enc = nn.Parameter(torch.zeros(num_frames, num_channels))

        # Fully connected layer for pixel prediction
        self.fc = nn.Linear(num_channels, 3)  # Assuming RGB images

    def forward(self, x):
        batch_size, num_frames, num_channels, height, width = x.size()

        # Reshape the input to be compatible with the transformer
        x = x.view(batch_size * num_frames, num_channels, height, width)

        # Apply spatial transformation
        x = x.permute(2, 3, 0, 1)  # (height, width, batch_size*num_frames, num_channels)
        spatial_pos_enc = self.spatial_pos_enc.unsqueeze(1).repeat(1, height * width, 1)  # (num_frames, height*width, num_channels)
        x = x + spatial_pos_enc
        x = x.view(height * width, batch_size * num_frames, num_channels)  # (height*width, batch_size*num_frames, num_channels)
        x = self.spatial_transformer(x)
        x = x.view(height, width, batch_size, num_frames, num_channels)
        x = x.permute(2, 3, 4, 0, 1)  # (batch_size, num_frames, num_channels, height, width)

        # Apply temporal transformation
        x = x.view(batch_size, num_frames, -1)  # (batch_size, num_frames, num_channels*height*width)
        temporal_pos_enc = self.temporal_pos_enc.unsqueeze(1).repeat(1, num_frames, 1)  # (num_frames, num_frames, num_channels)
        x = x + temporal_pos_enc
        x = x.permute(1, 0, 2)  # (num_frames, batch_size, num_channels*height*width)
        x = self.temporal_transformer(x)
        x = x.permute(1, 0, 2)  # (batch_size, num_frames, num_channels*height*width)

        # Pixel prediction
        x = x.view(batch_size * num_frames, -1)  # (batch_size*num_frames, num_channels*height*width)
        x = self.fc(x)
        x = x.view(batch_size, num_frames, 3, height, width)  # (batch_size, num_frames, 3, height, width)

        return x
