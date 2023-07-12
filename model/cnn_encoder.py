
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
        )

        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=(2, 3, 4))  # Global average pooling
        x = self.fc(x)
        return x
    
    
class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_channels):
        super(CNNDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(input_dim, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, output_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)