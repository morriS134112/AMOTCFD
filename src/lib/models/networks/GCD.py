import torch
from torch import nn


class GCD(nn.Module):
    def __init__(self, in_channels, context_dim):
        super(GCD, self).__init__()
        # Global context extraction (1x1 convolution as Wk)
        self.global_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Decoupling transformations for detection and ReID (with layer normalization)
        self.Wd1 = nn.Linear(in_channels, context_dim)
        self.Wd2 = nn.Linear(context_dim, in_channels)
        self.Wr1 = nn.Linear(in_channels, context_dim)
        self.Wr2 = nn.Linear(context_dim, in_channels)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm([context_dim])

    def forward(self, x):
        # First phase: Calculate global context vector z
        N, C, H, W = x.size()
        context_map = self.global_conv(x).view(N, 1, H * W)  # Shape: (N, 1, H*W)
        context_map = self.softmax(context_map)  # Softmax along the spatial dimension
        context_vector = torch.bmm(context_map, x.view(N, C, H * W).permute(0, 2, 1))  # Shape: (N, 1, C)
        z = context_vector.squeeze(1)  # Shape: (N, C)

        # Second phase: Decouple z into detection-specific and ReID-specific features
        z_d = self.ln(self.relu(self.Wd1(z)))
        z_r = self.ln(self.relu(self.Wr1(z)))

        d = x + self.Wd2(z_d).view(N, C, 1, 1)
        r = x + self.Wr2(z_r).view(N, C, 1, 1)

        # Optional: Bidirectional feature passing (to balance decoupling and common information)
        d = d + 0.1 * r
        r = r + 0.1 * d

        return d, r