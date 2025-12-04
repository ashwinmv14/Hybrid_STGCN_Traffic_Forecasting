import torch
import torch.nn as nn

class TemporalConv(nn.Module):
    """
    Simple temporal convolution used in the hybrid STGCN blocks.

    Input:  X [B, C, N, T]
    Output: same shape [B, C_out, N, T]
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Args:
            in_channels: input channels
            out_channels: output channels  
            kernel_size: temporal kernel size (default 3)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)   # keep T dimension same
        )
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, C, N, T]
        return self.relu(self.conv(X))
