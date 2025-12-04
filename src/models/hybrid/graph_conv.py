import torch
import torch.nn as nn

class GraphConv(nn.Module):
    """
    Simple GCN-style spatial convolution.

    X: [B, N, C_in]
    A: [N, N] or [B, N, N]

    Returns: [B, N, C_out]
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: [B, N, C_in]
        # A: [N, N] or [B, N, N]
        B, N, C = X.shape

        if A.dim() == 2:
            # expand static adjacency over batch
            A = A.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        elif A.dim() == 3:
            # assume already [B, N, N]
            assert A.shape[0] == B and A.shape[1] == N and A.shape[2] == N, \
                f"Adjacency shape {A.shape} incompatible with X {X.shape}"
        else:
            raise ValueError(f"Adjacency must be 2D or 3D, got shape {A.shape}")

        # spatial aggregation: [B, N, N] x [B, N, C] -> [B, N, C]
        AX = torch.bmm(A, X)
        return self.linear(AX)