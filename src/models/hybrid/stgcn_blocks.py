import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import TemporalConv
from .graph_conv import GraphConv


class STGCNBackbone(nn.Module):
    """
    STGCN backbone block:
        TCN → GraphConv (per time step) → TCN

    Input:  X [B, N, C_in, T]
    Output: H [B, N, C_hidden, T]
    """

    def __init__(self, node_num, in_dim, hidden_dim, kt=3):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim

        # First temporal conv (along time)
        self.tcn1 = TemporalConv(in_channels=in_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=kt)

        # Spatial graph conv on each time slice
        self.gc = GraphConv(in_channels=hidden_dim,
                            out_channels=hidden_dim)

        # Second temporal conv
        self.tcn2 = TemporalConv(in_channels=hidden_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=kt)

    def forward(self, X, A):
        """
        X: [B, N, C_in, T]
        A: [N, N] or [B, N, N]
        """
        B, N, C, T = X.shape
        assert N == self.node_num, f"Expected {self.node_num} nodes, got {N}"

        # ---- TCN 1 ----
        # TemporalConv expects [B, C, N, T]
        H = X.permute(0, 2, 1, 3).contiguous()   # [B, C_in, N, T]
        H = self.tcn1(H)                         # [B, hidden, N, T]

        # ---- GraphConv on each time step ----
        # Arrange as [B, T, N, hidden]
        H = H.permute(0, 3, 2, 1).contiguous()   # [B, T, N, hidden]

        out_t_list = []
        for t in range(T):
            # Slice one time step: [B, N, hidden]
            Xt = H[:, t, :, :]          # [B, N, hidden]

            # Apply graph conv
            Xt_gc = self.gc(Xt, A)      # [B, N, hidden]

            # Collect back with time dim at the end
            out_t_list.append(Xt_gc.unsqueeze(-1))  # [B, N, hidden, 1]

        H_spatial = torch.cat(out_t_list, dim=-1)   # [B, N, hidden, T]

        # ---- TCN 2 ----
        H_spatial = H_spatial.permute(0, 2, 1, 3).contiguous()  # [B, hidden, N, T]
        H_out = self.tcn2(H_spatial)                            # [B, hidden, N, T]

        # Back to [B, N, hidden, T]
        H_out = H_out.permute(0, 2, 1, 3).contiguous()          # [B, N, hidden, T]

        return H_out


class STGCNHead(nn.Module):
    """
    STGCN forecast head:
        backbone (TCN–GC–TCN) → 1×1 conv over channels → take last time step

    Input:  X [B, N, C_in, T]
    Output: Y [B, horizon, N, 1]
    """

    def __init__(self, node_num, in_dim, hidden_dim, horizon, kt=3):
        super().__init__()
        self.node_num = node_num
        self.horizon = horizon

        # Backbone producing hidden representation
        self.backbone = STGCNBackbone(
            node_num=node_num,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            kt=kt,
        )

        # 1×1 conv over channel dimension to produce H horizons
        # Input to this conv: [B, hidden_dim, N, T]
        self.proj = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=horizon,
            kernel_size=(1, 1)
        )

    def forward(self, X, A):
        """
        X: [B, N, C_in, T]
        A: [N, N] or [B, N, N]
        """
        # First run backbone: [B, N, hidden, T]
        H = self.backbone(X, A)              # [B, N, hidden, T]

        # Prepare for 1×1 conv: [B, hidden, N, T]
        H = H.permute(0, 2, 1, 3).contiguous()    # [B, hidden, N, T]

        # Project to horizon channels
        Y_full = self.proj(H)                # [B, horizon, N, T]

        # Use last time step for prediction
        Y_last = Y_full[:, :, :, -1]         # [B, horizon, N]

        # Match LargeST / STGCN format: [B, H, N, 1]
        return Y_last.unsqueeze(-1)
