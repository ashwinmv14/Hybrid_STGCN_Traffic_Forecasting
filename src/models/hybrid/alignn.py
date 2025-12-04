import torch
import torch.nn as nn


class ALIGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim=1, num_layers=2):
        super().__init__()

        layers = []
        in_dim = node_feature_dim * 2
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        self.chunk = 4096  # for safe processing

    def forward(self, X, edge_index):
        """
        X: [B,N,F]
        edge_index: [2,E]
        return: [B,E]
        """
        B, N, F = X.shape
        src = edge_index[0]  # [E]
        dst = edge_index[1]

        E = src.shape[0]
        out = []

        for i in range(0, E, self.chunk):
            s = src[i:i+self.chunk]  # [c]
            d = dst[i:i+self.chunk]

            xs = X[:, s, :]  # [B,c,F]
            xd = X[:, d, :]
            feat = torch.cat([xs, xd], dim=-1)  # [B,c,2F]

            g = self.mlp(feat)  # [B,c,1]
            g = torch.clamp(g, -5, 5)
            g = g.squeeze(-1)
            out.append(g)

        return torch.cat(out, dim=1)  # [B,E]
