import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # outputs gamma, beta
        )

    def forward(self, t_index, T):
        """
        t_index: [B, T]   (float TOD)
        returns:
            gamma, beta: [B, 1, 1, T]
        """
        B, T_ = t_index.shape
        assert T_ == T

        # [B, T, 1]
        t = t_index.unsqueeze(-1)

        # [B, T, 2]
        gb = self.mlp(t)

        # raw FiLM outputs
        gamma_raw = gb[:, :, 0]     # [B, T]
        beta_raw  = gb[:, :, 1]     # [B, T]

        # ✔ clamp with tanh for stability
        gamma_raw = torch.tanh(gamma_raw)
        beta_raw  = torch.tanh(beta_raw)

        # reshape
        gamma = gamma_raw.unsqueeze(1).unsqueeze(1)   # [B,1,1,T]
        beta  = beta_raw.unsqueeze(1).unsqueeze(1)    # [B,1,1,T]

        return gamma, beta
