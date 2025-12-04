import torch
import torch.nn as nn

from .stgcn_true_blocks import STConvBlock, OutputBlock
from .film import FiLM
from .alignn import ALIGNN
from .dynamic_adj import DynamicAdj


class HybridModel(nn.Module):

    def __init__(
        self,
        node_num,
        input_dim,
        horizon,
        A_static,
        edge_index,
        use_film=True,
        use_alignn=True,
        use_dynamic_adj=True,
        film_hidden=64,
        alignn_hidden=64,
        alpha_dyn=0.005
    ):
        super().__init__()

        self.node_num = node_num
        self.input_dim = input_dim
        self.horizon = horizon

        self.A_static = A_static
        self.edge_index = edge_index

        self.use_film = use_film
        self.use_alignn = use_alignn
        self.use_dynamic_adj = use_dynamic_adj

        # --- STGCN Backbone ---
        self.block1 = STConvBlock(
            Kt=3, Ks=3,
            node_num=node_num,
            last_c=input_dim,
            channel_list=[64, 64, 64],
            gso=A_static,
            dropout=0.1
        )

        self.block2 = STConvBlock(
            Kt=3, Ks=3,
            node_num=node_num,
            last_c=64,
            channel_list=[64, 64, 64],
            gso=A_static,
            dropout=0.1
        )

        # --- FiLM ---
        self.film = FiLM(hidden_dim=film_hidden)

        # --- ALIGNN ---
        self.alignn = ALIGNN(
            node_feature_dim=64,
            hidden_dim=alignn_hidden,
            out_dim=1
        )

        # --- Dynamic adjacency ---
        self.A_dyn = DynamicAdj(alpha=alpha_dyn)

        # --- STGCN Output head ---
        self.head = OutputBlock(
            Ko=3,
            last_c=64,
            channel_list=[64, 64],
            end_c=horizon,
            node_num=node_num,
            dropout=0.1
        )


    def forward(self, X, t_index):
        """
        X : [B, C, T, N]
        t_index : [B, T]
        """
        B, C, T, N = X.shape

        # --- Backbone ---
        H = self.block1(X)     # [B,64,T1,N]
        H = self.block2(H)     # [B,64,T2,N]
        B, Cmid, T2, N = H.shape

        # Reorder for FiLM / ALIGNN
        H = H.permute(0, 3, 1, 2)      # [B,N,64,T2]

        # --- FiLM ---
        if self.use_film:
            t_eff = t_index[:, -T2:]               # match backbone length
            gamma, beta = self.film(t_eff, T2)     # both [B,1,1,T2]
            H = gamma * H + beta                   # broadcast

        # --- ALIGNN gating ---
        if self.use_alignn:
            H_t = H[:, :, :, -1]                  # [B,N,64]
            gate = self.alignn(H_t, self.edge_index)  # [B,E]
            gate = torch.clamp(gate, -6, 6)       # stability
        else:
            gate = None

        # --- Dynamic adjacency ---
        if self.use_dynamic_adj:
            A_t = self.A_dyn(self.A_static, self.edge_index, gate)  # [B,N,N]
        else:
            A_t = self.A_static.unsqueeze(0).expand(B, -1, -1)

        # --- STGCN output head ---
        H_head = H.permute(0, 2, 3, 1)             # [B,64,T2,N]
        Y_raw = self.head(H_head)                  # [B,H,Tout,N]

        # Final output: last time step
        Y = Y_raw[:, :, -1, :].unsqueeze(-1)       # [B,H,N,1]

        return Y
