import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM
from .alignn import ALIGNN
from .dynamic_adj import DynamicAdj
from .stgcn_blocks import STGCNBackbone, STGCNHead


class HybridModel(nn.Module):
    """
    STGCN backbone → FiLM → ALIGNN gating → DynamicAdj → STGCN head

    Input : X [B,N,C,T]
    Output: Y [B,H,N,1]   (STGCN-compatible)
    """

    def __init__(self, node_num, input_dim, horizon,
                 A_static, edge_index,
                 use_film=True, use_alignn=True, use_dynamic_adj=True,
                 film_hidden=64, alignn_hidden=64, alpha_dyn=0.005):

        super().__init__()

        self.node_num = node_num
        self.input_dim = input_dim
        self.horizon = horizon

        self.A_static = A_static              # torch [N,N]
        self.edge_index = edge_index          # [2,E]

        self.use_film = use_film
        self.use_alignn = use_alignn
        self.use_dynamic_adj = use_dynamic_adj

        # STGCN backbone (2 layers)
        self.block1 = STGCNBackbone(node_num, input_dim, hidden_dim=64)
        self.block2 = STGCNBackbone(node_num, 64, hidden_dim=64)

        # FiLM temporal conditioner
        self.film = FiLM(hidden_dim=film_hidden)

        # ALIGNN edge gating
        self.alignn = ALIGNN(node_feature_dim=64,
                             hidden_dim=alignn_hidden,
                             out_dim=1)

        # Dynamic adjacency
        self.A_dyn = DynamicAdj(alpha=alpha_dyn)

        # Final STGCN head
        self.head = STGCNHead(
            node_num=node_num,
            in_dim=64,
            hidden_dim=64,
            horizon=horizon
        )

    def forward(self, X, t_index):
        """
        X       : [B,N,C,T]
        t_index : [B,T]   (TOD index)
        
        Returns : [B,H,N,1] (STGCN/LargeST compatible format)
        """

        B, N, C, T = X.shape
        A = self.A_static

        # --- Backbone STGCN ---
        H = self.block1(X, A)    # [B,N,64,T]
        H = self.block2(H, A)

        # --- FiLM ---
        if self.use_film:
            gamma, beta = self.film(t_index, T)
            H = gamma * H + beta

        # --- ALIGNN (edge gating) ---
        if self.use_alignn:
            # final time step features
            H_t = H[:, :, :, -1]       # [B,N,64]
            gate = self.alignn(H_t, self.edge_index)   # [B,E]
        else:
            gate = None

        # --- Dynamic A_t ---
        if self.use_dynamic_adj:
            A_t = self.A_dyn(A, self.edge_index, gate)   # [B,N,N]
        else:
            A_t = A.unsqueeze(0).repeat(B,1,1)

        # --- Final STGCN head ---
        Y = self.head(H, A_t)    # [B,N,H]
        
        return Y