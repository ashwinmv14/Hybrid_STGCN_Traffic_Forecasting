"""
True STGCN backbone blocks extracted from baseline src/models/stgcn.py
Used by HybridModel without modifying original STGCN.

Includes:
- Align
- CausalConv2d
- TemporalConvLayer
- ChebGraphConv
- GraphConvLayer
- STConvBlock
- OutputBlock
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ---------------------------------------------------------
# Align Layer
# ---------------------------------------------------------
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            B, _, T, N = x.shape
            pad = torch.zeros([B, self.c_out - self.c_in, T, N], device=x.device)
            x = torch.cat([x, pad], dim=1)
        return x


# ---------------------------------------------------------
# CausalConv2d (temporal)
# ---------------------------------------------------------
class CausalConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 enable_padding=False,
                 dilation=1,
                 groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)

        if enable_padding:
            self.__padding = [(kernel_size[i] - 1) * dilation[i] for i in range(2)]
        else:
            self.__padding = 0

        self.left_padding = nn.modules.utils._pair(self.__padding)
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=0,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)

    def forward(self, x):
        if self.__padding != 0:
            x = F.pad(x,
                      (self.left_padding[1], 0,
                       self.left_padding[0], 0))
        return super().forward(x)


# ---------------------------------------------------------
# TemporalConvLayer (gated)
# ---------------------------------------------------------
class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, node_num):
        super().__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.node_num = node_num

        self.align = Align(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        self.causal_conv = CausalConv2d(
            in_channels=c_in,
            out_channels=2 * c_out,
            kernel_size=(Kt, 1),
            enable_padding=False,
            dilation=1
        )

    def forward(self, x):
        # Align input to c_out
        x_in = self.align(x)[:, :, self.Kt - 1:, :]  # causal alignment

        x_conv = self.causal_conv(x)
        p = x_conv[:, :self.c_out, :, :]
        q = x_conv[:, -self.c_out:, :, :]

        return (p + x_in) * self.sigmoid(q)


# ---------------------------------------------------------
# Chebyshev Graph Convolution
# ---------------------------------------------------------
class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso

        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x:   [B, C_in, T, N]
        gso: [N, N] (graph)
        return: [B, C_out, T, N]
        """
        B, C_in, T, N = x.shape

        # Normalize adjacency on-the-fly (keeps eigenvalues <= 1)
        A = self.gso
        deg = torch.sum(A, dim=1) + 1e-6
        deg_inv_sqrt = deg.pow(-0.5)
        A_norm = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)  # [N,N]

        # Move channels to last so adjacency acts on node dimension
        x0 = x.permute(0, 2, 3, 1)        # [B, T, N, C_in]

        T_k = [x0]

        if self.Ks > 1:
            # T_1 = A_norm * T_0
            x1 = torch.einsum("ij,btjc->btic", A_norm, x0)
            x1 = torch.clamp(x1, -5, 5)              # stability clamp
            T_k.append(x1)

            for k in range(2, self.Ks):
                xk = 2 * torch.einsum("ij,btjc->btic", A_norm, T_k[-1]) - T_k[-2]
                xk = torch.clamp(xk, -5, 5)          # stability clamp
                T_k.append(xk)

        # Stack Chebyshev polynomials
        x_stack = torch.stack(T_k, dim=2)            # [B, T, Ks, N, C_in]

        # Apply learned weights over (Ks, C_in)
        x_out = torch.einsum("btkni,kic->btnc", x_stack, self.weight)  # [B,T,N,C_out]
        x_out = x_out + self.bias                   # broadcast over C_out

        # Back to [B, C_out, T, N]
        return x_out.permute(0, 3, 1, 2)


# ---------------------------------------------------------
# GraphConvLayer
# ---------------------------------------------------------
class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super().__init__()
        self.align = Align(c_in, c_out)
        self.cheb = ChebGraphConv(c_out, c_out, Ks, gso)

    def forward(self, x):
        # x: [B, C_in, T, N]
        x_in = self.align(x)          # [B, C_out, T, N]
        x_gc = self.cheb(x_in)        # [B, C_out, T, N]
        return x_gc + x_in


# ---------------------------------------------------------
# STConvBlock
# ---------------------------------------------------------
class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, node_num, last_c, channel_list, gso, dropout):
        super().__init__()
        c_in = last_c
        c1, c2, c3 = channel_list

        self.tmp1 = TemporalConvLayer(Kt, c_in, c1, node_num)
        self.gc = GraphConvLayer(c1, c2, Ks, gso)
        self.tmp2 = TemporalConvLayer(Kt, c2, c3, node_num)

        self.ln = nn.LayerNorm([node_num, c3])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tmp1(x)
        x = self.gc(x)
        x = self.relu(x)
        x = self.tmp2(x)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.drop(x)


# ---------------------------------------------------------
# OutputBlock
# ---------------------------------------------------------
class OutputBlock(nn.Module):
    def __init__(self, Ko, last_c, channel_list, end_c, node_num, dropout):
        super().__init__()
        c1, c2 = channel_list

        self.tmp1 = TemporalConvLayer(Ko, last_c, c1, node_num)
        self.fc1 = nn.Linear(c1, c2)
        self.fc2 = nn.Linear(c2, end_c)

        self.ln = nn.LayerNorm([node_num, c1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tmp1(x)
        x = self.ln(x.permute(0, 2, 3, 1))
        x = self.relu(self.fc1(x))
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x
