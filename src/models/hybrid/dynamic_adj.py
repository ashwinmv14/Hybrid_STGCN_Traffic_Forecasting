import torch
import torch.nn.functional as F


class DynamicAdj(torch.nn.Module):
    """
    Memory-efficient dynamic adjacency computation.
    Instead of creating full [B,N,N] matrices, we only update edges.
    """
    def __init__(self, alpha=0.005):
        super().__init__()
        self.alpha = alpha

    def forward(self, A_base, edge_index, edge_gates):
        """
        A_base: [N,N]
        edge_gates: [B,E]
        return: [B,N,N]
        
        Memory-efficient version:
        - Only materializes sparse updates
        - Vectorized batch operations
        """
        if edge_gates is None:
            return A_base.unsqueeze(0)  # [1,N,N]

        B, E = edge_gates.shape
        N = A_base.shape[0]
        
        # Method 1: Sparse update (most memory efficient)
        # Start with base adjacency repeated for batch
        A = A_base.unsqueeze(0).expand(B, -1, -1).clone()  # [B,N,N]
        
        # Vectorized edge updates (no loop over batch!)
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]
        
        # Compute edge weights: [B,E]
        edge_weights = self.alpha * torch.sigmoid(edge_gates)  # [B,E]
        
        # Update all batches at once using advanced indexing
        batch_idx = torch.arange(B, device=A.device)[:, None]  # [B,1]
        
        # A[batch_idx, src, dst] += edge_weights
        # Expand indices for batch dimension
        A[batch_idx, src, dst] = A[batch_idx, src, dst] + edge_weights
        
        return A


class DynamicAdjSparse(torch.nn.Module):
    """
    Even more memory-efficient version using sparse tensors.
    Only use this if you're still running out of memory.
    """
    def __init__(self, alpha=0.005):
        super().__init__()
        self.alpha = alpha

    def forward(self, A_base, edge_index, edge_gates):
        """
        A_base: [N,N]
        edge_gates: [B,E]
        return: [B,N,N] (but computed more efficiently)
        """
        if edge_gates is None:
            return A_base.unsqueeze(0)

        B, E = edge_gates.shape
        N = A_base.shape[0]
        
        # Compute edge weight deltas
        edge_gates = torch.clamp(edge_gates, -3, 3)
        edge_weights = self.alpha * torch.sigmoid(edge_gates)

        # Create batch indices
        batch_indices = torch.arange(B, device=edge_gates.device).view(-1, 1).expand(-1, E)  # [B,E]
        src_indices = edge_index[0].unsqueeze(0).expand(B, -1)  # [B,E]
        dst_indices = edge_index[1].unsqueeze(0).expand(B, -1)  # [B,E]
        
        # Flatten for sparse tensor creation
        batch_flat = batch_indices.flatten()  # [B*E]
        src_flat = src_indices.flatten()      # [B*E]
        dst_flat = dst_indices.flatten()      # [B*E]
        weights_flat = edge_weights.flatten() # [B*E]
        
        # Create sparse delta matrix
        indices = torch.stack([batch_flat, src_flat, dst_flat], dim=0)  # [3, B*E]
        delta_sparse = torch.sparse_coo_tensor(
            indices, weights_flat, (B, N, N), device=edge_gates.device
        )
        
        # Convert to dense and add to base
        delta_dense = delta_sparse.to_dense()
        A_base_batch = A_base.unsqueeze(0).expand(B, -1, -1)
        
        return A_base_batch + delta_dense