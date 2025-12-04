"""
CRITICAL DIAGNOSTIC TEST
========================

Before running full ablation, we need to answer:
"Does your STGCN-only match the paper's 18.99 MAE?"

This quick test (2-3 hours) will tell you if:
A) Your components help or hurt
B) Your implementation matches the paper

Run this FIRST before the 20-hour ablation!
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath("./"))

from src.models.hybrid.hybrid_model import HybridModel
from src.base.hybrid_engine import HybridEngine
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.logging import get_logger
from src.utils.graph_algo import calculate_sym_adj
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def load_edge_index(adj):
    row, col = np.nonzero(adj)
    return torch.tensor([row, col], dtype=torch.long)


def main():
    print("="*80)
    print("CRITICAL DIAGNOSTIC: STGCN Baseline Test")
    print("="*80)
    print("This will answer: Does your STGCN match the paper's 18.99 MAE?")
    print("Estimated time: 2-3 hours")
    print("="*80)
    
    device = torch.device("cuda:0")
    
    # Load dataset
    class DummyArgs:
        dataset = "CA"
        years = "2019"
        seq_len = 12
        horizon = 12
        input_dim = 3
        bs = 16  # ← INCREASED! You have memory for this
    
    args = DummyArgs()
    data_dir, adj_path, node_num = get_dataset_info(args.dataset)
    
    class DummyLogger:
        def info(self, msg):
            print(f"  {msg}")
    
    dataloader, _ = load_dataset(data_dir, args, logger=DummyLogger())
    
    # Load adjacency
    adj = load_adj_from_numpy(adj_path).astype(np.float32)
    A_norm = calculate_sym_adj(adj)
    A_norm = A_norm.todense().astype(np.float32)
    A_static = torch.tensor(A_norm, dtype=torch.float32, device=device)
    edge_index = load_edge_index(adj).to(device)
    
    # STGCN only (no components)
    model = HybridModel(
        node_num=node_num,
        input_dim=3,
        horizon=12,
        A_static=A_static,
        edge_index=edge_index,
        use_film=False,      # ← NO components
        use_alignn=False,    # ← NO components
        use_dynamic_adj=False,  # ← NO components
        film_hidden=64,
        alignn_hidden=64,
        alpha_dyn=0.005
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    
    log_dir = "experiments/diagnostic/stgcn_baseline"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, "diagnostic", "test.log")
    
    engine = HybridEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        loss_fn=torch.nn.L1Loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=5.0,
        max_epochs=50,
        patience=15,
        log_dir=log_dir,
        logger=logger
    )
    
    # Train
    print("\nTraining STGCN baseline...")
    engine.train()
    
    best_val_mae = engine.best_val
    
    # Analysis
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print("="*80)
    print(f"\nYour STGCN-only val MAE: {best_val_mae:.4f}")
    print(f"Paper's STGCN MAE:       18.99")
    print(f"Your Full model MAE:     19.63")
    
    diff_from_paper = best_val_mae - 18.99
    diff_from_full = best_val_mae - 19.63
    
    print(f"\nDifference from paper:  {diff_from_paper:+.4f}")
    print(f"Difference from full:   {diff_from_full:+.4f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if abs(diff_from_paper) < 0.5:
        print("\n✅ Your STGCN matches the paper (~18.99 MAE)")
        print("\n⚠️  BUT your full model (19.63) is WORSE!")
        print("    → Your components (FiLM/ALIGNN/DynAdj) are HURTING performance")
        print("\n❌ DO NOT RUN ABLATION with current components")
        print("\nRECOMMENDATIONS:")
        print("1. Increase batch size to 16 or 32")
        print("2. Lower learning rate to 0.0005")
        print("3. Tune component hyperparameters")
        print("4. Consider removing components that don't help")
        
    elif best_val_mae > 20.0:
        print("\n⚠️  Your STGCN is WORSE than paper (>20 vs 18.99)")
        print("    → Your STGCN implementation differs from paper")
        print("\n✅ Your full model (19.63) is BETTER than your baseline")
        print("    → Your components ARE helping!")
        print("\nRECOMMENDATIONS:")
        print("1. Fix your STGCN baseline to match paper")
        print("2. Check adjacency normalization (scalap vs symadj)")
        print("3. Increase batch size to 32")
        print("4. Then re-train full model with fixed baseline")
        
    else:
        print("\n⚠️  Results are ambiguous")
        print(f"    Your STGCN: {best_val_mae:.4f}")
        print(f"    Paper STGCN: 18.99")
        print(f"    Your Full: 19.63")
        print("\nRECOMMENDATIONS:")
        print("1. Increase batch size significantly (to 32)")
        print("2. This might close the gap")
        print("3. Then run full ablation")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()