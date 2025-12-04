"""
Run STGCN baseline (no hybrid components) for 100 epochs
Fair comparison baseline for your ALIGNN results
"""
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.models.hybrid.hybrid_model import HybridModel
from src.base.hybrid_engine import HybridEngine
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.logging import get_logger
from src.utils.graph_algo import calculate_sym_adj
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def load_edge_index(adj):
    """Convert [N,N] adjacency to edge_index=[2,E]"""
    row, col = np.nonzero(adj)
    return torch.tensor([row, col], dtype=torch.long)


def main():
    print("="*80)
    print("RUNNING STGCN BASELINE (NO COMPONENTS) FOR 100 EPOCHS")
    print("="*80)
    
    # Configuration
    dataset = "CA"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = {
        'use_film': False,
        'use_alignn': False,
        'use_dynamic_adj': False,
        'film_hidden': 64,
        'alignn_hidden': 64,
        'alpha_dyn': 0.005,
    }
    
    # Training params
    lr = 0.001
    clip = 5.0
    max_epochs = 100
    patience = 30
    bs = 64
    
    print(f"Device: {device}")
    print(f"Max epochs: {max_epochs}")
    print(f"Configuration: Pure STGCN (no hybrid components)")
    print()
    
    # Args setup
    class Args:
        pass
    
    args = Args()
    args.dataset = dataset
    args.years = "2019"
    args.seq_len = 12
    args.horizon = 12
    args.input_dim = 3
    args.bs = bs
    
    # Setup paths
    data_dir, adj_path, node_num = get_dataset_info(dataset)
    log_dir = f"experiments/stgcn_baseline_100ep/{dataset}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger
    logger = get_logger(log_dir, f"STGCN_Baseline_100ep_{dataset}", "training.log")
    logger.info("="*80)
    logger.info("STGCN BASELINE - 100 Epochs Training")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset} ({node_num} nodes)")
    logger.info(f"Config: Pure STGCN (no FiLM, no ALIGNN, no Dynamic Adj)")
    logger.info(f"Max epochs: {max_epochs}, Patience: {patience}")
    
    # Load adjacency
    adj = load_adj_from_numpy(adj_path).astype(np.float32)
    A_norm = calculate_sym_adj(adj)
    A_norm = A_norm.todense().astype(np.float32)
    A_static = torch.tensor(A_norm, device=device)
    edge_index = load_edge_index(adj).to(device)
    
    logger.info(f"Adjacency: {A_static.shape}, Edges: {edge_index.shape[1]}")
    
    # Load data
    dataloader, scaler = load_dataset(data_dir, args, logger)
    
    # Create model - PURE STGCN (all flags False)
    model = HybridModel(
        node_num=node_num,
        input_dim=args.input_dim,
        horizon=args.horizon,
        A_static=A_static,
        edge_index=edge_index,
        use_film=False,          # ← No FiLM
        use_alignn=False,        # ← No ALIGNN
        use_dynamic_adj=False,   # ← No Dynamic Adj
        film_hidden=config["film_hidden"],
        alignn_hidden=config["alignn_hidden"],
        alpha_dyn=config["alpha_dyn"]
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Optimizer & scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Training engine
    engine = HybridEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        loss_fn=torch.nn.L1Loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=clip,
        max_epochs=max_epochs,
        patience=patience,
        log_dir=log_dir,
        logger=logger
    )
    
    # Train
    logger.info("\nStarting training...")
    start_time = datetime.now()
    
    engine.train()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    logger.info("="*80)
    logger.info(f"Training completed in {duration:.2f} hours")
    logger.info(f"Best model saved to: {log_dir}/best_model.pt")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print(f"✅ STGCN Baseline Training Completed!")
    print(f"Duration: {duration:.2f} hours")
    print(f"Results saved to: {log_dir}")
    print("="*80)
    print("\nNext steps:")
    print("1. Wait for training to finish")
    print("2. Run evaluation: python eval_stgcn_baseline_100ep.py")
    print("3. Compare with your STGCN+ALIGNN results")
    print("="*80)


if __name__ == "__main__":
    main()