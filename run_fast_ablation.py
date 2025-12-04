"""
OPTIMIZED Ablation Study - Faster Training with Bug Fixes
==========================================================

Optimizations:
- Reduced to 30 epochs (from 50) with patience=10
- Fixed MAPE infinity bug
- Better batch size utilization
- Estimated time: 18-24 hours total

Models trained & evaluated:
1. STGCN only (baseline)
2. STGCN + FiLM
3. STGCN + ALIGNN  
4. STGCN + DynAdj
5. Full Model

Usage:
    python run_fast_ablation.py
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

sys.path.append(os.path.abspath("./"))

from src.models.hybrid.hybrid_model import HybridModel
from src.base.hybrid_engine import HybridEngine
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.logging import get_logger
from src.utils.graph_algo import calculate_sym_adj
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def load_edge_index(adj):
    """Convert adjacency to edge_index"""
    row, col = np.nonzero(adj)
    return torch.tensor([row, col], dtype=torch.long)


def masked_mae(preds, labels, null_val=np.nan):
    """MAE with masking"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    """RMSE with masking"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds, labels, null_val=np.nan, epsilon=1.0):
    """
    MAPE with masking - FIXED VERSION
    
    Added epsilon to avoid division by zero/very small numbers
    This prevents inf/nan in MAPE calculations
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # CRITICAL FIX: Add epsilon to denominator to avoid division by near-zero values
    # epsilon=1.0 is appropriate for traffic flow (typically 0-1000+ range)
    loss = torch.abs(preds - labels) / (torch.abs(labels) + epsilon)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def evaluate_model(model, test_loader, device, scaler, logger):
    """Evaluate model on test set with fixed MAPE"""
    
    logger.info("Starting evaluation on test set...")
    model.eval()
    
    horizon_mae = np.zeros(12)
    horizon_rmse = np.zeros(12)
    horizon_mape = np.zeros(12)
    count = 0
    
    with torch.no_grad():
        for x, y in test_loader.get_iterator():
            
            x = torch.tensor(x, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            
            t_index = x[:, :, 0, 1]
            x = x.permute(0, 3, 1, 2)
            
            y_pred = model(x, t_index)
            
            # Denormalize
            y_pred_denorm = scaler.inverse_transform(y_pred)
            y_true_denorm = scaler.inverse_transform(y)
            
            # Compute metrics per horizon
            for h in range(12):
                pred_h = y_pred_denorm[:, h, :, :]
                true_h = y_true_denorm[:, h, :, :]
                
                # Use fixed MAPE with epsilon
                mae = masked_mae(pred_h, true_h, null_val=np.nan).item()
                rmse = masked_rmse(pred_h, true_h, null_val=np.nan).item()
                mape = masked_mape(pred_h, true_h, null_val=np.nan, epsilon=1.0).item()
                
                horizon_mae[h] += mae
                horizon_rmse[h] += rmse
                horizon_mape[h] += mape
            
            count += 1
    
    horizon_mae /= count
    horizon_rmse /= count
    horizon_mape /= count
    
    results = {
        "Horizon3_MAE": horizon_mae[2],
        "Horizon3_RMSE": horizon_rmse[2],
        "Horizon3_MAPE": horizon_mape[2] * 100,
        
        "Horizon6_MAE": horizon_mae[5],
        "Horizon6_RMSE": horizon_rmse[5],
        "Horizon6_MAPE": horizon_mape[5] * 100,
        
        "Horizon12_MAE": horizon_mae[11],
        "Horizon12_RMSE": horizon_rmse[11],
        "Horizon12_MAPE": horizon_mape[11] * 100,
        
        "Average_MAE": float(np.mean(horizon_mae)),
        "Average_RMSE": float(np.mean(horizon_rmse)),
        "Average_MAPE": float(np.mean(horizon_mape)) * 100,
    }
    
    logger.info("Evaluation complete!")
    logger.info(f"Average MAE:  {results['Average_MAE']:.2f}")
    logger.info(f"Average RMSE: {results['Average_RMSE']:.2f}")
    logger.info(f"Average MAPE: {results['Average_MAPE']:.2f}%")
    
    return results


def train_and_evaluate_model(model_name, config, device, data_dir, adj_path, node_num, dataloader, scaler):
    """Train and evaluate a single model variant"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    print(f"Config: {config}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load adjacency
        adj = load_adj_from_numpy(adj_path).astype(np.float32)
        A_norm = calculate_sym_adj(adj)
        A_norm = A_norm.todense().astype(np.float32)
        A_static = torch.tensor(A_norm, dtype=torch.float32, device=device)
        edge_index = load_edge_index(adj).to(device)
        
        # Create model
        model = HybridModel(
            node_num=node_num,
            input_dim=3,
            horizon=12,
            A_static=A_static,
            edge_index=edge_index,
            use_film=config['use_film'],
            use_alignn=config['use_alignn'],
            use_dynamic_adj=config['use_dynamic_adj'],
            film_hidden=config['film_hidden'],
            alignn_hidden=config['alignn_hidden'],
            alpha_dyn=config['alpha_dyn']
        ).to(device)
        
        optimizer = Adam(model.parameters(), lr=config['lr'])
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        
        log_dir = f"experiments/ablation/CA/{model_name}"
        os.makedirs(log_dir, exist_ok=True)
        logger = get_logger(log_dir, f"ablation_{model_name}", "training.log")
        
        logger.info(f"Starting training: {model_name}")
        logger.info(f"Config: {config}")
        
        engine = HybridEngine(
            device=device,
            model=model,
            dataloader=dataloader,
            loss_fn=torch.nn.L1Loss(),
            optimizer=optimizer,
            scheduler=scheduler,
            clip_grad_value=config['clip'],
            max_epochs=config['max_epochs'],
            patience=config['patience'],
            log_dir=log_dir,
            logger=logger
        )
        
        print(f"  Training {model_name} (max {config['max_epochs']} epochs, patience={config['patience']})...")
        engine.train()
        
        final_val_mae = engine.best_val
        logger.info(f"Training complete: {model_name}")
        logger.info(f"Best validation MAE: {final_val_mae:.4f}")
        
        print(f"  Evaluating {model_name} on test set...")
        
        test_results = evaluate_model(
            model=model,
            test_loader=dataloader['test_loader'],
            device=device,
            scaler=scaler,
            logger=logger
        )
        
        results_csv = os.path.join(log_dir, "test_results.csv")
        df = pd.DataFrame([test_results])
        df.to_csv(results_csv, index=False)
        logger.info(f"Test results saved to: {results_csv}")
        
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        print(f"✅ {model_name} completed successfully!")
        print(f"   Val MAE:  {final_val_mae:.4f}")
        print(f"   Test MAE: {test_results['Average_MAE']:.2f}")
        print(f"   Test RMSE: {test_results['Average_RMSE']:.2f}")
        print(f"   Test MAPE: {test_results['Average_MAPE']:.2f}%")
        print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'status': 'success',
            'val_mae': final_val_mae,
            'test_results': test_results,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"❌ {model_name} FAILED!")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'model_name': model_name
        }


def main():
    print("="*80)
    print("OPTIMIZED ABLATION STUDY - FASTER TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models: 5")
    print(f"Max epochs per model: 30 (reduced from 50)")
    print(f"Estimated time: 18-24 hours")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\nLoading dataset...")
    
    class DummyArgs:
        dataset = "CA"
        years = "2019"
        seq_len = 12
        horizon = 12
        input_dim = 3
        bs = 4
    
    args = DummyArgs()
    data_dir, adj_path, node_num = get_dataset_info(args.dataset)
    
    class DummyLogger:
        def info(self, msg):
            print(f"  {msg}")
    
    dataloader, scaler = load_dataset(data_dir, args, logger=DummyLogger())
    
    print(f"Dataset loaded: {node_num} nodes")
    
    # OPTIMIZED: Reduced epochs and patience
    configs = {
        "1_stgcn_only": {
            "use_film": False,
            "use_alignn": False,
            "use_dynamic_adj": False,
            "film_hidden": 64,
            "alignn_hidden": 64,
            "alpha_dyn": 0.005,
            "lr": 0.001,
            "clip": 5.0,
            "max_epochs": 30,      # Reduced from 50
            "patience": 10          # Reduced from 15
        },
        
        "2_stgcn_film": {
            "use_film": True,
            "use_alignn": False,
            "use_dynamic_adj": False,
            "film_hidden": 64,
            "alignn_hidden": 64,
            "alpha_dyn": 0.005,
            "lr": 0.001,
            "clip": 5.0,
            "max_epochs": 30,
            "patience": 10
        },
        
        "3_stgcn_alignn": {
            "use_film": False,
            "use_alignn": True,
            "use_dynamic_adj": False,
            "film_hidden": 64,
            "alignn_hidden": 64,
            "alpha_dyn": 0.005,
            "lr": 0.001,
            "clip": 5.0,
            "max_epochs": 30,
            "patience": 10
        },
        
        "4_stgcn_dyn": {
            "use_film": False,
            "use_alignn": False,
            "use_dynamic_adj": True,
            "film_hidden": 64,
            "alignn_hidden": 64,
            "alpha_dyn": 0.005,
            "lr": 0.001,
            "clip": 5.0,
            "max_epochs": 30,
            "patience": 10
        },
        
        "5_full_model": {
            "use_film": True,
            "use_alignn": True,
            "use_dynamic_adj": True,
            "film_hidden": 64,
            "alignn_hidden": 64,
            "alpha_dyn": 0.005,
            "lr": 0.001,
            "clip": 5.0,
            "max_epochs": 30,
            "patience": 10
        }
    }
    
    results = {}
    start_time = datetime.now()
    
    for i, (model_name, config) in enumerate(configs.items(), 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/5: {model_name}")
        print(f"{'='*80}")
        
        model_start = datetime.now()
        
        result = train_and_evaluate_model(
            model_name=model_name,
            config=config,
            device=device,
            data_dir=data_dir,
            adj_path=adj_path,
            node_num=node_num,
            dataloader=dataloader,
            scaler=scaler
        )
        
        model_end = datetime.now()
        model_duration = (model_end - model_start).total_seconds() / 3600
        
        result['duration_hours'] = model_duration
        results[model_name] = result
        
        print(f"\nModel {i}/5 completed in {model_duration:.2f} hours")
        
        with open("experiments/ablation/progress.txt", "a") as f:
            f.write(f"{datetime.now()}: {model_name} - {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"  Test MAE: {result['test_results']['Average_MAE']:.2f}\n")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"Total time: {total_duration:.2f} hours")
    
    # [Rest of summary code same as before...]
    
    # Save comprehensive results
    if any(r['status'] == 'success' for r in results.values()):
        summary_data = []
        for model_name, result in sorted(results.items()):
            if result['status'] == 'success':
                row = {'Model': model_name}
                row.update(result['test_results'])
                summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        summary_csv = "experiments/ablation/CA/ablation_results.csv"
        os.makedirs("experiments/ablation/CA", exist_ok=True)
        df_summary.to_csv(summary_csv, index=False)
        
        print(f"\n💾 Results saved to: {summary_csv}")
        print("\nResults:")
        print(df_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("🎉 DONE!")
    print("="*80)


if __name__ == "__main__":
    main()