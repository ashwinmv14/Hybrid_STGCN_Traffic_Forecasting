"""
Complete Ablation Study - Train + Evaluate All Models
======================================================

This script trains AND evaluates all 5 ablation models sequentially.
Estimated total time: 10-15 hours

Models trained & evaluated:
1. STGCN only (baseline)
2. STGCN + FiLM
3. STGCN + ALIGNN  
4. STGCN + DynAdj
5. Full Model (all components)

Usage:
    python run_complete_ablation.py

The script will:
- Train each model for 50 epochs (early stopping patience=15)
- Evaluate each model on test set after training
- Save all results to experiments/ablation/CA/
- Create summary CSV with all metrics
- Continue even if one model fails
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
    """MAE with masking - matches LargeST paper"""
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


def masked_mape(preds, labels, null_val=np.nan):
    """MAPE with masking"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def evaluate_model(model, test_loader, device, scaler, logger):
    """Evaluate model on test set - matches eval_horizons.py"""
    
    logger.info("Starting evaluation on test set...")
    model.eval()
    
    # Accumulators for 12 horizons
    horizon_mae = np.zeros(12)
    horizon_rmse = np.zeros(12)
    horizon_mape = np.zeros(12)
    count = 0
    
    with torch.no_grad():
        for x, y in test_loader.get_iterator():
            
            x = torch.tensor(x, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            
            # Extract TOD index
            t_index = x[:, :, 0, 1]
            
            # Reorder to [B, C, T, N]
            x = x.permute(0, 3, 1, 2)
            
            # Forward pass
            y_pred = model(x, t_index)
            
            # Denormalize
            y_pred_denorm = scaler.inverse_transform(y_pred)
            y_true_denorm = scaler.inverse_transform(y)
            
            # Compute metrics per horizon
            for h in range(12):
                pred_h = y_pred_denorm[:, h, :, :]
                true_h = y_true_denorm[:, h, :, :]
                
                mae = masked_mae(pred_h, true_h, null_val=np.nan).item()
                rmse = masked_rmse(pred_h, true_h, null_val=np.nan).item()
                mape = masked_mape(pred_h, true_h, null_val=np.nan).item()
                
                horizon_mae[h] += mae
                horizon_rmse[h] += rmse
                horizon_mape[h] += mape
            
            count += 1
    
    # Average over batches
    horizon_mae /= count
    horizon_rmse /= count
    horizon_mape /= count
    
    # Calculate averages
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
    
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)
    print(f"Config: {config}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ================================================================
        # TRAINING PHASE
        # ================================================================
        
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
        
        # Optimizer
        optimizer = Adam(model.parameters(), lr=config['lr'])
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        
        # Logger
        log_dir = f"experiments/ablation/CA/{model_name}"
        os.makedirs(log_dir, exist_ok=True)
        logger = get_logger(log_dir, f"ablation_{model_name}", "training.log")
        
        logger.info(f"Starting training: {model_name}")
        logger.info(f"Config: {config}")
        
        # Training engine
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
        
        # Train
        print(f"  Training {model_name}...")
        engine.train()
        
        # Get training result
        final_val_mae = engine.best_val
        logger.info(f"Training complete: {model_name}")
        logger.info(f"Best validation MAE: {final_val_mae:.4f}")
        
        # ================================================================
        # EVALUATION PHASE
        # ================================================================
        
        print(f"  Evaluating {model_name} on test set...")
        
        # Evaluate on test set
        test_results = evaluate_model(
            model=model,
            test_loader=dataloader['test_loader'],
            device=device,
            scaler=scaler,
            logger=logger
        )
        
        # Save test results
        results_csv = os.path.join(log_dir, "test_results.csv")
        df = pd.DataFrame([test_results])
        df.to_csv(results_csv, index=False)
        logger.info(f"Test results saved to: {results_csv}")
        
        # Cleanup
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
    print("COMPLETE ABLATION STUDY - TRAIN + EVALUATE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models: 5")
    print(f"Estimated time: 10-15 hours")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # ================================================================
    # LOAD DATASET (once, reuse for all models)
    # ================================================================
    
    print("\nLoading dataset...")
    
    class DummyArgs:
        dataset = "CA"
        years = "2019"
        seq_len = 12
        horizon = 12
        input_dim = 3
        bs = 4  # Your batch size
    
    args = DummyArgs()
    data_dir, adj_path, node_num = get_dataset_info(args.dataset)
    
    class DummyLogger:
        def info(self, msg):
            print(f"  {msg}")
    
    dataloader, scaler = load_dataset(data_dir, args, logger=DummyLogger())
    
    print(f"Dataset loaded: {node_num} nodes")
    print(f"Scaler mean: {scaler.mean}")
    print(f"Scaler std: {scaler.std}")
    
    # ================================================================
    # DEFINE ALL CONFIGURATIONS
    # ================================================================
    
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
            "max_epochs": 50,
            "patience": 15
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
            "max_epochs": 50,
            "patience": 15
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
            "max_epochs": 50,
            "patience": 15
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
            "max_epochs": 50,
            "patience": 15
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
            "max_epochs": 50,
            "patience": 15
        }
    }
    
    # ================================================================
    # TRAIN + EVALUATE ALL MODELS SEQUENTIALLY
    # ================================================================
    
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
        
        # Save intermediate progress
        with open("experiments/ablation/progress.txt", "a") as f:
            f.write(f"{datetime.now()}: {model_name} - {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"  Val MAE: {result['val_mae']:.4f}\n")
                f.write(f"  Test MAE: {result['test_results']['Average_MAE']:.2f}\n")
    
    # ================================================================
    # CREATE COMPREHENSIVE SUMMARY
    # ================================================================
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"Total time: {total_duration:.2f} hours")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ================================================================
    # RESULTS TABLE
    # ================================================================
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    successful = []
    failed = []
    
    print(f"\n{'Model':<20} {'Status':<10} {'Test MAE':<10} {'Test RMSE':<10} {'Test MAPE':<10}")
    print("-" * 70)
    
    for model_name, result in sorted_results:
        status = result['status']
        
        if status == 'success':
            test_mae = result['test_results']['Average_MAE']
            test_rmse = result['test_results']['Average_RMSE']
            test_mape = result['test_results']['Average_MAPE']
            print(f"{model_name:<20} ✅ Success   {test_mae:<10.2f} {test_rmse:<10.2f} {test_mape:<9.2f}%")
            successful.append((model_name, result['test_results']))
        else:
            print(f"{model_name:<20} ❌ Failed    N/A        N/A        N/A")
            failed.append(model_name)
    
    # ================================================================
    # SAVE COMPREHENSIVE RESULTS CSV
    # ================================================================
    
    if successful:
        # Create DataFrame with all results
        summary_data = []
        for model_name, test_results in successful:
            row = {'Model': model_name}
            row.update(test_results)
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_csv = "experiments/ablation/CA/ablation_results.csv"
        os.makedirs("experiments/ablation/CA", exist_ok=True)
        df_summary.to_csv(summary_csv, index=False)
        
        print(f"\n💾 Results saved to: {summary_csv}")
        
        # ================================================================
        # ABLATION ANALYSIS
        # ================================================================
        
        if len(successful) >= 2:
            print("\n" + "="*80)
            print("ABLATION ANALYSIS")
            print("="*80)
            
            results_dict = {name: test_results['Average_MAE'] for name, test_results in successful}
            
            if '1_stgcn_only' in results_dict:
                baseline_mae = results_dict['1_stgcn_only']
                print(f"\n📊 Component Contributions (vs STGCN baseline):")
                print(f"  Baseline MAE: {baseline_mae:.2f}")
                print()
                
                # Analyze each component
                if '2_stgcn_film' in results_dict:
                    film_mae = results_dict['2_stgcn_film']
                    film_delta = baseline_mae - film_mae
                    symbol = "✓" if film_delta > 0 else "✗"
                    print(f"  {symbol} +FiLM:     {film_mae:.2f} MAE (Δ {film_delta:+.2f})")
                
                if '3_stgcn_alignn' in results_dict:
                    alignn_mae = results_dict['3_stgcn_alignn']
                    alignn_delta = baseline_mae - alignn_mae
                    symbol = "✓" if alignn_delta > 0 else "✗"
                    print(f"  {symbol} +ALIGNN:   {alignn_mae:.2f} MAE (Δ {alignn_delta:+.2f})")
                
                if '4_stgcn_dyn' in results_dict:
                    dyn_mae = results_dict['4_stgcn_dyn']
                    dyn_delta = baseline_mae - dyn_mae
                    symbol = "✓" if dyn_delta > 0 else "✗"
                    print(f"  {symbol} +DynAdj:   {dyn_mae:.2f} MAE (Δ {dyn_delta:+.2f})")
                
                if '5_full_model' in results_dict:
                    full_mae = results_dict['5_full_model']
                    full_delta = baseline_mae - full_mae
                    symbol = "✓" if full_delta > 0 else "✗"
                    print(f"  {symbol} Full Model: {full_mae:.2f} MAE (Δ {full_delta:+.2f})")
                    print()
                    
                    if full_delta > 0:
                        print(f"✅ Full model improves by {full_delta:.2f} MAE ({(full_delta/baseline_mae)*100:.1f}%)")
                    else:
                        print(f"⚠️  Full model is worse by {abs(full_delta):.2f} MAE")
    
    # ================================================================
    # NEXT STEPS RECOMMENDATION
    # ================================================================
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if len(successful) == 5:
        print("\n✅ All models completed successfully!")
        print("\n1. Review results CSV:")
        print("   cat experiments/ablation/CA/ablation_results.csv")
        print("\n2. Analyze which components help:")
        print("   - Positive Δ = component helps")
        print("   - Negative Δ = component hurts")
        print("\n3. Next actions based on results:")
        print("   - If components help: Use for hyperparameter tuning")
        print("   - If components hurt: Remove them and retrain")
        print("   - If mixed: Keep only beneficial components")
        
    elif len(successful) >= 3:
        print(f"\n⚠️  {len(successful)}/5 models succeeded")
        print("\n1. Review successful models")
        print("2. Check logs for failed models")
        print("3. Proceed with available results")
    
    print("\n" + "="*80)
    print("🎉 ABLATION STUDY COMPLETE!")
    print("="*80)
    

if __name__ == "__main__":
    main()