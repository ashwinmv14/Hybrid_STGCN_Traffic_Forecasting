"""
Automated Ablation Study - Train All 5 Models Back-to-Back
==========================================================

This script trains all 5 ablation models sequentially (unattended).
Estimated total time: 15-20 hours

Models trained:
1. STGCN only (baseline)
2. STGCN + FiLM
3. STGCN + ALIGNN  
4. STGCN + DynAdj
5. Full Model

Usage:
    python train_ablation_batch.py

The script will:
- Train each model for 50 epochs (with early stopping patience=15)
- Save results to experiments/ablation/CA/{model_name}/
- Log everything
- Continue even if one model fails
- Send you a summary at the end
"""

import os
import sys
import yaml
import torch
import numpy as np
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


def train_single_model(model_name, config, device, data_dir, adj_path, node_num, dataloader):
    """Train a single model variant"""
    
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)
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
        engine.train()
        
        # Get final validation MAE
        final_val_mae = engine.best_val
        
        logger.info(f"Training complete: {model_name}")
        logger.info(f"Best validation MAE: {final_val_mae:.4f}")
        
        # Cleanup
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        print(f"✅ {model_name} completed successfully!")
        print(f"   Best val MAE: {final_val_mae:.4f}")
        print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'status': 'success',
            'val_mae': final_val_mae,
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
    print("AUTOMATED ABLATION STUDY - BATCH TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models to train: 5")
    print(f"Estimated time: 15-20 hours")
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
    
    dataloader, _ = load_dataset(data_dir, args, logger=DummyLogger())
    
    print(f"Dataset loaded: {node_num} nodes")
    
    # ================================================================
    # DEFINE ALL CONFIGURATIONS
    # ================================================================
    
    # Reduced epochs for faster training (50 instead of 100)
    # With early stopping patience=15, most models will stop around 30-40 epochs
    
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
            "max_epochs": 50,      # Reduced from 100
            "patience": 15          # Reduced from 30
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
    # TRAIN ALL MODELS SEQUENTIALLY
    # ================================================================
    
    results = {}
    start_time = datetime.now()
    
    for i, (model_name, config) in enumerate(configs.items(), 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/5: {model_name}")
        print(f"{'='*80}")
        
        model_start = datetime.now()
        
        result = train_single_model(
            model_name=model_name,
            config=config,
            device=device,
            data_dir=data_dir,
            adj_path=adj_path,
            node_num=node_num,
            dataloader=dataloader
        )
        
        model_end = datetime.now()
        model_duration = (model_end - model_start).total_seconds() / 3600
        
        result['duration_hours'] = model_duration
        results[model_name] = result
        
        print(f"\nModel {i}/5 completed in {model_duration:.2f} hours")
        
        # Save intermediate results
        with open("experiments/ablation/progress.txt", "a") as f:
            f.write(f"{datetime.now()}: {model_name} - {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"  Val MAE: {result['val_mae']:.4f}\n")
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"Total time: {total_duration:.2f} hours")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by model name (which has numerical prefix)
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    print(f"\n{'Model':<20} {'Status':<10} {'Val MAE':<10} {'Time (hrs)':<10}")
    print("-" * 60)
    
    successful = []
    failed = []
    
    for model_name, result in sorted_results:
        status = result['status']
        duration = result['duration_hours']
        
        if status == 'success':
            val_mae = result['val_mae']
            print(f"{model_name:<20} ✅ Success   {val_mae:<10.4f} {duration:<10.2f}")
            successful.append((model_name, val_mae))
        else:
            error = result.get('error', 'Unknown error')
            print(f"{model_name:<20} ❌ Failed    N/A        {duration:<10.2f}")
            print(f"  Error: {error[:60]}")
            failed.append(model_name)
    
    # ================================================================
    # ABLATION ANALYSIS
    # ================================================================
    
    if len(successful) >= 2:
        print("\n" + "="*80)
        print("ABLATION ANALYSIS")
        print("="*80)
        
        # Extract results
        results_dict = {name: mae for name, mae in successful}
        
        if '5_full_model' in results_dict and '1_stgcn_only' in results_dict:
            baseline_mae = results_dict['1_stgcn_only']
            full_mae = results_dict['5_full_model']
            total_improvement = baseline_mae - full_mae
            
            print(f"\n📊 Overall Performance:")
            print(f"  Baseline (STGCN only): {baseline_mae:.4f}")
            print(f"  Full Model:            {full_mae:.4f}")
            print(f"  Total Improvement:     {total_improvement:.4f} ({(total_improvement/baseline_mae)*100:.1f}%)")
            
            if total_improvement > 0:
                print(f"\n✅ Full model is BETTER than baseline by {total_improvement:.4f} MAE")
            else:
                print(f"\n⚠️  WARNING: Full model is WORSE than baseline!")
        
        # Individual component contributions
        print(f"\n📊 Component Contributions:")
        
        if '2_stgcn_film' in results_dict and '1_stgcn_only' in results_dict:
            film_contribution = results_dict['1_stgcn_only'] - results_dict['2_stgcn_film']
            print(f"  FiLM:     {film_contribution:+.4f} MAE")
        
        if '3_stgcn_alignn' in results_dict and '1_stgcn_only' in results_dict:
            alignn_contribution = results_dict['1_stgcn_only'] - results_dict['3_stgcn_alignn']
            print(f"  ALIGNN:   {alignn_contribution:+.4f} MAE")
        
        if '4_stgcn_dyn' in results_dict and '1_stgcn_only' in results_dict:
            dyn_contribution = results_dict['1_stgcn_only'] - results_dict['4_stgcn_dyn']
            print(f"  DynAdj:   {dyn_contribution:+.4f} MAE")
    
    # ================================================================
    # SAVE RESULTS
    # ================================================================
    
    summary_path = "experiments/ablation/CA/ablation_summary.txt"
    os.makedirs("experiments/ablation/CA", exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_duration:.2f} hours\n")
        f.write(f"Successful: {len(successful)}/5\n")
        f.write(f"Failed: {len(failed)}/5\n\n")
        
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        for model_name, result in sorted_results:
            if result['status'] == 'success':
                f.write(f"{model_name}: {result['val_mae']:.4f} MAE\n")
            else:
                f.write(f"{model_name}: FAILED\n")
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    # ================================================================
    # NEXT STEPS
    # ================================================================
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if len(successful) == 5:
        print("\n✅ All models trained successfully!")
        print("\n1. Evaluate all models on test set:")
        print("   python eval_ablation_all.py")
        print("\n2. Create ablation table for paper")
        print("\n3. Visualize component contributions")
        
    elif len(successful) >= 3:
        print(f"\n⚠️  {len(successful)}/5 models succeeded")
        print(f"   Failed: {', '.join(failed)}")
        print("\n1. Check logs for failed models")
        print("2. Retrain failed models with adjusted configs")
        print("3. Proceed with successful models for now")
        
    else:
        print(f"\n❌ Only {len(successful)}/5 models succeeded")
        print("   Check error logs and configurations")
        print(f"   Failed models: {', '.join(failed)}")
    
    print("\n" + "="*80)
    print("🎉 BATCH TRAINING COMPLETE!")
    print("="*80)
    

if __name__ == "__main__":
    main()