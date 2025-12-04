"""
Evaluate STGCN + ALIGNN (100 epochs) on test set
This will give you the DENORMALIZED metrics to compare with Table 2
"""
import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from src.models.hybrid.hybrid_model import HybridModel
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import calculate_sym_adj
from src.utils.metrics import masked_mae, masked_rmse, masked_mape


def load_edge_index(adj):
    row, col = np.nonzero(adj)
    return torch.tensor([row, col], dtype=torch.long)


def evaluate_model(model, loader, device, scaler):
    """Evaluate and return DENORMALIZED metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader.get_iterator():
            x = torch.tensor(x, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            
            t_index = x[:, :, 0, 1]
            x = x.permute(0, 3, 1, 2)
            
            y_pred = model(x, t_index)
            
            all_preds.append(y_pred.cpu())
            all_labels.append(y.cpu())
    
    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    
    # DENORMALIZE (this is the key!)
    y_pred_denorm = scaler.inverse_transform(y_pred)
    y_true_denorm = scaler.inverse_transform(y_true)
    
    # Compute metrics on DENORMALIZED data
    mae = masked_mae(y_pred_denorm, y_true_denorm, null_val=0.0).item()
    rmse = masked_rmse(y_pred_denorm, y_true_denorm, null_val=0.0).item()
    mape = masked_mape(y_pred_denorm, y_true_denorm, null_val=0.0).item()
    
    # Also compute per-horizon metrics for detailed analysis
    horizon_results = []
    for h in range(12):
        pred_h = y_pred_denorm[:, h, :, :]
        true_h = y_true_denorm[:, h, :, :]
        
        h_mae = masked_mae(pred_h, true_h, null_val=0.0).item()
        h_rmse = masked_rmse(pred_h, true_h, null_val=0.0).item()
        h_mape = masked_mape(pred_h, true_h, null_val=0.0).item()
        
        horizon_results.append({
            'horizon': h + 1,
            'mae': h_mae,
            'rmse': h_rmse,
            'mape': h_mape
        })
    
    return mae, rmse, mape, horizon_results


def main():
    print("="*80)
    print("EVALUATING STGCN + ALIGNN (100 EPOCHS) ON TEST SET")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = "CA"
    
    # Args setup
    class Args:
        pass
    
    args = Args()
    args.dataset = dataset
    args.years = "2019"
    args.seq_len = 12
    args.horizon = 12
    args.input_dim = 3
    args.bs = 64
    
    # Load dataset
    data_dir, adj_path, node_num = get_dataset_info(dataset)
    
    class DummyLogger:
        def info(self, msg):
            print(msg)
    
    dataloader, scaler = load_dataset(data_dir, args, DummyLogger())
    test_loader = dataloader["test_loader"]
    
    print(f"\nScaler mean: {scaler.mean}")
    print(f"Scaler std: {scaler.std}")
    
    # Load adjacency
    adj = load_adj_from_numpy(adj_path).astype(np.float32)
    A_norm = calculate_sym_adj(adj)
    A_norm = A_norm.todense().astype(np.float32)
    A_static = torch.tensor(A_norm, device=device)
    edge_index = load_edge_index(adj).to(device)
    
    # Create model
    model = HybridModel(
        node_num=node_num,
        input_dim=3,
        horizon=12,
        A_static=A_static,
        edge_index=edge_index,
        use_film=False,
        use_alignn=True,
        use_dynamic_adj=False,
        film_hidden=64,
        alignn_hidden=64,
        alpha_dyn=0.005
    ).to(device)
    
    # Load checkpoint
    ckpt_path = "experiments/hybrid/CA/best_model.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"\n❌ ERROR: Checkpoint not found at {ckpt_path}")
        print("Please wait for training to finish!")
        return
    
    print(f"\nLoading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print("✅ Model loaded successfully!")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET...")
    print("="*80)
    
    mae, rmse, mape, horizon_results = evaluate_model(model, test_loader, device, scaler)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS (DENORMALIZED - Comparable to Table 2)")
    print("="*80)
    
    print(f"\nAverage across all 12 horizons:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    print(f"\n" + "-"*80)
    print("COMPARISON TO LARGESÐ¢ PAPER TABLE 2 (CA dataset):")
    print("-"*80)
    print(f"STGCN (paper):        MAE: 21.33  RMSE: 36.39  MAPE: 16.53%")
    print(f"GWNET (paper):        MAE: 21.72  RMSE: 34.20  MAPE: 17.40%")
    print(f"STGODE (paper BEST): MAE: 20.77  RMSE: 36.60  MAPE: 16.80%")
    print(f"{'─'*80}")
    print(f"Your STGCN+ALIGNN:    MAE: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.2f}%")
    
    if mae < 20.77:
        print(f"\n🎉 YOU BEAT THE PAPER'S BEST MODEL (STGODE)!")
    elif mae < 21.33:
        print(f"\n✅ YOU BEAT THE BASELINE STGCN!")
    else:
        print(f"\n⚠️  Performance is below baseline")
    
    # Detailed horizon breakdown
    print(f"\n" + "="*80)
    print("DETAILED PER-HORIZON RESULTS")
    print("="*80)
    print(f"\nHorizon |  MAE  |  RMSE  | MAPE")
    print("-" * 40)
    for h_result in horizon_results:
        print(f"   {h_result['horizon']:2d}   | {h_result['mae']:6.2f} | {h_result['rmse']:6.2f} | {h_result['mape']:5.2f}%")
    
    # Save results
    out_dir = "experiments/stgcn_alignn_100ep/CA"
    out_path = os.path.join(out_dir, "test_results_100ep.csv")
    
    results_df = pd.DataFrame([{
        'Model': 'STGCN+ALIGNN_100ep',
        'Average_MAE': mae,
        'Average_RMSE': rmse,
        'Average_MAPE': mape,
        'Horizon3_MAE': horizon_results[2]['mae'],
        'Horizon6_MAE': horizon_results[5]['mae'],
        'Horizon12_MAE': horizon_results[11]['mae'],
    }])
    
    results_df.to_csv(out_path, index=False)
    print(f"\n📊 Results saved to: {out_path}")
    print("="*80)


if __name__ == "__main__":
    main()