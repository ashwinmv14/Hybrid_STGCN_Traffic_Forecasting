"""
Results Analysis - Why is Hybrid Model Underperforming?
========================================================

This script analyzes your results and suggests improvements.
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("RESULTS ANALYSIS")
print("="*80)
print()

# Your results
your_mae = 22.60
your_rmse = 38.20
your_mape = 15.53

# LargeST baselines
baselines = {
    'STGCN': {'MAE': 21.33, 'RMSE': 36.39, 'MAPE': 16.53},
    'GWNET': {'MAE': 21.72, 'RMSE': 34.20, 'MAPE': 17.40},
    'STGODE': {'MAE': 20.77, 'RMSE': 36.60, 'MAPE': 16.80},
    'DCRNN': {'MAE': 21.87, 'RMSE': 34.41, 'MAPE': 17.06},
}

print("YOUR HYBRID MODEL vs BASELINES")
print("-" * 80)
print(f"{'Method':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'vs You':<15}")
print("-" * 80)
print(f"{'Your Hybrid':<15} {your_mae:<10.2f} {your_rmse:<10.2f} {your_mape:<10.2f}% {'(baseline)':<15}")
print()

for name, metrics in sorted(baselines.items(), key=lambda x: x[1]['MAE']):
    mae_diff = your_mae - metrics['MAE']
    rmse_diff = your_rmse - metrics['RMSE']
    mape_diff = your_mape - metrics['MAPE']
    
    status = "✓ Better" if mae_diff < 0 else "✗ Worse"
    print(f"{name:<15} {metrics['MAE']:<10.2f} {metrics['RMSE']:<10.2f} {metrics['MAPE']:<10.2f}% ", end="")
    print(f"{status:<15}")

print()
print("="*80)
print("DETAILED COMPARISON")
print("="*80)
print()

best_mae = min(m['MAE'] for m in baselines.values())
best_rmse = min(m['RMSE'] for m in baselines.values())
best_mape = min(m['MAPE'] for m in baselines.values())

print(f"Best MAE:  {best_mae:.2f} (STGODE) - You: {your_mae:.2f} (+{your_mae-best_mae:.2f}, {(your_mae-best_mae)/best_mae*100:.1f}% worse)")
print(f"Best RMSE: {best_rmse:.2f} (GWNET)  - You: {your_rmse:.2f} (+{your_rmse-best_rmse:.2f}, {(your_rmse-best_rmse)/best_rmse*100:.1f}% worse)")
print(f"Best MAPE: {best_mape:.2f}% (MAPE)   - You: {your_mape:.2f}% ({your_mape-best_mape:.2f}%, {abs(your_mape-best_mape)/best_mape*100:.1f}% better) ✓")

print()
print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

print("🔍 Key Observations:")
print()
print("1. MAPE Performance: GOOD")
print("   - Your MAPE (15.53%) beats STGCN (16.53%)")
print("   - This suggests your model handles relative errors well")
print()

print("2. MAE Performance: NEEDS IMPROVEMENT")
print("   - Gap of +1.27 MAE compared to STGCN")
print("   - Gap of +1.83 MAE compared to STGODE (best)")
print("   - This is ~6-9% worse than baselines")
print()

print("3. RMSE Performance: NEEDS IMPROVEMENT")
print("   - Gap of +1.81 RMSE compared to STGCN")
print("   - Gap of +4.00 RMSE compared to GWNET (best)")
print("   - RMSE penalizes large errors more → suggests occasional large mispredictions")
print()

print("="*80)
print("POSSIBLE CAUSES")
print("="*80)
print()

print("❓ Why might hybrid components not be helping?")
print()
print("1. Component Integration Issues:")
print("   - FiLM/ALIGNN/DynamicAdj might be interfering with each other")
print("   - Ablation study will reveal which components actually help")
print()

print("2. Hyperparameter Mismatch:")
print("   - alpha_dyn=0.005 might be too small (dynamic adj has minimal effect)")
print("   - Learning rate or other params might not be optimal")
print("   - Clip gradient norm (5.0) might be limiting learning")
print()

print("3. Training Stability:")
print("   - Your validation MAE (0.1034) is worse than diagnostic baseline (0.1001)")
print("   - This suggests the model might not have converged optimally")
print()

print("4. Architecture Balance:")
print("   - Adding components increased model complexity")
print("   - May need more training or different architecture balance")
print()

print("="*80)
print("RECOMMENDED ACTION PLAN")
print("="*80)
print()

print("📋 Priority Order:")
print()
print("1. ⭐ RUN ABLATION STUDY (CRITICAL)")
print("   Why: You NEED to know which components help/hurt")
print("   How: python run_ablation_study.py")
print("   Time: ~8-12 hours")
print("   Output: Will tell you if FiLM/ALIGNN/DynamicAdj are beneficial")
print()

print("2. Based on ablation results:")
print()
print("   If all components help:")
print("   → Move to hyperparameter tuning")
print("   → Focus on: learning rate, alpha_dyn, hidden dims")
print()
print("   If some components hurt:")
print("   → Remove weak components")
print("   → Retrain with only beneficial ones")
print()
print("   If no components help:")
print("   → Your baseline STGCN is actually the best")
print("   → Research contribution becomes 'rigorous evaluation showing X doesn't help'")
print()

print("3. After finding best configuration:")
print("   → Optuna tuning for 10-20 trials")
print("   → Should get you to MAE ~21.0, RMSE ~35.0")
print()

print("="*80)
print("IMMEDIATE NEXT STEP")
print("="*80)
print()
print("Run this command now:")
print("  python run_ablation_study.py")
print()
print("Then come back in ~10 hours and run:")
print("  python analyze_ablation_results.py")
print()
print("This will show you which components to keep/remove.")
print()
print("="*80)