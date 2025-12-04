"""
Quick Diagnostic Checker - Run this to see what you have
=========================================================

This script checks all your experimental results and tells you what to do next.

Usage:
    python quick_check.py
"""

import os
from pathlib import Path

def check_exists(path):
    """Check if path exists and return file size"""
    p = Path(path)
    if p.exists():
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            return True, f"{size_mb:.1f} MB"
        return True, "DIR"
    return False, None

def read_last_lines(filepath, n=10):
    """Read last n lines of file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

print("="*80)
print("DIAGNOSTIC RESULTS CHECK".center(80))
print("="*80)
print()

# Define paths
base = Path("experiments")
diag_ckpt = base / "diagnostic/stgcn_baseline/best_model.pt"
diag_log = base / "diagnostic/stgcn_baseline/test.log"
hybrid_ckpt = base / "hybrid/CA/best_model.pt"
hybrid_log = base / "hybrid/CA/record.log"
test_csv = base / "hybrid/CA/test_horizons_corrected.csv"
ablation_log = base / "ablation/CA/1_stgcn_only/training.log"

# Check diagnostic
print("📊 DIAGNOSTIC TEST (STGCN Baseline)")
print("-" * 80)
exists, size = check_exists(diag_ckpt)
if exists:
    print(f"✓ Checkpoint: {diag_ckpt} ({size})")
else:
    print(f"✗ Checkpoint: NOT FOUND")

exists, size = check_exists(diag_log)
if exists:
    print(f"✓ Test Log:   {diag_log} ({size})")
    print("\n  Last 10 lines:")
    for line in read_last_lines(diag_log, 10):
        print(f"    {line.rstrip()}")
else:
    print(f"✗ Test Log:   NOT FOUND")

print()

# Check hybrid model
print("🔬 HYBRID MODEL")
print("-" * 80)
exists, size = check_exists(hybrid_ckpt)
if exists:
    print(f"✓ Checkpoint: {hybrid_ckpt} ({size})")
else:
    print(f"✗ Checkpoint: NOT FOUND")

exists, size = check_exists(hybrid_log)
if exists:
    print(f"✓ Train Log:  {hybrid_log} ({size})")
    print("\n  Last 10 lines:")
    for line in read_last_lines(hybrid_log, 10):
        print(f"    {line.rstrip()}")
else:
    print(f"✗ Train Log:  NOT FOUND")

print()

# Check test results
print("📈 TEST RESULTS")
print("-" * 80)
exists, size = check_exists(test_csv)
if exists:
    print(f"✓ Results CSV: {test_csv} ({size})")
    print("\n  Contents:")
    try:
        with open(test_csv, 'r') as f:
            for line in f:
                print(f"    {line.rstrip()}")
    except:
        print("    (Could not read file)")
else:
    print(f"✗ Results CSV: NOT FOUND")

print()

# Check ablation
print("🧪 ABLATION STUDY")
print("-" * 80)
exists, size = check_exists(ablation_log)
if exists:
    print(f"✓ Ablation 1: {ablation_log} ({size})")
    print("\n  Last 5 lines:")
    for line in read_last_lines(ablation_log, 5):
        print(f"    {line.rstrip()}")
else:
    print(f"✗ Ablation:   NOT STARTED")

print()
print("="*80)
print("SUMMARY & NEXT STEPS".center(80))
print("="*80)
print()

# Determine status
has_diag = check_exists(diag_ckpt)[0]
has_hybrid = check_exists(hybrid_ckpt)[0]
has_results = check_exists(test_csv)[0]
has_ablation = check_exists(ablation_log)[0]

print("Current Status:")
print(f"  Diagnostic Test:  {'✓ COMPLETE' if has_diag else '✗ INCOMPLETE'}")
print(f"  Hybrid Training:  {'✓ COMPLETE' if has_hybrid else '✗ INCOMPLETE'}")
print(f"  Evaluation:       {'✓ COMPLETE' if has_results else '✗ INCOMPLETE'}")
print(f"  Ablation Study:   {'✓ STARTED' if has_ablation else '✗ NOT STARTED'}")
print()

# Recommendations
if has_results:
    print("✨ Great! You have complete results. Here's what you can do next:")
    print()
    print("  Option A: Run Ablation Study")
    print("    → Understand which components contribute most")
    print("    → Command: python run_ablation_study.py")
    print()
    print("  Option B: Hyperparameter Tuning")
    print("    → Improve metrics beyond current results")
    print("    → Command: python optuna_tune.py")
    print()
    print("  Option C: Scale to Larger Datasets")
    print("    → Test on GBA (2,352 nodes) or GLA (3,834 nodes)")
    print("    → Command: python main.py --dataset GBA --device cuda:0")
    print()
    
    # Show current metrics if available
    if has_results:
        print("  Your Current Metrics (CA dataset):")
        try:
            import pandas as pd
            df = pd.read_csv(test_csv)
            if 'Average_MAE' in df.columns:
                print(f"    Average MAE:  {df['Average_MAE'].values[0]:.2f}")
                print(f"    Average RMSE: {df['Average_RMSE'].values[0]:.2f}")
                print(f"    Average MAPE: {df['Average_MAPE'].values[0]:.2f}%")
                print()
                print("  LargeST Paper Baselines (CA dataset - for comparison):")
                print("    STGCN:   MAE=21.33, RMSE=36.39, MAPE=16.53%")
                print("    GWNET:   MAE=21.72, RMSE=34.20, MAPE=17.40%")
                print("    STGODE:  MAE=20.77, RMSE=36.60, MAPE=16.80%")
        except:
            pass

elif has_hybrid:
    print("⚠️  You have a trained model but no evaluation results.")
    print()
    print("  Next Step: Run Evaluation")
    print("    cd experiments/hybrid")
    print("    python eval_horizons.py")
    print()

else:
    print("⚠️  Training incomplete or not found.")
    print()
    print("  Next Step: Complete Training")
    print("    python eval_horizons.py")
    print()
    print("  This will:")
    print("    - Train the full hybrid model on CA dataset")
    print("    - Evaluate on all 12 horizons")
    print("    - Take approximately 2-3 hours")
    print()

print("="*80)