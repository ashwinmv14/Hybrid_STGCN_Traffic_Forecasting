"""
Quick script to check the actual mean and std used for normalization in CA dataset.
This tells you the exact normalization parameters.
"""

import numpy as np
import os

# Load the CA dataset
data_path = './data/ca/2019/his.npz'

if os.path.exists(data_path):
    data = np.load(data_path)
    
    print("="*80)
    print("CA DATASET NORMALIZATION PARAMETERS")
    print("="*80)
    
    # Check what's in the file
    print("\nKeys in dataset:", list(data.keys()))
    
    # Get mean and std
    if 'mean' in data:
        mean = data['mean']
        print(f"\nMean shape: {mean.shape}")
        print(f"Mean values: {mean}")
    
    if 'std' in data:
        std = data['std']
        print(f"\nStd shape: {std.shape}")
        print(f"Std values: {std}")
    
    # Get data shape
    if 'data' in data:
        his = data['data']
        print(f"\nData shape: {his.shape}")
        print(f"  Time steps (T): {his.shape[0]}")
        print(f"  Nodes (N): {his.shape[1]}")
        print(f"  Features (C): {his.shape[2]}")
    
    # If mean/std are scalars or arrays
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    if 'std' in data:
        std = data['std']
        if isinstance(std, np.ndarray):
            if std.shape == (3,):  # [flow_std, tod_std, dow_std]
                print(f"\nTraffic Flow std: {std[0]:.4f}")
                print(f"Time-of-Day std: {std[1]:.4f}")
                print(f"Day-of-Week std: {std[2]:.4f}")
                
                print(f"\n📊 To denormalize predictions:")
                print(f"   Actual MAE = Normalized MAE × {std[0]:.4f}")
                print(f"   Example: If normalized MAE = 0.32")
                print(f"            Actual MAE ≈ 0.32 × {std[0]:.4f} = {0.32 * std[0]:.2f}")
            else:
                print(f"\nStd: {std}")
        else:
            print(f"\nStd (scalar): {std}")
            print(f"\n📊 To denormalize predictions:")
            print(f"   Actual MAE = Normalized MAE × {std:.4f}")
    
    print("\n" + "="*80)
    
else:
    print(f"❌ File not found: {data_path}")
    print("\nMake sure you're running from the C:\\LargeST directory")
    print("and that the data has been preprocessed.")