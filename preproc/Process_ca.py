import h5py
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="ca_his_raw_2019.h5")
    parser.add_argument("--output", type=str, default="ca_his_2019.npz")
    parser.add_argument("--no_resample", action="store_true")
    args = parser.parse_args()

    print("==============================================")
    print("   LargeST CA Dataset Preprocessing (FINAL)")
    print("==============================================")

    # ------------------------------------------------------
    # LOAD FLOW MATRIX + TIMESTAMPS USING H5PY
    # ------------------------------------------------------
    with h5py.File(args.input, "r") as f:
        g = f["t"]
        data = g["block0_values"][:]         # (T, N)
        raw_ts = g["axis1"][:]               # int64 nanoseconds
        timestamps = pd.to_datetime(raw_ts)

    print(f"Loaded raw data shape: {data.shape}")
    print(f"Date range: {timestamps[0]} → {timestamps[-1]}")

    # ------------------------------------------------------
    # CONVERT TO DATAFRAME FOR RESAMPLING
    # ------------------------------------------------------
    df = pd.DataFrame(data, index=timestamps)

    if not args.no_resample:
        print("Resampling from 5-min → 15-min...")
        df = df.resample("15min").mean().round(0)
        print("After resample:", df.shape)
    else:
        print("Skipping resample.")

    print("Filling NaNs...")
    df = df.fillna(0)

    # ------------------------------------------------------
    # SAVE AS NPZ (AVOID PYTABLES COMPLETELY)
    # ------------------------------------------------------
    print(f"Saving cleaned file → {args.output}")
    np.savez_compressed(
        args.output,
        data=df.values,
        timestamps=df.index.astype(np.int64)
    )

    print("==============================================")
    print("       Preprocessing Completed Successfully ✔")
    print("==============================================")

if __name__ == "__main__":
    main()
