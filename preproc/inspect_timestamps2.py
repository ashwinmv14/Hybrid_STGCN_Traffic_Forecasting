import h5py
import numpy as np

with h5py.File("ca_his_raw_2019.h5", "r") as f:
    g = f["t"]
    raw_ts = g["axis1"][:]

print("dtype:", raw_ts.dtype)
print("first 20 values:", raw_ts[:20])
print("last 20 values:", raw_ts[-20:])
print("min:", raw_ts.min(), "max:", raw_ts.max())
print("length:", len(raw_ts))
