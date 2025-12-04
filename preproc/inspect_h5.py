import h5py

path = "ca_his_raw_2019.h5"

with h5py.File(path, "r") as f:
    print("=== HDF5 Structure ===")
    print("Top-level keys:", list(f.keys()))
    print()

    for key in f.keys():
        print(f"Key: {key}")
        obj = f[key]
        print("  Type:", type(obj))

        # If it's a group, list subkeys
        if isinstance(obj, h5py.Group):
            print("  Sub-keys:", list(obj.keys()))
            for sub in obj.keys():
                print(f"    Sub {sub}: type={type(obj[sub])}, shape={getattr(obj[sub], 'shape', None)}")
        else:
            print("  Shape:", obj.shape)
            print("  Dtype:", obj.dtype)

        # Print attributes
        print("  Attributes:", dict(obj.attrs))
        print("-----------------------------------")
