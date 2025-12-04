import os
import argparse
import numpy as np
import pandas as pd


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_clean_npz(path):
    """Load the cleaned NPZ file produced by process_ca.py"""
    print(f"Loading cleaned data from: {path}")
    npz = np.load(path)

    data = npz["data"]                    # shape (T, N)
    timestamps = npz["timestamps"]        # int64 nanoseconds
    index = pd.to_datetime(timestamps)    # convert to DateTimeIndex

    df = pd.DataFrame(data, index=index)
    print("Loaded DataFrame:", df.shape)
    return df


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape

    # Expand to (T, N, 1)
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]

    # --- Time of Day ---
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) \
                    / np.timedelta64(1, 'D')
        tod = np.tile(time_ind, (num_nodes, 1)).T   # (T, N)
        tod = np.expand_dims(tod, axis=-1)
        feature_list.append(tod)

    # --- Day of Week ---
    if add_day_of_week:
        dow = df.index.dayofweek
        dow = np.tile(dow, (num_nodes, 1)).T / 7.0
        dow = np.expand_dims(dow, axis=-1)
        feature_list.append(dow)

    # Concatenate along last dimension → (T, N, C)
    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))
    print('idx min & max:', min_t, max_t)

    idx = np.arange(min_t, max_t)
    return data, idx


def generate_train_val_test(args):
    years = args.years.split('_')
    df = pd.DataFrame()

    # -----------------------------------------------
    # LOAD ALL YEARS AS .npz (NO PyTables)
    # -----------------------------------------------
    for y in years:
        file_path = f"{args.dataset}/{args.dataset}_his_{y}.npz"
        df_y = load_clean_npz(file_path)
        df = pd.concat([df, df_y])

    print("Final merged data shape:", df.shape)

    # -----------------------------------------------
    # Offsets
    # -----------------------------------------------
    seq_x, seq_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_x - 1), 1)
    y_offsets = np.arange(1, seq_y + 1)

    # -----------------------------------------------
    # Build (T, N, C) feature tensor
    # -----------------------------------------------
    data, idx = generate_data_and_idx(df, x_offsets, y_offsets,
                                      args.tod, args.dow)

    print("Final data shape:", data.shape, "| idx:", idx.shape)

    # -----------------------------------------------
    # Train/Val/Test split
    # -----------------------------------------------
    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)

    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    # -----------------------------------------------
    # NORMALIZE ONLY FLOW CHANNEL
    # -----------------------------------------------
    x_train = data[:idx_val[0] - seq_x, :, 0]
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())

    data[..., 0] = scaler.transform(data[..., 0])

    # -----------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------
    out_dir = f"{args.dataset}/{args.years}"
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(os.path.join(out_dir, 'his.npz'),
                        data=data,
                        mean=scaler.mean,
                        std=scaler.std)

    np.save(os.path.join(out_dir, 'idx_train.npy'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val.npy'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test.npy'), idx_test)

    print("All files saved under:", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ca')
    parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--seq_length_x', type=int, default=12)
    parser.add_argument('--seq_length_y', type=int, default=12)
    parser.add_argument('--tod', type=int, default=1)
    parser.add_argument('--dow', type=int, default=1)

    args = parser.parse_args()
    generate_train_val_test(args)
