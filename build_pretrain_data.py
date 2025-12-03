import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ===== CONFIG =====
PAIRS_FILE = "medmod_pairs.csv"
OUT_TRAIN = "train.jsonl"
OUT_VAL   = "val.jsonl"
VAL_SPLIT = 0.1  # fraction of validation samples
# ==================

def load_timeseries(path):
    """Load episode CSV and process Hours."""
    df = pd.read_csv(path)

    # Only process _timeseries.csv files
    if "_timeseries.csv" not in path:
        raise ValueError(f"Not a timeseries CSV: {path}")

    # Detect time column
    time_col = None
    for col in df.columns:
        if col.lower() in ["hours", "time"]:
            time_col = col
            break

    if time_col is None:
        raise ValueError(f"No time column found in {path}. Columns: {df.columns}")

    # Shift time so min = 0
    if df[time_col].min() < 0:
        df[time_col] = df[time_col] - df[time_col].min()

    # Remove duplicate hours
    df = df.groupby(time_col).mean().reset_index()

    # Resample to 1-hour bins
    df = df.set_index(time_col)
    all_hours = np.arange(0, df.index.max() + 1)
    df = df.reindex(all_hours)

    # Fill missing values forward then backward
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df

def normalize_timeseries(df):
    """Normalize all EHR values."""
    scaler = StandardScaler()
    values = df.values
    # Some columns may be all NaN; replace with 0
    values = np.nan_to_num(values)
    values = scaler.fit_transform(values)
    return values

def build_records():
    """Build dataset records from medmod_pairs.csv."""
    pairs = pd.read_csv(PAIRS_FILE)
    records = []
    skipped_files = []

    for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
        episode_path = row["episode_file"]
        img_path = row["image_path"]

        # Skip missing files
        if not os.path.exists(episode_path):
            skipped_files.append((episode_path, "missing episode"))
            continue
        if not os.path.exists(img_path):
            skipped_files.append((img_path, "missing image"))
            continue

        # Load & process EHR timeseries
        try:
            ehr_df = load_timeseries(episode_path)
        except ValueError as e:
            skipped_files.append((episode_path, str(e)))
            continue

        ehr_matrix = normalize_timeseries(ehr_df)

        record = {
            "ehr": ehr_matrix.tolist(),
            "image": img_path,
            "los": float(row["time_diff_minutes"]) if "time_diff_minutes" in row else None
        }
        records.append(record)

    return records, skipped_files

def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    print("Building MedMod pretrain dataset...")

    records, skipped = build_records()
    print(f"Total valid samples: {len(records)}")
    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for f, reason in skipped:
            print(f" - {f}: {reason}")

    # Train/validation split
    n_val = int(len(records) * VAL_SPLIT)
    val_records = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, OUT_TRAIN)
    write_jsonl(val_records, OUT_VAL)

    print(f"Done! Wrote {len(train_records)} train and {len(val_records)} val samples.")
