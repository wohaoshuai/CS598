import os
import pandas as pd
from glob import glob

# ------------------------------
# 1️⃣ Aggregate EHR episodes
# ------------------------------
root_train = "mimic4extract/data/root/train"
root_test  = "mimic4extract/data/root/test"

rows = []

for root in [root_train, root_test]:
    for subject_id in os.listdir(root):
        subj_path = os.path.join(root, subject_id)
        if not os.path.isdir(subj_path):
            continue
        # Load episode CSVs (summary info)
        ep_files = glob(os.path.join(subj_path, "episode*.csv"))
        for ep_file in ep_files:
            df = pd.read_csv(ep_file)
            if df.empty:
                continue
            # Add subject_id and file reference
            df['subject_id'] = int(subject_id)
            df['episode_file'] = ep_file
            # Keep only the first row as summary for pairing
            rows.append(df.iloc[0])

episodes = pd.DataFrame(rows)
print(f"Loaded {len(episodes)} episodes before filtering timestamps.")

# ------------------------------
# 1a️⃣ Load episode times for pairing
# ------------------------------
# Load stays file (contains INTIME/OUTTIME)
stays_file = "mimic4extract/data/root/all_stays.csv"
stays = pd.read_csv(stays_file, parse_dates=["intime", "outtime"])

# Merge episodes with stays to get INTIME/OUTTIME
episodes = episodes.merge(
    stays[['subject_id', 'stay_id', 'intime', 'outtime']],
    on='subject_id',
    how='left'
)

# Drop episodes without valid timestamps
episodes = episodes.dropna(subset=["intime", "outtime"])
print(f"Loaded {len(episodes)} episodes with valid timestamps.")

# ------------------------------
# 2️⃣ Load CXR metadata (AP only)
# ------------------------------
cxr_meta_file = "data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz"
cxr = pd.read_csv(cxr_meta_file)

# Keep AP view only
cxr_ap = cxr[cxr["ViewPosition"] == "AP"].copy()

# Convert StudyDate + StudyTime to datetime
cxr_ap['study_time'] = pd.to_datetime(
    cxr_ap['StudyDate'].astype(str) + ' ' + cxr_ap['StudyTime'].astype(str),
    errors='coerce'
)
cxr_ap = cxr_ap.dropna(subset=['study_time'])
print(f"Loaded AP CXRs: {cxr_ap.shape}")

# ------------------------------
# 3️⃣ Pair episodes with closest CXRs
# ------------------------------
pairs = []

for _, ep in episodes.iterrows():
    sid = ep['subject_id']
    intime = ep['intime']
    outtime = ep['outtime']
    midpoint = intime + (outtime - intime)/2

    # CXRs for this subject
    cxr_subj = cxr_ap[cxr_ap.subject_id == sid]
    if cxr_subj.empty:
        continue

    # Compute closest study
    cxr_subj['time_diff'] = (cxr_subj['study_time'] - midpoint).abs()
    best_cxr = cxr_subj.sort_values('time_diff').iloc[0]

    # Construct image path
    image_path = os.path.join(
        "data/mimic-cxr-jpg/files",
        f"p{sid}",
        f"s{best_cxr['study_id']}",
        f"{best_cxr['dicom_id']}.jpg"
    )

    if not os.path.exists(image_path):
        # Skip if image file is missing
        continue

    pairs.append({
        'subject_id': sid,
        'episode_file': ep['episode_file'],
        'study_id': best_cxr['study_id'],
        'image_path': image_path,
        'time_diff_minutes': (best_cxr['study_time'] - midpoint).total_seconds()/60
    })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("medmod_pairs.csv", index=False)
print(f"Saved medmod_pairs.csv with {len(pairs_df)} pairs.")

