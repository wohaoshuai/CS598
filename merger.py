import pandas as pd

# Load pairs and all_stays
pairs = pd.read_csv("medmod_pairs.csv")
stays = pd.read_csv("mimic4extract/data/root/all_stays.csv")

# Merge on subject_id
merged = pd.merge(pairs, stays[['subject_id', 'mortality_inhospital', 'los']], on='subject_id', how='left')

# Save task-specific CSVs
merged[['image_path', 'episode_file', 'mortality_inhospital']].dropna().to_csv("medmod_mortality.csv", index=False)
merged[['image_path', 'episode_file', 'los']].dropna().to_csv("medmod_los.csv", index=False)

print("Saved medmod_mortality.csv and medmod_los.csv")

