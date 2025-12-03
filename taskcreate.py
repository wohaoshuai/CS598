import pandas as pd

# Load medmod_pairs.csv (links episodes to CXR images)
pairs = pd.read_csv("medmod_pairs.csv")

# Load episode metadata (contains ICU info and labels)
episodes = pd.read_csv("episodes_index.csv")

# Make sure we have a column to join on. 
# If 'episode_file' paths match exactly, we can use them; otherwise, use 'subject_id' + episode number.
# Here we assume episode_file in pairs matches 'episode_file' in episodes
merged = pd.merge(pairs, episodes[['episode_file', 'subject_id', 'hospital_expire_flag', 'los', 'decompensation_48h']],
                  on=['episode_file', 'subject_id'],
                  how='left')

# Check for missing labels
missing_mort = merged['hospital_expire_flag'].isna().sum()
missing_los = merged['los'].isna().sum()
missing_decomp = merged['decompensation_48h'].isna().sum()
print(f"Missing labels - Mortality: {missing_mort}, LOS: {missing_los}, Decomp: {missing_decomp}")

# Save task-specific CSVs
merged[['image_path', 'episode_file', 'hospital_expire_flag']].dropna().rename(
    columns={'hospital_expire_flag':'mortality_inhospital'}).to_csv("medmod_mortality.csv", index=False)

merged[['image_path', 'episode_file', 'los']].dropna().to_csv("medmod_los.csv", index=False)

merged[['image_path', 'episode_file', 'decompensation_48h']].dropna().to_csv("medmod_decompensation.csv", index=False)

print("Saved medmod_mortality.csv, medmod_los.csv, medmod_decompensation.csv")

