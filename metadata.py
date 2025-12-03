import pandas as pd

# Load the metadata
cxr = pd.read_csv("data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz")

# Filter for AP view (MedMod uses frontal only)
cxr_ap = cxr[cxr["ViewPosition"] == "AP"]

# Save filtered CSV if needed
cxr_ap.to_csv("data/mimic-cxr-jpg/cxr_metadata_ap.csv", index=False)

print("AP metadata loaded:", cxr_ap.shape)
print(cxr_ap.head())

