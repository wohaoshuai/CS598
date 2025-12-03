import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# ------------------------------
# 1️⃣ Load paired episodes and CXRs
# ------------------------------
pairs_file = "medmod_pairs.csv"
pairs_df = pd.read_csv(pairs_file)
print(f"Loaded {len(pairs_df)} pairs")

# ------------------------------
# 2️⃣ Verify image paths
# ------------------------------
missing_images = pairs_df[~pairs_df['image_path'].apply(os.path.exists)]
if len(missing_images) > 0:
    print(f"Warning: {len(missing_images)} images are missing")
    print(missing_images[['subject_id', 'study_id', 'image_path']])
else:
    print("All images exist ✅")

# ------------------------------
# 3️⃣ Split into train and validation sets
# ------------------------------
train_df, val_df = train_test_split(pairs_df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

# ------------------------------
# 4️⃣ Define helpers to load data
# ------------------------------
def load_ehr(episode_file):
    """Load EHR time series as a DataFrame"""
    if os.path.exists(episode_file):
        df = pd.read_csv(episode_file)
        return df
    else:
        print(f"Missing episode file: {episode_file}")
        return None

def load_cxr(image_path, target_size=(224, 224)):
    """Load CXR image as a numpy array (resized)"""
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        return np.array(img)
    else:
        print(f"Missing image: {image_path}")
        return None

# ------------------------------
# 5️⃣ Example: load first pair in train set
# ------------------------------
example_pair = train_df.iloc[0]
ehr_example = load_ehr(example_pair['episode_file'])
cxr_example = load_cxr(example_pair['image_path'])

print("EHR example shape:", ehr_example.shape)
print("CXR example shape:", cxr_example.shape)

