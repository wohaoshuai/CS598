import pandas as pd

def parse_cxr_datetime(row):
    # StudyDate: YYYYMMDD
    date_str = str(int(row['StudyDate']))  # remove decimals
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    # StudyTime: HHMMSS.SSS
    time_val = row['StudyTime']
    if pd.isna(time_val):
        return pd.Timestamp(year, month, day)
    
    time_val = float(time_val)
    hh = int(time_val // 10000)
    mm = int((time_val % 10000) // 100)
    ss = int(time_val % 100)
    
    return pd.Timestamp(year, month, day, hh, mm, ss)

# Load your CSV
cxr = pd.read_csv("data/mimic-cxr-jpg/cxr_metadata_ap.csv")

# Apply the fix
cxr['study_time'] = cxr.apply(parse_cxr_datetime, axis=1)

print(cxr[['StudyDate','StudyTime','study_time']].head())

