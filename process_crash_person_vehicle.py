import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATA_DIR = "archive"
YEARS = ["16", "17", "18", "19", "20"]
all_years = []

def build_approx_crash_datetime(df):
    # Approximate datetime with day fixed at 15 (no real calendar day)
    if all(col in df.columns for col in ['YEAR', 'MONTH', 'HOUR', 'MINUTE']):
        return pd.to_datetime({
            'year': df['YEAR'],
            'month': df['MONTH'],
            'day': 15,
            'hour': df['HOUR'].where(df['HOUR'] < 24, 0),
            'minute': df['MINUTE'].where(df['MINUTE'] < 60, 0),
            'second': 0
        }, errors='coerce')
    else:
        print("Missing required columns to build approximate datetime.")
        return pd.NaT

for year in YEARS:
    print(f"Processing year 20{year}...")

    acc = pd.read_csv(f"{DATA_DIR}/acc_{year}.csv", low_memory=False, encoding='latin1')
    pers = pd.read_csv(f"{DATA_DIR}/pers_{year}.csv", low_memory=False, encoding='latin1')
    veh = pd.read_csv(f"{DATA_DIR}/veh_{year}.csv", low_memory=False, encoding='latin1')

    acc['CRASH_DATETIME'] = build_approx_crash_datetime(acc)
    acc = acc.dropna(subset=['CRASH_DATETIME']).drop_duplicates()
    pers = pers.drop_duplicates()
    veh = veh.drop_duplicates()

    acc_pers = pd.merge(acc, pers, how='left', on='CASENUM', suffixes=('_acc', '_pers'))
    full_df = pd.merge(acc_pers, veh, how='left', on='CASENUM', suffixes=('', '_veh'))

    # --- Handle missing values ---
    for col in full_df.select_dtypes(include=['float', 'int']).columns:
        full_df[col] = full_df[col].fillna(full_df[col].median())
    for col in full_df.select_dtypes(include=['object']).columns:
        full_df[col] = full_df[col].fillna(full_df[col].mode().iloc[0])

    # --- Create all new features in a separate DataFrame ---
    new_features = pd.DataFrame(index=full_df.index)

    # Categorical severity feature
    if 'MAX_SEV' in full_df.columns:
        new_features['SEV_CAT'] = pd.cut(full_df['MAX_SEV'], bins=[0,1,2,3,4],
                                        labels=['None','Minor','Serious','Fatal'])

    # Time-based features
    new_features['crash_hour'] = full_df['CRASH_DATETIME'].dt.hour
    new_features['crash_month'] = full_df['CRASH_DATETIME'].dt.month
    new_features['crash_weekday'] = full_df['CRASH_DATETIME'].dt.dayofweek
    new_features['is_night'] = new_features['crash_hour'].apply(lambda x: 1 if (x <= 5 or x >= 20) else 0)

    # Spatial feature
    if 'URBANICITY' in full_df.columns:
        new_features['is_urban'] = full_df['URBANICITY'].apply(lambda x: 1 if str(x).strip() in ['1','Urban','urban'] else 0)

    # Merge new features into full_df once, minimizing fragmentation
    full_df = pd.concat([full_df, new_features], axis=1)

    # --- Encode categorical features ---
    cat_cols = [col for col in ['SEV_CAT', 'REGION', 'URBANICITY'] if col in full_df.columns]
    for col in cat_cols:
        le = LabelEncoder()
        full_df[col+'_LE'] = le.fit_transform(full_df[col].astype(str))

    # One-hot encode weekday
    full_df = pd.get_dummies(full_df, columns=['crash_weekday'], prefix='dow')

    # --- Scale numerical features ---
    num_cols = [c for c in ['crash_hour','crash_month','MAX_SEV','HOUR','MINUTE'] if c in full_df.columns]
    scaler = MinMaxScaler()
    full_df[num_cols] = scaler.fit_transform(full_df[num_cols])

    # Defragment frame after all inserts
    full_df = full_df.copy()
    print(full_df.columns.tolist())
    print(f"Shape: {full_df.shape}")
    print("Sample rows:")
    print(full_df.head())

    all_years.append(full_df)
    print(f"Shape: {full_df.shape}")
    print("Sample rows:")
    print(full_df.head)
    print(f"Finished processing year 20{year} with shape {full_df.shape}")

# Combine all years
if all_years:
    combined_df = pd.concat(all_years, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(full_df.columns.tolist())
    # Optionally save to CSV:
    # combined_df.to_csv(os.path.join(DATA_DIR, "combined_featured_crash_person_vehicle.csv"), index=False)
else:
    print("No data combined - check for errors.")
