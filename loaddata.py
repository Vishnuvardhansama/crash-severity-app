import pandas as pd
import numpy as np
from config import COMBINED_CSV_PATH, SELECTED_FEATURES

def load_and_process_data():
    print("Loading dataset:", COMBINED_CSV_PATH)
    df = pd.read_csv(COMBINED_CSV_PATH, low_memory=False)
    print("Original dataset shape:", df.shape)

    if 'MAX_SEV' in df.columns:
        print("MAX_SEV unique values (sample):", np.unique(df['MAX_SEV'])[:10])
        print("MAX_SEV value counts (top):\n", df['MAX_SEV'].value_counts().head())

    # Impute missing values
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

    # Derive time features if missing
    if 'CRASH_DATETIME' in df.columns:
        df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
        if 'crash_hour' not in df.columns:
            df['crash_hour'] = df['CRASH_DATETIME'].dt.hour
        if 'crash_month' not in df.columns:
            df['crash_month'] = df['CRASH_DATETIME'].dt.month
        if 'HOUR' not in df.columns:
            df['HOUR'] = df['CRASH_DATETIME'].dt.hour
        if 'MINUTE' not in df.columns:
            df['MINUTE'] = df['CRASH_DATETIME'].dt.minute

    if 'is_night' not in df.columns:
        df['is_night'] = df['crash_hour'].apply(lambda x: 1 if (pd.notna(x) and (x <= 5 or x >= 20)) else 0)

    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"The following required features are missing from your dataset: {missing}")

    if 'MAX_SEV' not in df.columns:
        raise KeyError("MAX_SEV column not found; cannot create target as specified.")
    df['target'] = df['MAX_SEV'].apply(lambda x: 1 if x >= 0.5 else 0)
    print("Target distribution:\n", df['target'].value_counts())

    return df
