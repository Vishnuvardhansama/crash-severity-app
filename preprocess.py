import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import SELECTED_FEATURES

def preprocess(df):
    X_df = df[SELECTED_FEATURES].copy()
    y = df['target'].values

    for col in SELECTED_FEATURES:
        if X_df[col].isnull().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_scaled, y, scaler
