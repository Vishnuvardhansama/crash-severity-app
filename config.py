import os

DATA_DIR = "archive"
COMBINED_CSV_PATH = os.path.join(DATA_DIR, "combined_featured_crash_person_vehicle.csv")
MODEL_PATH = os.path.join(DATA_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "minmax_scaler.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "features.pkl")

SELECTED_FEATURES = ['crash_hour', 'crash_month', 'HOUR', 'MINUTE', 'is_night']
