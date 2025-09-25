from loaddata import load_and_process_data
from preprocess import preprocess
from train import train_model
from evaluate import evaluate_model
from artifacts import save_artifacts, load_artifacts, sanity_check
from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, DATA_DIR, SELECTED_FEATURES

# Load and preprocess
df = load_and_process_data()
X_scaled, y, scaler = preprocess(df)

# Train model
lr, X_train, X_test, y_train, y_test = train_model(X_scaled, y)

# Evaluate
evaluate_model(lr, X_test, y_test, DATA_DIR)

# Save artifacts
save_artifacts(lr, scaler, SELECTED_FEATURES, MODEL_PATH, SCALER_PATH, FEATURES_PATH)

# Load and sanity check
loaded_model, loaded_scaler, loaded_feats = load_artifacts(MODEL_PATH, SCALER_PATH, FEATURES_PATH)
sanity_check(loaded_model, loaded_scaler, loaded_feats)
