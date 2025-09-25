import joblib

def save_artifacts(model, scaler, features, model_path, scaler_path, features_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(features, features_path)
    print("\nSaved model:", model_path)
    print("Saved scaler:", scaler_path)
    print("Saved features list:", features_path)

def load_artifacts(model_path, scaler_path, features_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    return model, scaler, features

def sanity_check(model, scaler, features):
    print("\nSanity checks after loading:")
    print("Loaded model expects n_features =", model.coef_.shape[1])
    if hasattr(scaler, "feature_names_in_"):
        print("Loaded scaler.feature_names_in_:", list(scaler.feature_names_in_))
    print("Loaded features list:", features)
