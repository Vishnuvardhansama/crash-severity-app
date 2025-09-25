from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

def evaluate_model(lr, X_test, y_test, output_dir):
    y_pred = lr.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    precision = precision_score(y_test, y_pred, average=None, labels=[0,1])
    recall = recall_score(y_test, y_pred, average=None, labels=[0,1])
    f1 = f1_score(y_test, y_pred, average=None, labels=[0,1])
    print(f"Per-class Precision: {precision}")
    print(f"Per-class Recall:    {recall}")
    print(f"Per-class F1-score:  {f1}")

    y_prob = lr.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_prob)
    print(f"Brier score: {brier:.4f}")

    try:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.figure(figsize=(6,5))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression')
        plt.plot([0,1],[0,1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve (no leakage)')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, "calibration_curve_no_leakage.png")
        plt.savefig(save_path)
        plt.close()
        print("Calibration curve saved to:", save_path)
    except Exception as e:
        print("Calibration curve skipped:", e)
