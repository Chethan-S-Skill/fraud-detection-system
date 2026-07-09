import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

def evaluate_model(models_dir="models", reports_dir="reports", threshold=0.5):
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load model and test data
    model_path = os.path.join(models_dir, "xgboost_model.joblib")
    test_data_path = os.path.join(models_dir, "test_data.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError("Trained model or test data not found. Please run train.py first.")
        
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)
    
    # 2. Get predictions and risk probabilities
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # 3. Calculate Core Metrics
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Print metrics report
    print("=" * 60)
    print("                 MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Classification Threshold: {threshold:.2f}")
    print(f"Area Under ROC (ROC-AUC):  {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {pr_auc:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  Predicted Legit   - True Legit: {tn:6d} (TN) | True Fraud: {fn:4d} (FN)")
    print(f"  Predicted Fraud   - True Legit: {fp:6d} (FP) | True Fraud: {tp:4d} (TP)")
    print("-" * 60)
    
    report = classification_report(y_test, preds, target_names=['Legitimate', 'Fraud'])
    print(report)
    print("=" * 60)
    
    # 4. Generate Performance Plots
    plt.figure(figsize=(14, 5))
    
    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, color='#8b5cf6', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#6b7280', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precisions, recalls, _ = precision_recall_curve(y_test, probs)
    plt.plot(recalls, precisions, color='#10b981', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PR)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "performance_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Saved evaluation plots to {plot_path}")
    
    # Return metrics as dict for MLOps reporting
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_fraud": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall_fraud": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "f1_fraud": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained XGBoost fraud detector.")
    parser.add_argument("--models", type=str, default="models", help="Directory where models are saved")
    parser.add_argument("--reports", type=str, default="reports", help="Directory to save report curves")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    args = parser.parse_args()
    
    evaluate_model(models_dir=args.models, reports_dir=args.reports, threshold=args.threshold)
