import os
import joblib
import numpy as np
import pandas as pd
import shap

class FraudSHAPExplainer:
    """
    Integrates SHAP explainability into the production inference pipeline.
    Calculates exact feature contributions for individual transactions.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
            
        self.model = joblib.load(model_path)
        
        # Initialize TreeExplainer (optimized for tree-based models like XGBoost)
        # We pass a background dataset or run it directly. TreeExplainer can compute
        # tree-dependent SHAP values without an explicit background dataset, making it extremely fast.
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_transaction(self, X_preprocessed):
        """
        Calculates feature contributions for a single transaction.
        X_preprocessed: pd.DataFrame with 1 row, containing preprocessed features.
        """
        # Ensure input is a DataFrame with correct feature names
        if not isinstance(X_preprocessed, pd.DataFrame):
            raise ValueError("Input X_preprocessed must be a pandas DataFrame.")
            
        # Compute SHAP values
        shap_values_obj = self.explainer(X_preprocessed)
        
        # Extract values
        shap_vals = shap_values_obj.values[0]
        base_val = shap_values_obj.base_values[0]
        
        # Handle cases where SHAP returns multi-dimensional output for binary classification (multiclass structure)
        if isinstance(base_val, np.ndarray) and len(base_val) > 1:
            base_val = base_val[1]
            shap_vals = shap_vals[:, 1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) > 1 and shap_vals.shape[-1] == 2:
            # If shape is (num_features, 2)
            shap_vals = shap_vals[:, 1]
            if isinstance(base_val, np.ndarray):
                base_val = base_val[1]
                
        feature_names = X_preprocessed.columns.tolist()
        
        contributions = []
        for name, val in zip(feature_names, shap_vals):
            # Map raw scaling feature names to reader-friendly terms for the UI
            clean_name = self._get_friendly_feature_name(name)
            contributions.append({
                "raw_feature": name,
                "feature": clean_name,
                "shap_value": float(val)
            })
            
        # Sort contributions by absolute impact (highest first)
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Calculate approximation of final score in probability space
        # Log-odds = base_value + sum(shap_values)
        # Probability = 1 / (1 + exp(-log_odds))
        log_odds_base = float(base_val)
        log_odds_total = log_odds_base + sum(shap_vals)
        prob_base = 1.0 / (1.0 + np.exp(-log_odds_base))
        prob_total = 1.0 / (1.0 + np.exp(-log_odds_total))
        
        return {
            "base_value_log_odds": log_odds_base,
            "total_value_log_odds": log_odds_total,
            "base_probability": float(prob_base),
            "predicted_probability": float(prob_total),
            "contributions": contributions
        }
        
    def _get_friendly_feature_name(self, name):
        """
        Converts machine-readable feature names into clear business terms.
        """
        name_mapping = {
            'amount': 'Transaction Amount ($)',
            'distance_from_home': 'Distance from Home (km)',
            'hour_of_day': 'Hour of Day (0-23)',
            'day_of_week': 'Day of Week (0-6)',
            'txn_count_1h': 'Transaction Count (Last 1 Hour)',
            'spend_sum_1h': 'Cumulative Spend (Last 1 Hour)',
            'txn_count_24h': 'Transaction Count (Last 24 Hours)',
            'spend_sum_24h': 'Cumulative Spend (Last 24 Hours)',
            'card_present': 'Card Physically Present',
            'is_new_device': 'Device ID Mismatched',
        }
        
        # For one-hot encoded columns (e.g., merchant_category_transfer)
        if name.startswith('merchant_category_'):
            cat = name.replace('merchant_category_', '').title()
            return f"Merchant Category: {cat}"
            
        return name_mapping.get(name, name.title())

if __name__ == '__main__':
    # Test explainability
    print("Testing SHAP Explainer class load...")
    import sys
    try:
        explainer = FraudSHAPExplainer("models/xgboost_model.joblib")
        print("Successfully loaded explainer!")
    except Exception as e:
        print(f"Error (expected if models are not trained yet): {e}")
