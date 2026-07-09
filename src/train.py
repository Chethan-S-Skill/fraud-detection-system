import os
import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from data_simulation import generate_synthetic_data
from preprocessing import FraudFeatureExtractor, PipelinePreprocessor

# Suppress Optuna logging to clean output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_model(data_path, models_dir, num_trials=10):
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Simulating data first...")
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # 2. Extract batch features
    print("Running feature extractor on batch data...")
    extractor = FraudFeatureExtractor()
    extractor.fit(df)
    df_feats = extractor.extract_batch_features(df)
    
    # Save the feature extractor state for real-time history caches
    extractor_path = os.path.join(models_dir, "feature_extractor.joblib")
    joblib.dump(extractor, extractor_path)
    print(f"Saved feature extractor cache to {extractor_path}")

    # 3. Train-test split (Chronological to prevent data leakage)
    df_feats = df_feats.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df_feats) * 0.8)
    
    train_df = df_feats.iloc[:split_idx].copy()
    test_df = df_feats.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows.")
    
    # Define columns to drop from feature set
    non_feature_cols = ['transaction_id', 'timestamp', 'user_id', 'device_id', 'is_fraud']
    
    X_train_raw = train_df.drop(columns=non_feature_cols)
    y_train = train_df['is_fraud'].values
    
    X_test_raw = test_df.drop(columns=non_feature_cols)
    y_test = test_df['is_fraud'].values
    
    # 4. Preprocessing Pipeline
    print("Fitting sklearn preprocessing pipeline...")
    preprocessor = PipelinePreprocessor()
    preprocessor.fit(X_train_raw)
    
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Save the fitted preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved fitted preprocessor to {preprocessor_path}")
    
    # Calculate scale_pos_weight for handling class imbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos
    print(f"Class distribution: {num_neg} Legitimate, {num_pos} Fraud. Imbalance weight: {scale_pos_weight:.2f}")
    
    # 5. Optuna Hyperparameter Optimization
    print(f"Starting Optuna hyperparameter tuning ({num_trials} trials)...")
    
    # We will optimize Average Precision (PR-AUC) as it focuses on minority class correctness
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'use_label_encoder': False
        }
        
        # Chronological cross validation
        # Split train set further into train/validation
        sub_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:sub_split], X_train.iloc[sub_split:]
        y_tr, y_val = y_train[:sub_split], y_train[sub_split:]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        
        # Predict probability
        probs = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, probs)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_trials)
    
    print("\nOptuna Hyperparameter Tuning Complete!")
    print(f"Best Trial Score (Validation Average Precision): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
        
    study_path = os.path.join(models_dir, "optuna_study.joblib")
    joblib.dump(study, study_path)
    
    # 6. Train final model with best parameters on full train set
    print("\nTraining final model with best hyperparameters...")
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_ap = average_precision_score(y_test, test_probs)
    print(f"Test Set Average Precision (PR-AUC): {test_ap:.4f}")
    
    # Save final model
    model_path = os.path.join(models_dir, "xgboost_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"Saved trained XGBoost model to {model_path}")
    
    # Save split test datasets for evaluation script
    joblib.dump((X_test, y_test), os.path.join(models_dir, "test_data.joblib"))
    print("Saved test data partitions to models/test_data.joblib")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detector with Optuna.")
    parser.add_argument("--data", type=str, default="data/transactions.csv", help="Input data path")
    parser.add_argument("--models", type=str, default="models", help="Directory to save models")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials")
    args = parser.parse_args()
    
    train_model(data_path=args.data, models_dir=args.models, num_trials=args.trials)
