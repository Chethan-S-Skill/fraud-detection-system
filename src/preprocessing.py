import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class FraudFeatureExtractor:
    """
    Computes rolling window features and behavioral flags from raw transaction data.
    Suitable for both batch training and real-time inference.
    """
    def __init__(self):
        self.user_history = {} # In-memory cache for user transactions to calculate rolling stats in real-time
        self.user_seen_devices = {}

    def fit(self, df):
        # Sort data to ensure chronological rolling calculations
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        
        # Populate history and device caches for all users in training set
        for _, row in df_sorted.iterrows():
            uid = row['user_id']
            tid = row['transaction_id']
            t_time = row['timestamp']
            amt = row['amount']
            dev = row['device_id']
            
            if uid not in self.user_history:
                self.user_history[uid] = []
            self.user_history[uid].append({'timestamp': t_time, 'amount': amt})
            
            if uid not in self.user_seen_devices:
                self.user_seen_devices[uid] = set()
            self.user_seen_devices[uid].add(dev)
            
        return self

    def extract_batch_features(self, df):
        """
        Extracts rolling window and time-based features on a batch dataframe.
        """
        df_feats = df.sort_values('timestamp').copy()
        df_feats['timestamp'] = pd.to_datetime(df_feats['timestamp'])
        
        # Time components
        df_feats['hour_of_day'] = df_feats['timestamp'].dt.hour
        df_feats['day_of_week'] = df_feats['timestamp'].dt.dayofweek
        
        # Rolling features
        print("Calculating rolling window stats (1h and 24h)...")
        
        # -------------------------------------------------------------------------
 # 1-hour rolling features (inclusive of current row)
        roll_1h = df_feats.groupby('user_id').rolling('1h', on='timestamp')['amount'].agg(
            txn_count_1h='count',
            spend_sum_1h='sum'
        ).reset_index(level=0, drop=True)
        
        # 24-hour rolling features (inclusive of current row)
        roll_24h = df_feats.groupby('user_id').rolling('24h', on='timestamp')['amount'].agg(
            txn_count_24h='count',
            spend_sum_24h='sum'
        ).reset_index(level=0, drop=True)
        
        df_feats['txn_count_1h'] = roll_1h['txn_count_1h']
        df_feats['spend_sum_1h'] = roll_1h['spend_sum_1h']
        df_feats['txn_count_24h'] = roll_24h['txn_count_24h']
        df_feats['spend_sum_24h'] = roll_24h['spend_sum_24h']
        
        

        
        # Device mismatch feature (is new device)
        print("Computing device mismatch features...")
        is_new_device = []
        seen_devices_by_user = {}
        
        for _, row in df_feats.iterrows():
            uid = row['user_id']
            dev = row['device_id']
            if uid not in seen_devices_by_user:
                seen_devices_by_user[uid] = set([dev])
                is_new_device.append(0)
            else:
                if dev not in seen_devices_by_user[uid]:
                    is_new_device.append(1)
                    seen_devices_by_user[uid].add(dev)
                else:
                    is_new_device.append(0)
                    
        df_feats['is_new_device'] = is_new_device
        
        return df_feats

    def extract_single_feature(self, txn, update_history=True):
        """
        Extracts rolling window and time features for a single transaction (real-time).
        txn: dict containing keys: user_id, timestamp, amount, device_id, etc.
        """
        uid = txn['user_id']
        t_time = pd.to_datetime(txn['timestamp'])
        amt = float(txn['amount'])
        dev = txn['device_id']
        
        # Initialize user cache if missing
        if uid not in self.user_history:
            self.user_history[uid] = []
        if uid not in self.user_seen_devices:
            self.user_seen_devices[uid] = set()
            
        # Calculate device mismatch before adding new device to set
        is_new_dev = 1 if len(self.user_seen_devices[uid]) > 0 and dev not in self.user_seen_devices[uid] else 0
        
        # Filter history for rolling windows
        one_hour_ago = t_time - pd.Timedelta(hours=1)
        twenty_four_hours_ago = t_time - pd.Timedelta(days=1)
        
        txn_1h = [t for t in self.user_history[uid] if t['timestamp'] >= one_hour_ago and t['timestamp'] <= t_time]
        txn_24h = [t for t in self.user_history[uid] if t['timestamp'] >= twenty_four_hours_ago and t['timestamp'] <= t_time]
        
        # Compute stats (including current transaction details)
        txn_count_1h = len(txn_1h) + 1
        spend_sum_1h = sum([t['amount'] for t in txn_1h]) + amt
        
        txn_count_24h = len(txn_24h) + 1
        spend_sum_24h = sum([t['amount'] for t in txn_24h]) + amt
        
        # Update cache in memory if requested
        if update_history:
            self.user_history[uid].append({'timestamp': t_time, 'amount': amt})
            self.user_seen_devices[uid].add(dev)
            
            # Prune cache to keep only past 24 hours of data to prevent memory leaks
            self.user_history[uid] = [t for t in self.user_history[uid] if t['timestamp'] >= t_time - pd.Timedelta(days=1)]
            
        # Prepare output features dictionary
        output = txn.copy()
        output['hour_of_day'] = t_time.hour
        output['day_of_week'] = t_time.dayofweek
        output['txn_count_1h'] = txn_count_1h
        output['spend_sum_1h'] = spend_sum_1h
        output['txn_count_24h'] = txn_count_24h
        output['spend_sum_24h'] = spend_sum_24h
        output['is_new_device'] = is_new_dev
        
        return output


class PipelinePreprocessor(BaseEstimator, TransformerMixin):
    """
    Handles scaling, imputation, and categorical encoding using standard sklearn primitives.
    """
    def __init__(self):
        self.num_cols = [
            'amount', 'distance_from_home', 'hour_of_day', 'day_of_week',
            'txn_count_1h', 'spend_sum_1h', 'txn_count_24h', 'spend_sum_24h'
        ]
        self.cat_cols = ['merchant_category']
        self.passthrough_cols = ['card_present', 'is_new_device']
        
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.feature_names_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        # Fit numerical
        self.num_imputer.fit(X_df[self.num_cols])
        X_num = self.num_imputer.transform(X_df[self.num_cols])
        self.scaler.fit(X_num)
        
        # Fit categorical
        self.encoder.fit(X_df[self.cat_cols])
        
        # Store feature names
        enc_cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_ = self.num_cols + self.passthrough_cols + enc_cat_names
        
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        
        # Transform numerical
        X_num = self.num_imputer.transform(X_df[self.num_cols])
        X_num_scaled = self.scaler.transform(X_num)
        
        # Transform categorical
        X_cat_encoded = self.encoder.transform(X_df[self.cat_cols])
        
        # Passthrough features
        X_pass = X_df[self.passthrough_cols].values
        
        # Concatenate columns
        X_out = np.hstack([X_num_scaled, X_pass, X_cat_encoded])
        return pd.DataFrame(X_out, columns=self.feature_names_)
