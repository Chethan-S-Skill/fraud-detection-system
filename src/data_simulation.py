import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=50000, num_users=1000, random_seed=42):
    """
    Generates a highly imbalanced synthetic transaction dataset for fraud detection.
    """
    np.random.seed(random_seed)
    
    print(f"Generating {num_records} transaction records for {num_users} users...")
    
    # 1. Generate User Profiles
    user_ids = [f"USR_{i:04d}" for i in range(num_users)]
    user_home_lat = np.random.uniform(25.0, 49.0, size=num_users)  # US latitude range roughly
    user_home_lon = np.random.uniform(-125.0, -70.0, size=num_users)  # US longitude range roughly
    
    # Typical device for each user
    user_devices = [f"DEV_{np.random.randint(1000, 9999)}" for _ in range(num_users)]
    
    user_profiles = pd.DataFrame({
        'user_id': user_ids,
        'home_lat': user_home_lat,
        'home_lon': user_home_lon,
        'primary_device': user_devices
    })
    
    # 2. Generate Transaction Core details
    # Start date of transaction simulation
    start_date = datetime(2026, 6, 1, 0, 0, 0)
    
    # Generate random timestamps
    random_offsets = np.random.randint(0, 30 * 24 * 3600, size=num_records)  # 30 days of transactions
    timestamps = [start_date + timedelta(seconds=int(offset)) for offset in random_offsets]
    
    # Associate transactions with users (power law or zipf distribution of transactions per user)
    # Some users are very active, some are not
    user_idx = np.random.zipf(a=1.5, size=num_records)
    user_idx = np.clip(user_idx - 1, 0, num_users - 1)
    chosen_users = [user_ids[idx] for idx in user_idx]
    
    # Amount distribution: Log-normal (mostly small purchases, occasionally large ones)
    amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=num_records)
    amounts = np.round(amounts, 2)
    amounts = np.clip(amounts, 1.0, 15000.0)  # Keep transaction amount between $1 and $15000
    
    # Merchant categories
    categories = ['grocery', 'gas_station', 'dining', 'e_commerce', 'travel', 'electronics', 'entertainment', 'transfer']
    category_probs = [0.35, 0.15, 0.20, 0.15, 0.03, 0.05, 0.05, 0.02]
    chosen_categories = np.random.choice(categories, size=num_records, p=category_probs)
    
    # Combine core variables
    df = pd.DataFrame({
        'transaction_id': [f"TXN_{i:07d}" for i in range(num_records)],
        'timestamp': timestamps,
        'user_id': chosen_users,
        'amount': amounts,
        'merchant_category': chosen_categories
    })
    
    # Merge with user profile data to add home location and primary device
    df = df.merge(user_profiles, on='user_id', how='left')
    
    # 3. Simulate Transaction Context features
    # Device ID: 92% use primary device, 8% use a new device
    device_mismatch = np.random.rand(num_records) > 0.92
    df['device_id'] = df['primary_device']
    new_devices = [f"DEV_{np.random.randint(1000, 9999)}" for _ in range(sum(device_mismatch))]
    df.loc[device_mismatch, 'device_id'] = new_devices
    
    # Card present indicator (e-commerce/transfer are usually card_not_present)
    card_present = []
    for cat in df['merchant_category']:
        if cat in ['e_commerce', 'transfer']:
            card_present.append(0)
        elif cat in ['grocery', 'gas_station']:
            card_present.append(1)
        else:
            card_present.append(int(np.random.rand() > 0.3))
    df['card_present'] = card_present
    
    # Location details: 95% near home, 5% far away (mismatch)
    loc_mismatch = np.random.rand(num_records) > 0.95
    df['user_lat'] = df['home_lat'] + np.random.normal(0, 0.05, size=num_records)
    df['user_lon'] = df['home_lon'] + np.random.normal(0, 0.05, size=num_records)
    
    # For mismatched, make them far away
    df.loc[loc_mismatch, 'user_lat'] = df.loc[loc_mismatch, 'user_lat'] + np.random.uniform(-10.0, 10.0, size=sum(loc_mismatch))
    df.loc[loc_mismatch, 'user_lon'] = df.loc[loc_mismatch, 'user_lon'] + np.random.uniform(-20.0, 20.0, size=sum(loc_mismatch))
    
    # Calculate distance from home (approximate using simple Euclidean distance for performance, or Haversine)
    # We will use Haversine distance in python
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km
    
    df['distance_from_home'] = haversine_np(df['home_lon'], df['home_lat'], df['user_lon'], df['user_lat'])
    
    # 4. Define Fraud Probability Risk Rules to determine label
    # This ensures fraud is not random but follows logical features that XGBoost can discover.
    fraud_prob = np.zeros(num_records)
    
    # Rule 1: High Transaction Amount
    # Very high amount increases risk
    fraud_prob += np.where(df['amount'] > 2000, 0.20, 0.0)
    fraud_prob += np.where(df['amount'] > 8000, 0.40, 0.0)
    
    # Rule 2: Night Transactions (between 1 AM and 5 AM)
    df_hours = pd.DatetimeIndex(df['timestamp']).hour
    is_night = (df_hours >= 1) & (df_hours <= 5)
    fraud_prob += np.where(is_night, 0.08, 0.0)
    
    # Rule 3: Mismatched location (distance > 300 km)
    fraud_prob += np.where(df['distance_from_home'] > 300.0, 0.15, 0.0)
    
    # Rule 4: Device mismatch combined with Card Not Present
    is_mismatched_cnp = (device_mismatch) & (df['card_present'] == 0)
    fraud_prob += np.where(is_mismatched_cnp, 0.25, 0.0)
    
    # Rule 5: Transfer / Travel categories have slightly higher base risk
    fraud_prob += np.where(df['merchant_category'] == 'transfer', 0.05, 0.0)
    fraud_prob += np.where((df['merchant_category'] == 'travel') & (df['amount'] > 1000), 0.12, 0.0)
    
    # Base risk is low
    fraud_prob += 0.002
    
    # Clip prob to [0, 0.98]
    fraud_prob = np.clip(fraud_prob, 0.0, 0.98)
    
    # Assign target labels based on probabilities
    # We want a target of roughly 1% fraud, so we'll adjust the Bernoulli trials or use top 1.2%
    # or just use random binomial with the probabilities and then scale if it's too high/low.
    raw_labels = np.random.binomial(1, fraud_prob)
    
    # Force target imbalance to be exactly ~1% by selecting the top-k highest probability items if needed,
    # or simply adjusting. Let's just sample with fraud_prob but scale probabilities so the mean is ~0.01.
    scale_factor = 0.012 / np.mean(raw_labels) if np.mean(raw_labels) > 0 else 1.0
    adjusted_prob = np.clip(fraud_prob * scale_factor, 0.0, 0.99)
    labels = np.random.binomial(1, adjusted_prob)
    
    df['is_fraud'] = labels
    
    # Clean up intermediate columns used for profile generation
    df = df.drop(columns=['home_lat', 'home_lon', 'primary_device'])
    
    # Sort by timestamp so it mimics a live chronological transaction stream
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"Dataset generated. Fraud Rate: {df['is_fraud'].mean() * 100:.2f}% ({df['is_fraud'].sum()} fraud out of {len(df)} transactions)")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data for fraud detection.")
    parser.add_argument("--records", type=int, default=50000, help="Number of records to generate")
    parser.add_argument("--users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--output", type=str, default="data/transactions.csv", help="Output CSV path")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = generate_synthetic_data(num_records=args.records, num_users=args.users)
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
