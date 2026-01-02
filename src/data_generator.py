"""
Synthetic Financial Transaction Data Generator
Generates realistic transaction data mimicking PayPal's ecosystem
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_transaction_data(n_transactions: int = 50000, fraud_rate: float = 0.013) -> pd.DataFrame:
    """
    Generate synthetic financial transaction data with realistic fraud patterns.
    
    Parameters:
    -----------
    n_transactions : int
        Number of transactions to generate
    fraud_rate : float
        Target fraud rate (default 1.3% - industry average)
    
    Returns:
    --------
    pd.DataFrame
        Transaction dataset with features and fraud labels
    """
    np.random.seed(42)
    random.seed(42)
    
    # Calculate fraud/legit split
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud
    
    # Transaction types (PayPal-relevant)
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    type_weights = [0.35, 0.25, 0.20, 0.15, 0.05]
    fraud_type_weights = [0.15, 0.30, 0.45, 0.08, 0.02]  # Fraudsters prefer cash-out
    
    # Merchant categories
    merchant_categories = [
        'retail', 'dining', 'entertainment', 'travel', 'utilities',
        'healthcare', 'education', 'crypto', 'gambling', 'electronics'
    ]
    risky_categories = ['crypto', 'gambling', 'electronics']
    
    # Generate legitimate transactions
    legit_data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_legit)],
        'amount': np.clip(np.random.lognormal(4.5, 1.2, n_legit), 1, 10000),
        'transaction_type': np.random.choice(transaction_types, n_legit, p=type_weights),
        'merchant_category': np.random.choice(merchant_categories, n_legit),
        'hour_of_day': np.random.choice(range(24), n_legit, p=_get_hour_distribution()),
        'day_of_week': np.random.randint(0, 7, n_legit),
        'account_age_days': np.random.exponential(365, n_legit).astype(int) + 30,
        'transaction_count_24h': np.random.poisson(2, n_legit),
        'avg_transaction_amount': np.clip(np.random.lognormal(4.5, 0.8, n_legit), 10, 5000),
        'is_international': np.random.choice([0, 1], n_legit, p=[0.92, 0.08]),
        'device_trust_score': np.clip(np.random.beta(8, 2, n_legit), 0.3, 1.0),
        'is_fraud': 0
    }
    
    # Generate fraudulent transactions with distinct patterns
    fraud_data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_legit, n_transactions)],
        'amount': np.clip(np.random.lognormal(6.0, 1.5, n_fraud), 100, 50000),  # Higher amounts
        'transaction_type': np.random.choice(transaction_types, n_fraud, p=fraud_type_weights),
        'merchant_category': np.random.choice(merchant_categories, n_fraud, 
                                              p=_get_risky_category_weights(merchant_categories, risky_categories)),
        'hour_of_day': np.random.choice(range(24), n_fraud, p=_get_fraud_hour_distribution()),
        'day_of_week': np.random.randint(0, 7, n_fraud),
        'account_age_days': np.random.exponential(60, n_fraud).astype(int) + 5,  # Newer accounts
        'transaction_count_24h': np.random.poisson(8, n_fraud),  # More transactions
        'avg_transaction_amount': np.clip(np.random.lognormal(3.5, 0.5, n_fraud), 10, 500),  # Different from current
        'is_international': np.random.choice([0, 1], n_fraud, p=[0.65, 0.35]),  # More international
        'device_trust_score': np.clip(np.random.beta(2, 5, n_fraud), 0.0, 0.7),  # Lower trust
        'is_fraud': 1
    }
    
    # Combine datasets
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    
    # Add derived features
    df['amount_to_avg_ratio'] = df['amount'] / df['avg_transaction_amount']
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_risky_category'] = df['merchant_category'].isin(risky_categories).astype(int)
    
    # Add timestamp
    base_date = datetime(2024, 1, 1)
    df['timestamp'] = [base_date + timedelta(
        days=random.randint(0, 365),
        hours=row['hour_of_day'],
        minutes=random.randint(0, 59)
    ) for _, row in df.iterrows()]
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def _get_hour_distribution():
    """Normal transaction hour distribution - peaks during business hours"""
    probs = np.array([
        0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM (low)
        0.03, 0.05, 0.07, 0.08, 0.09, 0.09,  # 6-11 AM (rising)
        0.08, 0.08, 0.07, 0.06, 0.06, 0.05,  # 12-5 PM (plateau)
        0.05, 0.04, 0.03, 0.03, 0.03, 0.02   # 6-11 PM (declining)
    ])
    return probs / probs.sum()


def _get_fraud_hour_distribution():
    """Fraud transaction hour distribution - more activity at odd hours"""
    probs = np.array([
        0.06, 0.07, 0.07, 0.06, 0.05, 0.04,  # 0-5 AM (higher than normal)
        0.03, 0.03, 0.03, 0.04, 0.04, 0.04,  # 6-11 AM 
        0.04, 0.04, 0.04, 0.04, 0.04, 0.05,  # 12-5 PM
        0.05, 0.05, 0.05, 0.05, 0.06, 0.06   # 6-11 PM (higher)
    ])
    return probs / probs.sum()


def _get_risky_category_weights(categories, risky):
    """Higher weights for risky merchant categories in fraud"""
    weights = []
    for cat in categories:
        if cat in risky:
            weights.append(0.25)
        else:
            weights.append(0.05)
    weights = np.array(weights)
    return weights / weights.sum()


def save_data(df: pd.DataFrame, output_path: str = 'data/transactions.csv'):
    """Save generated data to CSV"""
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to {output_path}")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"   Legit transactions: {(~df['is_fraud'].astype(bool)).sum():,}")


if __name__ == "__main__":
    print("🔄 Generating synthetic financial transaction data...")
    df = generate_transaction_data(n_transactions=50000)
    save_data(df, 'data/transactions.csv')
    
    # Print sample
    print("\n📊 Sample transactions:")
    print(df[['transaction_id', 'amount', 'transaction_type', 'merchant_category', 'is_fraud']].head(10))
