"""
Feature Engineering Pipeline for Fraud Detection
Implements industry best practices for financial fraud feature creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List


class FraudFeatureEngineer:
    """
    Feature engineering pipeline for financial fraud detection.
    Implements PayPal-relevant feature transformations.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.engineered_features: List[str] = []
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix
        """
        df = df.copy()
        
        # Step 1: Create velocity-based features
        df = self._create_velocity_features(df)
        
        # Step 2: Create amount-based anomaly features
        df = self._create_amount_features(df)
        
        # Step 3: Create time-based features
        df = self._create_time_features(df)
        
        # Step 4: Create risk score features
        df = self._create_risk_features(df)
        
        # Step 5: Encode categorical variables
        df = self._encode_categoricals(df)
        
        # Step 6: Scale numerical features
        df = self._scale_numericals(df)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        df = df.copy()
        df = self._create_velocity_features(df)
        df = self._create_amount_features(df)
        df = self._create_time_features(df)
        df = self._create_risk_features(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numericals(df, fit=False)
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction velocity features.
        High transaction frequency is a strong fraud indicator.
        """
        # Transaction velocity score
        df['velocity_score'] = df['transaction_count_24h'] / (df['account_age_days'] / 30 + 1)
        
        # Abnormal velocity flag
        velocity_threshold = df['velocity_score'].quantile(0.95)
        df['high_velocity_flag'] = (df['velocity_score'] > velocity_threshold).astype(int)
        
        self.engineered_features.extend(['velocity_score', 'high_velocity_flag'])
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based anomaly features.
        Unusual amounts relative to account history indicate fraud.
        """
        # Amount deviation from average
        df['amount_deviation'] = np.abs(df['amount'] - df['avg_transaction_amount'])
        df['amount_zscore'] = (df['amount'] - df['avg_transaction_amount']) / (df['avg_transaction_amount'] + 1)
        
        # Log-transformed amount (reduces skewness)
        df['log_amount'] = np.log1p(df['amount'])
        
        # Amount percentile within dataset
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Large transaction flag (>95th percentile)
        amount_threshold = df['amount'].quantile(0.95)
        df['large_transaction_flag'] = (df['amount'] > amount_threshold).astype(int)
        
        self.engineered_features.extend([
            'amount_deviation', 'amount_zscore', 'log_amount', 
            'amount_percentile', 'large_transaction_flag'
        ])
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        Fraud often occurs during off-peak hours.
        """
        # Cyclical encoding for hour (captures periodicity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Business hours flag (9 AM - 5 PM)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        self.engineered_features.extend([
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_business_hours'
        ])
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk score features.
        Combine multiple risk signals into unified scores.
        """
        # Account age risk (newer accounts = higher risk)
        df['account_age_risk'] = 1 / (np.log1p(df['account_age_days']) + 1)
        
        # Device risk (inverse of trust score)
        df['device_risk'] = 1 - df['device_trust_score']
        
        # Composite risk score
        df['composite_risk_score'] = (
            df['account_age_risk'] * 0.2 +
            df['device_risk'] * 0.3 +
            df['is_international'] * 0.2 +
            df['is_risky_category'] * 0.15 +
            df['is_night_transaction'] * 0.15
        )
        
        # Interaction features
        df['amount_x_velocity'] = df['amount_to_avg_ratio'] * df['velocity_score']
        df['risk_x_amount'] = df['composite_risk_score'] * df['log_amount']
        
        self.engineered_features.extend([
            'account_age_risk', 'device_risk', 'composite_risk_score',
            'amount_x_velocity', 'risk_x_amount'
        ])
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding"""
        categorical_cols = ['transaction_type', 'merchant_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    
        self.categorical_features = [f'{col}_encoded' for col in categorical_cols]
        return df
    
    def _scale_numericals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        # Define numerical columns to scale
        numerical_cols = [
            'amount', 'account_age_days', 'transaction_count_24h',
            'avg_transaction_amount', 'device_trust_score', 'amount_to_avg_ratio',
            'velocity_score', 'amount_deviation', 'amount_zscore', 'log_amount',
            'amount_percentile', 'account_age_risk', 'device_risk',
            'composite_risk_score', 'amount_x_velocity', 'risk_x_amount'
        ]
        
        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        self.numerical_features = numerical_cols
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names for model training"""
        binary_features = [
            'is_international', 'is_weekend', 'is_night_transaction', 
            'is_risky_category', 'high_velocity_flag', 'large_transaction_flag',
            'is_business_hours'
        ]
        cyclical_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        return (
            self.numerical_features + 
            binary_features + 
            cyclical_features + 
            self.categorical_features
        )


def prepare_data_for_training(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare data for model training.
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_names : List[str]
        List of feature column names
    """
    engineer = FraudFeatureEngineer()
    df_transformed = engineer.fit_transform(df)
    
    # Get feature columns
    feature_names = engineer.get_feature_names()
    
    # Filter to existing columns
    feature_names = [f for f in feature_names if f in df_transformed.columns]
    
    X = df_transformed[feature_names]
    y = df_transformed['is_fraud']
    
    return X, y, feature_names


if __name__ == "__main__":
    # Test feature engineering
    print("🔄 Testing feature engineering pipeline...")
    
    # Load sample data
    df = pd.read_csv('data/transactions.csv')
    print(f"   Loaded {len(df):,} transactions")
    
    # Apply feature engineering
    X, y, features = prepare_data_for_training(df)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Features created: {len(features)}")
    print(f"   Feature names: {features}")
    print(f"\n   X shape: {X.shape}")
    print(f"   y distribution:\n{y.value_counts()}")
