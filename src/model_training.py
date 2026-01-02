"""
Model Training Pipeline for Fraud Detection
Implements stacking ensemble with XGBoost, LightGBM, and CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# CatBoost removed due to Python 3.14 compatibility issues
# from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
import sys
sys.path.append('.')
from src.feature_engineering import prepare_data_for_training


class FraudDetectionModel:
    """
    Production-ready fraud detection model with ensemble learning.
    """
    
    def __init__(self, use_resampling: bool = True):
        self.use_resampling = use_resampling
        self.models = {}
        self.ensemble = None
        self.best_threshold = 0.5
        self.feature_importances = None
        
    def _get_base_models(self):
        """
        Initialize base models with fraud-optimized hyperparameters.
        """
        # Calculate class weight for imbalanced data
        # PayPal context: False negatives (missed fraud) cost ~10x false positives
        
        from sklearn.ensemble import GradientBoostingClassifier
        
        models = {
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=10,  # Handle imbalance
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        }
        return models
    
    def _create_stacking_ensemble(self):
        """
        Create stacking ensemble with logistic regression meta-learner.
        """
        base_models = self._get_base_models()
        
        estimators = [
            ('xgboost', base_models['xgboost']),
            ('lightgbm', base_models['lightgbm']),
            ('gradient_boosting', base_models['gradient_boosting']),
        ]
        
        # Stacking with logistic regression meta-learner
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking, base_models
    
    def _apply_resampling(self, X: np.ndarray, y: np.ndarray):
        """
        Apply SMOTE-ENN hybrid resampling for imbalanced data.
        SMOTE generates synthetic minority samples.
        ENN removes noisy majority samples near decision boundary.
        """
        print("   Applying SMOTE-ENN resampling...")
        print(f"   Original class distribution: {np.bincount(y.astype(int))}")
        
        # Use SMOTE-ENN for best performance on fraud detection
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5),
            random_state=42,
            n_jobs=-1
        )
        
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        print(f"   Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
        
        return X_resampled, y_resampled
    
    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: list):
        """
        Train the fraud detection ensemble model.
        """
        print("\n" + "="*60)
        print("🚀 TRAINING FRAUD DETECTION MODEL")
        print("="*60)
        
        # Convert to numpy
        X_np = X.values
        y_np = y.values
        
        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_np
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        print(f"   Fraud rate (train): {y_train.mean()*100:.2f}%")
        
        # Apply resampling if enabled
        if self.use_resampling:
            X_train_res, y_train_res = self._apply_resampling(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        
        # Create ensemble
        print("\n🔧 Building Stacking Ensemble...")
        self.ensemble, self.models = self._create_stacking_ensemble()
        
        # Train individual models for comparison
        print("\n📈 Training Individual Models...")
        results = {}
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            model.fit(X_train_res, y_train_res)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            
            results[name] = {
                'auc_roc': auc,
                'avg_precision': ap,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            print(f"      AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
        
        # Train stacking ensemble
        print("\n🏆 Training Stacking Ensemble...")
        self.ensemble.fit(X_train_res, y_train_res)
        
        # Ensemble predictions
        y_pred_proba_ensemble = self.ensemble.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold using precision-recall curve
        self.best_threshold = self._find_optimal_threshold(y_test, y_pred_proba_ensemble)
        y_pred_ensemble = (y_pred_proba_ensemble >= self.best_threshold).astype(int)
        
        # Ensemble metrics
        auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
        ap_ensemble = average_precision_score(y_test, y_pred_proba_ensemble)
        
        results['stacking_ensemble'] = {
            'auc_roc': auc_ensemble,
            'avg_precision': ap_ensemble,
            'predictions': y_pred_ensemble,
            'probabilities': y_pred_proba_ensemble
        }
        
        print(f"   Stacking Ensemble AUC-ROC: {auc_ensemble:.4f}")
        print(f"   Stacking Ensemble Avg Precision: {ap_ensemble:.4f}")
        print(f"   Optimal Threshold: {self.best_threshold:.3f}")
        
        # Calculate feature importances from XGBoost
        self.feature_importances = dict(zip(
            feature_names, 
            self.models['xgboost'].feature_importances_
        ))
        
        # Store results
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        
        return results
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Find optimal classification threshold based on F1 score.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find threshold with best F1
        best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        
        return thresholds[best_idx]
    
    def evaluate(self) -> dict:
        """
        Generate comprehensive evaluation metrics.
        """
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        # Get ensemble predictions
        y_pred = self.results['stacking_ensemble']['predictions']
        y_pred_proba = self.results['stacking_ensemble']['probabilities']
        
        # Classification report
        print("\n📋 Classification Report (Stacking Ensemble):")
        print(classification_report(self.y_test, y_pred, 
                                    target_names=['Legitimate', 'Fraud']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0,0]:,}")
        print(f"   False Positives: {cm[0,1]:,}")
        print(f"   False Negatives: {cm[1,0]:,}")
        print(f"   True Positives:  {cm[1,1]:,}")
        
        # Business metrics
        total_fraud = self.y_test.sum()
        detected_fraud = cm[1,1]
        false_alarms = cm[0,1]
        
        print("\n💰 Business Metrics:")
        print(f"   Fraud Detection Rate: {detected_fraud/total_fraud*100:.1f}%")
        print(f"   False Alarm Rate: {false_alarms/(cm[0,0]+cm[0,1])*100:.2f}%")
        
        # Model comparison
        print("\n🏅 Model Comparison (AUC-ROC):")
        for name, res in sorted(self.results.items(), 
                                key=lambda x: x[1]['auc_roc'], 
                                reverse=True):
            print(f"   {name:20s}: {res['auc_roc']:.4f}")
        
        # Top features
        print("\n🔑 Top 10 Most Important Features:")
        sorted_features = sorted(
            self.feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for i, (feat, imp) in enumerate(sorted_features, 1):
            print(f"   {i:2d}. {feat:30s}: {imp:.4f}")
        
        return {
            'confusion_matrix': cm,
            'auc_roc': self.results['stacking_ensemble']['auc_roc'],
            'feature_importances': self.feature_importances
        }
    
    def save_model(self, path: str = 'models/ensemble_model.pkl'):
        """Save trained model to disk"""
        joblib.dump({
            'ensemble': self.ensemble,
            'best_threshold': self.best_threshold,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances
        }, path)
        print(f"\n✅ Model saved to {path}")
    
    def load_model(self, path: str = 'models/ensemble_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(path)
        self.ensemble = model_data['ensemble']
        self.best_threshold = model_data['best_threshold']
        self.feature_names = model_data['feature_names']
        self.feature_importances = model_data['feature_importances']
        print(f"✅ Model loaded from {path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        proba = self.ensemble.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for new data"""
        return self.ensemble.predict_proba(X)[:, 1]


def generate_performance_report(model: FraudDetectionModel, output_path: str = 'reports/model_performance.xlsx'):
    """
    Generate Excel report with model performance metrics.
    """
    # Model comparison
    model_comparison = pd.DataFrame([
        {
            'Model': name,
            'AUC-ROC': res['auc_roc'],
            'Average Precision': res['avg_precision']
        }
        for name, res in model.results.items()
    ]).sort_values('AUC-ROC', ascending=False)
    
    # Feature importance
    feature_importance = pd.DataFrame([
        {'Feature': name, 'Importance': imp}
        for name, imp in sorted(
            model.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ])
    
    # Confusion matrix
    cm = confusion_matrix(model.y_test, model.results['stacking_ensemble']['predictions'])
    cm_df = pd.DataFrame(
        cm,
        index=['Actual: Legitimate', 'Actual: Fraud'],
        columns=['Predicted: Legitimate', 'Predicted: Fraud']
    )
    
    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        model_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
        feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
    
    print(f"\n✅ Performance report saved to {output_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("🔐 PAYPAL FRAUD DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n📂 Loading transaction data...")
    df = pd.read_csv('data/transactions.csv')
    print(f"   Loaded {len(df):,} transactions")
    
    # Feature engineering
    print("\n🔧 Applying feature engineering...")
    X, y, feature_names = prepare_data_for_training(df)
    print(f"   Created {len(feature_names)} features")
    
    # Train model
    model = FraudDetectionModel(use_resampling=True)
    model.train(X, y, feature_names)
    
    # Evaluate
    model.evaluate()
    
    # Save model
    model.save_model('models/ensemble_model.pkl')
    
    # Generate report
    generate_performance_report(model)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()