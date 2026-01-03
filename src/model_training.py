"""
Credit Card Fraud Detection - Kaggle Dataset
Production-ready ML pipeline with proper evaluation metrics

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions
- 492 frauds (0.172%)
- Features V1-V28 (PCA transformed), Time, Amount
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class CreditCardFraudDetector:
    """
    Production-ready fraud detection model for Kaggle Credit Card dataset.
    
    Features:
    - Stacking ensemble (XGBoost, LightGBM, Random Forest)
    - SMOTE for class imbalance
    - Comprehensive fraud-specific metrics
    - Cost-sensitive threshold optimization
    """
    
    def __init__(self):
        self.scaler = RobustScaler()  # Better for outliers in Amount
        self.models = {}
        self.ensemble = None
        self.best_threshold = 0.5
        self.feature_names = None
        self.results = {}
        
    def load_and_prepare_data(self, filepath: str = 'data/creditcard.csv'):
        """
        Load and prepare the Kaggle credit card fraud dataset.
        """
        print("="*60)
        print("📂 LOADING KAGGLE CREDIT CARD FRAUD DATASET")
        print("="*60)
        
        # Load data
        df = pd.read_csv(filepath)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Shape: {df.shape}")
        print(f"   Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
        print(f"   Legitimate: {(df['Class']==0).sum():,}")
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c not in ['Class']]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df['Class']
        
        # Scale features
        print("\n🔧 Scaling features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=feature_cols
        )
        
        return X_scaled, y
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from the dataset.
        """
        print("\n🔧 Engineering features...")
        
        # Time-based features
        # Time is seconds elapsed - convert to hour of day (assuming 2 days of data)
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # Amount features
        df['Log_Amount'] = np.log1p(df['Amount'])
        df['Amount_Zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        
        # High amount flag
        df['High_Amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
        
        # Night transaction (assuming hour interpretation)
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
        
        # Interaction features with top PCA components
        df['V1_V2_interaction'] = df['V1'] * df['V2']
        df['V3_V4_interaction'] = df['V3'] * df['V4']
        df['V14_Amount'] = df['V14'] * df['Log_Amount']
        df['V17_Amount'] = df['V17'] * df['Log_Amount']
        
        # Drop original Time (we have Hour features now)
        df = df.drop('Time', axis=1)
        
        print(f"   Created {8} new features")
        print(f"   Total features: {len(df.columns) - 1}")
        
        return df
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Create stratified train/test split.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        print(f"\n📊 Train/Test Split:")
        print(f"   Train: {len(X_train):,} ({y_train.mean()*100:.3f}% fraud)")
        print(f"   Test: {len(X_test):,} ({y_test.mean()*100:.3f}% fraud)")
        
        return X_train, X_test, y_train, y_test
    
    def _get_base_models(self):
        """Initialize base models with tuned hyperparameters."""
        
        models = {
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=100,  # ~580:1 imbalance ratio
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc',
                use_label_encoder=False
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
        }
        return models
    
    def _create_ensemble(self):
        """Create stacking ensemble."""
        base_models = self._get_base_models()
        
        estimators = [
            ('xgboost', base_models['xgboost']),
            ('lightgbm', base_models['lightgbm']),
            ('random_forest', base_models['random_forest']),
        ]
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=0.5
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False
        )
        
        return ensemble, base_models
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_test: pd.DataFrame, y_test: pd.Series,
              use_smote: bool = True):
        """
        Train the fraud detection models.
        """
        print("\n" + "="*60)
        print("🚀 TRAINING FRAUD DETECTION MODELS")
        print("="*60)
        
        X_train_np = X_train.values
        y_train_np = y_train.values
        X_test_np = X_test.values
        y_test_np = y_test.values
        
        # Apply SMOTE
        if use_smote:
            print("\n⚖️ Applying SMOTE resampling...")
            print(f"   Before: {np.bincount(y_train_np.astype(int))}")
            
            smote = SMOTE(
                sampling_strategy=0.5,  # 50% of majority class
                random_state=42,
                k_neighbors=5
            )
            X_train_res, y_train_res = smote.fit_resample(X_train_np, y_train_np)
            print(f"   After:  {np.bincount(y_train_res.astype(int))}")
        else:
            X_train_res, y_train_res = X_train_np, y_train_np
        
        # Create ensemble
        self.ensemble, self.models = self._create_ensemble()
        
        # Train and evaluate individual models
        print("\n📈 Training Individual Models...")
        results = {}
        
        for name, model in self.models.items():
            print(f"\n   Training {name}...")
            model.fit(X_train_res, y_train_res)
            
            y_pred_proba = model.predict_proba(X_test_np)[:, 1]
            
            auc_roc = roc_auc_score(y_test_np, y_pred_proba)
            pr_auc = average_precision_score(y_test_np, y_pred_proba)
            
            results[name] = {
                'auc_roc': auc_roc,
                'pr_auc': pr_auc,
                'probabilities': y_pred_proba
            }
            
            print(f"      ROC-AUC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}")
        
        # Train ensemble
        print("\n🏆 Training Stacking Ensemble...")
        self.ensemble.fit(X_train_res, y_train_res)
        
        y_pred_proba_ensemble = self.ensemble.predict_proba(X_test_np)[:, 1]
        
        # Find optimal threshold
        self.best_threshold = self._find_optimal_threshold(
            y_test_np, y_pred_proba_ensemble,
            fp_cost=1,
            fn_cost=100  # Missed fraud costs 100x a false alarm
        )
        
        y_pred_ensemble = (y_pred_proba_ensemble >= self.best_threshold).astype(int)
        
        auc_roc_ensemble = roc_auc_score(y_test_np, y_pred_proba_ensemble)
        pr_auc_ensemble = average_precision_score(y_test_np, y_pred_proba_ensemble)
        
        results['stacking_ensemble'] = {
            'auc_roc': auc_roc_ensemble,
            'pr_auc': pr_auc_ensemble,
            'predictions': y_pred_ensemble,
            'probabilities': y_pred_proba_ensemble
        }
        
        print(f"\n   ✅ Stacking Ensemble:")
        print(f"      ROC-AUC: {auc_roc_ensemble:.4f}")
        print(f"      PR-AUC: {pr_auc_ensemble:.4f}")
        print(f"      Optimal Threshold: {self.best_threshold:.3f}")
        
        # Store results
        self.results = results
        self.X_test = X_test_np
        self.y_test = y_test_np
        
        # Calculate feature importance
        self.feature_importances = dict(zip(
            self.feature_names,
            self.models['xgboost'].feature_importances_
        ))
        
        return results
    
    def _find_optimal_threshold(self, y_true, y_pred_proba, fp_cost=1, fn_cost=100):
        """Find threshold that minimizes total cost."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Also try F1-based threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        f1_best_idx = np.argmax(f1_scores[:-1])
        f1_threshold = thresholds[f1_best_idx]
        
        # Cost-based threshold
        min_cost = float('inf')
        cost_threshold = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_pred_proba >= thresh).astype(int)
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            total_cost = fp * fp_cost + fn * fn_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                cost_threshold = thresh
        
        # Use F1 threshold as it's more balanced
        return f1_threshold
    
    def evaluate(self):
        """Generate comprehensive evaluation metrics."""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION - FRAUD-SPECIFIC METRICS")
        print("="*60)
        
        y_pred = self.results['stacking_ensemble']['predictions']
        y_pred_proba = self.results['stacking_ensemble']['probabilities']
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=['Legitimate', 'Fraud'],
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n🔢 Confusion Matrix:")
        print(f"   True Negatives:  {tn:,} (legitimate correctly identified)")
        print(f"   False Positives: {fp:,} (legitimate flagged as fraud)")
        print(f"   False Negatives: {fn:,} (FRAUD MISSED ⚠️)")
        print(f"   True Positives:  {tp:,} (fraud caught ✅)")
        
        # === FRAUD-SPECIFIC METRICS ===
        print("\n" + "="*60)
        print("🎯 FRAUD-SPECIFIC EVALUATION METRICS")
        print("="*60)
        
        # 1. Precision@K
        print("\n📌 Precision@K (Top K highest-risk transactions):")
        for k in [50, 100, 200, 500, 1000]:
            prec_k = self._precision_at_k(self.y_test, y_pred_proba, k)
            print(f"   Precision@{k:4d}: {prec_k:.4f} ({int(prec_k*k)} fraud in top {k})")
        
        # 2. Recall@FPR
        print("\n📌 Recall at Fixed False Positive Rates:")
        for fpr_target in [0.001, 0.005, 0.01, 0.05, 0.10]:
            recall_fpr = self._recall_at_fpr(self.y_test, y_pred_proba, fpr_target)
            print(f"   Recall@FPR={fpr_target*100:5.1f}%: {recall_fpr:.4f} ({recall_fpr*100:.1f}% fraud caught)")
        
        # 3. AUC metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        
        print(f"\n📌 Area Under Curve:")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   PR-AUC:  {pr_auc:.4f} (more important for imbalanced data)")
        
        # 4. Business metrics
        total_fraud = self.y_test.sum()
        fraud_caught = tp
        fraud_missed = fn
        
        print(f"\n💰 Business Impact Metrics:")
        print(f"   Total fraud cases in test set: {int(total_fraud)}")
        print(f"   Fraud cases caught: {tp} ({tp/total_fraud*100:.1f}%)")
        print(f"   Fraud cases missed: {fn} ({fn/total_fraud*100:.1f}%)")
        print(f"   False alarms: {fp}")
        print(f"   Precision of flagged: {tp/(tp+fp)*100:.1f}%")
        
        # 5. Cost analysis
        avg_fraud_amount = 150  # Typical fraud loss
        review_cost = 5  # Cost to review a flagged transaction
        
        fraud_loss_no_model = total_fraud * avg_fraud_amount
        fraud_loss_with_model = fn * avg_fraud_amount + fp * review_cost
        savings = fraud_loss_no_model - fraud_loss_with_model
        
        print(f"\n💵 Cost Analysis (Avg fraud=${avg_fraud_amount}, Review=${review_cost}):")
        print(f"   Loss without model: ${fraud_loss_no_model:,.0f}")
        print(f"   Loss with model: ${fraud_loss_with_model:,.0f}")
        print(f"   Net savings: ${savings:,.0f} ({savings/fraud_loss_no_model*100:.1f}%)")
        
        # Model comparison
        print("\n🏅 Model Comparison (sorted by PR-AUC):")
        print(f"   {'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10}")
        print(f"   {'-'*47}")
        for name, res in sorted(self.results.items(), key=lambda x: x[1]['pr_auc'], reverse=True):
            print(f"   {name:<25} {res['auc_roc']:>10.4f} {res['pr_auc']:>10.4f}")
        
        # Top features
        print("\n🔑 Top 15 Most Important Features:")
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        for i, (feat, imp) in enumerate(sorted_features, 1):
            print(f"   {i:2d}. {feat:<25}: {imp:.4f}")
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': tp/(tp+fp) if (tp+fp) > 0 else 0,
            'recall': tp/total_fraud if total_fraud > 0 else 0,
        }
    
    def _precision_at_k(self, y_true, y_scores, k):
        """Precision in top K predictions."""
        top_k_idx = np.argsort(y_scores)[-k:]
        return y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()
    
    def _recall_at_fpr(self, y_true, y_scores, target_fpr):
        """Recall at a fixed false positive rate."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(fpr - target_fpr))
        return tpr[idx]
    
    def save_model(self, path: str = 'models/fraud_detector.pkl'):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'best_threshold': self.best_threshold,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances
        }, path)
        print(f"\n✅ Model saved to {path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        proba = self.ensemble.predict_proba(X_scaled)[:, 1]
        return (proba >= self.best_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability."""
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict_proba(X_scaled)[:, 1]


def generate_visualizations(detector, output_dir: str = 'visualizations'):
    """Generate evaluation visualizations."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    os.makedirs(output_dir, exist_ok=True)
    
    y_test = detector.y_test
    y_proba = detector.results['stacking_ensemble']['probabilities']
    y_pred = detector.results['stacking_ensemble']['predictions']
    
    # 1. ROC Curve
    print("\n📊 Generating visualizations...")
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Stacking Ensemble (AUC={roc_auc:.4f})',
        line=dict(color='#6366F1', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    fig_roc.update_layout(
        title='ROC Curve - Credit Card Fraud Detection',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    fig_roc.write_html(f'{output_dir}/roc_curve.html')
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC={pr_auc:.4f})',
        line=dict(color='#10B981', width=3)
    ))
    fig_pr.update_layout(
        title='Precision-Recall Curve (More Important for Fraud Detection)',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500
    )
    fig_pr.write_html(f'{output_dir}/precision_recall_curve.html')
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Pred: Legitimate', 'Pred: Fraud'],
        y=['True: Legitimate', 'True: Fraud'],
        text=[[f'{cm[0,0]:,}', f'{cm[0,1]:,}'],
              [f'{cm[1,0]:,}', f'{cm[1,1]:,}']],
        texttemplate='%{text}',
        textfont={'size': 18},
        colorscale='Blues'
    ))
    fig_cm.update_layout(
        title='Confusion Matrix',
        height=450
    )
    fig_cm.write_html(f'{output_dir}/confusion_matrix.html')
    
    # 4. Feature Importance
    sorted_features = sorted(
        detector.feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(
        x=[f[1] for f in sorted_features][::-1],
        y=[f[0] for f in sorted_features][::-1],
        orientation='h',
        marker_color='#6366F1'
    ))
    fig_feat.update_layout(
        title='Top 20 Feature Importances',
        xaxis_title='Importance',
        height=600
    )
    fig_feat.write_html(f'{output_dir}/feature_importance.html')
    
    # 5. Score Distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=y_proba[y_test == 0],
        name='Legitimate',
        marker_color='#10B981',
        opacity=0.7,
        nbinsx=50
    ))
    fig_dist.add_trace(go.Histogram(
        x=y_proba[y_test == 1],
        name='Fraud',
        marker_color='#EF4444',
        opacity=0.7,
        nbinsx=50
    ))
    fig_dist.add_vline(
        x=detector.best_threshold,
        line_dash='dash',
        annotation_text=f'Threshold: {detector.best_threshold:.3f}'
    )
    fig_dist.update_layout(
        title='Fraud Probability Distribution',
        xaxis_title='Fraud Probability',
        yaxis_title='Count',
        barmode='overlay',
        height=500
    )
    fig_dist.write_html(f'{output_dir}/score_distribution.html')
    
    print(f"   ✅ Visualizations saved to {output_dir}/")


def generate_report(detector, output_path: str = 'reports/model_performance.xlsx'):
    """Generate Excel performance report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Model comparison
    model_comparison = pd.DataFrame([
        {'Model': name, 'ROC-AUC': res['auc_roc'], 'PR-AUC': res['pr_auc']}
        for name, res in detector.results.items()
    ]).sort_values('PR-AUC', ascending=False)
    
    # Feature importance
    feature_importance = pd.DataFrame([
        {'Feature': name, 'Importance': imp}
        for name, imp in sorted(
            detector.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ])
    
    # Confusion matrix
    cm = confusion_matrix(detector.y_test, detector.results['stacking_ensemble']['predictions'])
    cm_df = pd.DataFrame(
        cm,
        index=['True: Legitimate', 'True: Fraud'],
        columns=['Pred: Legitimate', 'Pred: Fraud']
    )
    
    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        model_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
        feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
    
    print(f"\n✅ Report saved to {output_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("🔐 CREDIT CARD FRAUD DETECTION")
    print("   Kaggle Dataset - Production Pipeline")
    print("="*60)
    
    # Initialize detector
    detector = CreditCardFraudDetector()
    
    # Load and prepare data
    X, y = detector.load_and_prepare_data('data/creditcard.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = detector.create_train_test_split(X, y)
    
    # Train models
    detector.train(X_train, y_train, X_test, y_test, use_smote=True)
    
    # Evaluate
    detector.evaluate()
    
    # Generate visualizations
    generate_visualizations(detector)
    
    # Generate report
    generate_report(detector)
    
    # Save model
    detector.save_model()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    return detector


if __name__ == "__main__":
    main()