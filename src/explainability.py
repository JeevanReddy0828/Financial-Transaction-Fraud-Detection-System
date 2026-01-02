"""
Model Explainability Module using SHAP
Provides interpretable explanations for fraud predictions
"""

import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from src.feature_engineering import prepare_data_for_training


class FraudExplainer:
    """
    SHAP-based explainability for fraud detection model.
    Critical for regulatory compliance and stakeholder trust.
    """
    
    def __init__(self, model_path: str = 'models/ensemble_model.pkl'):
        self.model_data = joblib.load(model_path)
        self.ensemble = self.model_data['ensemble']
        self.feature_names = self.model_data['feature_names']
        self.shap_values = None
        self.explainer = None
        
    def compute_shap_values(self, X: np.ndarray, sample_size: int = 500):
        """
        Compute SHAP values for model predictions.
        Uses TreeExplainer for gradient boosting models.
        """
        print("🔍 Computing SHAP values...")
        
        # Sample data for efficiency
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
            
        # Use KernelExplainer for stacking ensemble
        # Sample background data
        background = shap.sample(X, min(100, len(X)))
        
        # Create explainer
        self.explainer = shap.KernelExplainer(
            lambda x: self.ensemble.predict_proba(x)[:, 1],
            background
        )
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)
        self.X_sample = X_sample
        
        print(f"   ✅ SHAP values computed for {len(X_sample)} samples")
        return self.shap_values
    
    def get_global_importance(self) -> pd.DataFrame:
        """
        Calculate global feature importance from SHAP values.
        """
        if self.shap_values is None:
            raise ValueError("Run compute_shap_values first")
        
        # Mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP Importance': mean_shap
        }).sort_values('SHAP Importance', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, X_single: np.ndarray, transaction_id: str = "TXN_001"):
        """
        Explain a single prediction with feature contributions.
        """
        # Get prediction
        fraud_prob = self.ensemble.predict_proba(X_single.reshape(1, -1))[0, 1]
        is_fraud = fraud_prob >= self.model_data['best_threshold']
        
        # Compute SHAP for single prediction
        background = shap.sample(self.X_sample, 50)
        explainer = shap.KernelExplainer(
            lambda x: self.ensemble.predict_proba(x)[:, 1],
            background
        )
        shap_single = explainer.shap_values(X_single.reshape(1, -1))
        
        # Create explanation
        explanation = {
            'transaction_id': transaction_id,
            'fraud_probability': float(fraud_prob),
            'prediction': 'FRAUD' if is_fraud else 'LEGITIMATE',
            'top_risk_factors': [],
            'top_legitimate_factors': []
        }
        
        # Sort by SHAP contribution
        shap_contributions = list(zip(self.feature_names, shap_single[0]))
        shap_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Top factors pushing toward fraud
        for feat, val in shap_contributions[:5]:
            if val > 0:
                explanation['top_risk_factors'].append({
                    'feature': feat,
                    'contribution': float(val)
                })
        
        # Top factors pushing toward legitimate
        for feat, val in reversed(shap_contributions[-5:]):
            if val < 0:
                explanation['top_legitimate_factors'].append({
                    'feature': feat,
                    'contribution': float(val)
                })
        
        return explanation
    
    def create_summary_plot(self, output_path: str = 'visualizations/shap_summary.html'):
        """
        Create interactive SHAP summary plot.
        """
        if self.shap_values is None:
            raise ValueError("Run compute_shap_values first")
        
        # Get importance
        importance_df = self.get_global_importance().head(15)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['SHAP Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['SHAP Importance'],
                colorscale='RdYlBu_r'
            )
        ))
        
        fig.update_layout(
            title={
                'text': '🔍 SHAP Feature Importance for Fraud Detection',
                'font': {'size': 20}
            },
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Feature',
            height=600,
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        fig.write_html(output_path)
        print(f"✅ SHAP summary plot saved to {output_path}")
        
        return fig
    
    def create_force_plot(self, X_single: np.ndarray, output_path: str = 'visualizations/shap_force.html'):
        """
        Create SHAP force plot for single prediction.
        """
        # Compute SHAP for single prediction
        background = shap.sample(self.X_sample, 50)
        explainer = shap.KernelExplainer(
            lambda x: self.ensemble.predict_proba(x)[:, 1],
            background
        )
        shap_single = explainer.shap_values(X_single.reshape(1, -1))
        
        # Create waterfall-style plot
        shap_contributions = list(zip(self.feature_names, shap_single[0], X_single))
        shap_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 10
        top_contributions = shap_contributions[:10]
        
        features = [x[0] for x in top_contributions]
        values = [x[1] for x in top_contributions]
        colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors
        ))
        
        fig.update_layout(
            title={
                'text': '🎯 SHAP Force Plot - Single Transaction',
                'font': {'size': 18}
            },
            xaxis_title='SHAP Value (Impact on Fraud Probability)',
            yaxis_title='Feature',
            height=500,
            template='plotly_white',
            yaxis={'categoryorder': 'array', 'categoryarray': features[::-1]}
        )
        
        fig.write_html(output_path)
        print(f"✅ SHAP force plot saved to {output_path}")
        
        return fig


def create_visualization_dashboard(X: np.ndarray, y: np.ndarray, 
                                   model_path: str = 'models/ensemble_model.pkl'):
    """
    Create comprehensive visualization dashboard.
    """
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATION DASHBOARD")
    print("="*60)
    
    # Load model
    model_data = joblib.load(model_path)
    ensemble = model_data['ensemble']
    feature_names = model_data['feature_names']
    
    # Get predictions
    y_pred_proba = ensemble.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= model_data['best_threshold']).astype(int)
    
    # 1. ROC Curve
    print("   Creating ROC curves...")
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Stacking Ensemble (AUC = {roc_auc:.3f})',
        line=dict(color='#6366F1', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig_roc.update_layout(
        title={'text': '📈 ROC Curve - Fraud Detection Model', 'font': {'size': 20}},
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=500
    )
    fig_roc.write_html('visualizations/roc_curves.html')
    print("   ✅ ROC curve saved")
    
    # 2. Confusion Matrix
    print("   Creating confusion matrix...")
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y, y_pred)
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Legitimate', 'Predicted Fraud'],
        y=['Actual Legitimate', 'Actual Fraud'],
        text=[[f'{cm[0,0]:,}', f'{cm[0,1]:,}'],
              [f'{cm[1,0]:,}', f'{cm[1,1]:,}']],
        texttemplate='%{text}',
        textfont={'size': 20},
        colorscale='Blues'
    ))
    
    fig_cm.update_layout(
        title={'text': '🔢 Confusion Matrix', 'font': {'size': 20}},
        template='plotly_white',
        height=500
    )
    fig_cm.write_html('visualizations/confusion_matrix.html')
    print("   ✅ Confusion matrix saved")
    
    # 3. Feature Importance
    print("   Creating feature importance plot...")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model_data['feature_importances'].values()
    }).sort_values('Importance', ascending=True).tail(15)
    
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis'
        )
    ))
    
    fig_imp.update_layout(
        title={'text': '🔑 Top 15 Feature Importance', 'font': {'size': 20}},
        xaxis_title='Importance Score',
        template='plotly_white',
        height=600
    )
    fig_imp.write_html('visualizations/feature_importance.html')
    print("   ✅ Feature importance saved")
    
    # 4. Probability Distribution
    print("   Creating probability distribution...")
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=y_pred_proba[y == 0],
        name='Legitimate',
        marker_color='#4ECDC4',
        opacity=0.7,
        nbinsx=50
    ))
    fig_dist.add_trace(go.Histogram(
        x=y_pred_proba[y == 1],
        name='Fraud',
        marker_color='#FF6B6B',
        opacity=0.7,
        nbinsx=50
    ))
    
    fig_dist.add_vline(
        x=model_data['best_threshold'],
        line_dash='dash',
        annotation_text=f"Threshold: {model_data['best_threshold']:.3f}"
    )
    
    fig_dist.update_layout(
        title={'text': '📊 Fraud Probability Distribution', 'font': {'size': 20}},
        xaxis_title='Fraud Probability',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_white',
        height=500
    )
    fig_dist.write_html('visualizations/probability_distribution.html')
    print("   ✅ Probability distribution saved")
    
    print("\n✅ All visualizations created successfully!")


def main():
    """Main explainability pipeline"""
    print("="*60)
    print("🔍 SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('data/transactions.csv')
    X, y, feature_names = prepare_data_for_training(df)
    
    # Create visualizations
    create_visualization_dashboard(X.values, y.values)
    
    # SHAP analysis (computationally expensive, using smaller sample)
    print("\n🔍 Running SHAP analysis...")
    explainer = FraudExplainer()
    
    # Sample for SHAP
    sample_idx = np.random.choice(len(X), min(200, len(X)), replace=False)
    X_sample = X.values[sample_idx]
    
    explainer.compute_shap_values(X_sample, sample_size=100)
    explainer.create_summary_plot()
    
    # Example single prediction explanation
    print("\n📋 Single Transaction Explanation:")
    fraud_idx = np.where(y.values == 1)[0][0]
    explanation = explainer.explain_prediction(
        X.values[fraud_idx], 
        df.iloc[fraud_idx]['transaction_id']
    )
    
    print(f"   Transaction: {explanation['transaction_id']}")
    print(f"   Prediction: {explanation['prediction']}")
    print(f"   Fraud Probability: {explanation['fraud_probability']:.3f}")
    print("\n   Top Risk Factors:")
    for factor in explanation['top_risk_factors'][:3]:
        print(f"      • {factor['feature']}: +{factor['contribution']:.4f}")
    
    print("\n" + "="*60)
    print("✅ EXPLAINABILITY ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
