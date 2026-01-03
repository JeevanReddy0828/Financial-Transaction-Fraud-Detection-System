# 🔐 Credit Card Fraud Detection System

## Machine Learning Portfolio Project | Data Scientist

Production-ready ML pipeline for detecting fraudulent credit card transactions using the **Kaggle Credit Card Fraud Detection Dataset** with industry-standard evaluation metrics.

> 💡 *This project demonstrates fraud detection techniques applicable to fintech platforms like PayPal, Venmo, Stripe, and similar payment processors.*

---

## 📊 Dataset

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Metric | Value |
|--------|-------|
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.172%) |
| Features | V1-V28 (PCA transformed), Time, Amount |
| Time Period | 2 days (September 2013, European cardholders) |

---

## 🎯 Model Performance

### Results Achieved

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.9829 |
| **PR-AUC** | 0.8490 |
| **Fraud Recall** | 85.7% (84/98 caught) |
| **Precision** | 82.4% |
| **Precision@50** | 98% (49/50 top flags were fraud) |
| **False Positive Rate** | 0.03% |
| **Cost Savings** | 85.1% reduction in fraud losses |

### Confusion Matrix

|  | Pred: Legitimate | Pred: Fraud |
|--|------------------|-------------|
| **True: Legitimate** | 56,846 ✅ | 18 |
| **True: Fraud** | 14 ⚠️ | 84 ✅ |

---

## 🚀 Quick Start

### 1. Clone/Download Project
```bash
mkdir credit_card_fraud_detection
cd credit_card_fraud_detection
mkdir data src models reports visualizations
```

### 2. Download Dataset
```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/

# Option B: Manual download
# Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Download and extract creditcard.csv to data/ folder
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn plotly openpyxl
```

### 4. Run Training
```bash
python src/model_training.py
```

---

## 📁 Project Structure

```
credit_card_fraud_detection/
├── README.md
├── requirements.txt
├── data/
│   └── creditcard.csv              # Download from Kaggle
├── src/
│   └── model_training.py           # Main training pipeline
├── models/
│   └── fraud_detector.pkl          # Trained model
├── reports/
│   └── model_performance.xlsx      # Metrics report
└── visualizations/
    ├── roc_curve.html              # ROC curve (AUC=0.98)
    ├── precision_recall_curve.html # PR curve (AUC=0.85)
    ├── confusion_matrix.html       # Confusion matrix
    ├── feature_importance.html     # Top 20 features
    └── score_distribution.html     # Score separation
```

---

## 🔧 Technical Implementation

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `Hour_sin`, `Hour_cos` | Cyclical time encoding |
| `Log_Amount` | Log transform for skewed amounts |
| `Amount_Zscore` | Standardized transaction amount |
| `High_Amount` | Flag for amounts > 95th percentile |
| `Is_Night` | Night transaction indicator (10pm-5am) |
| `V1_V2_interaction` | PCA component interaction |
| `V14_Amount` | Top fraud predictor × amount |

### Model Architecture

```
Stacking Ensemble
├── XGBoost (scale_pos_weight=100)
├── LightGBM (class_weight='balanced')
├── Random Forest (class_weight='balanced_subsample')
└── Meta-learner: Logistic Regression
```

### Class Imbalance Handling

- **SMOTE** oversampling (50% of majority class)
- **Class weights** in all base models
- **Cost-sensitive threshold** optimization (FN cost = 100× FP cost)

---

## 📈 Evaluation Metrics

### Why These Metrics Matter for Fraud Detection

| Metric | Why Important |
|--------|---------------|
| **PR-AUC** | Better than ROC-AUC for imbalanced data (0.17% fraud) |
| **Precision@K** | Accuracy when reviewing top K flagged transactions |
| **Recall@FPR** | Fraud caught at acceptable false alarm rates |
| **Cost Analysis** | Real dollar impact of model decisions |

### Fraud-Specific Results

```
Precision@K (Top K highest-risk transactions):
   Precision@  50: 0.9800 (49 fraud in top 50)
   Precision@ 100: 0.8200 (82 fraud in top 100)
   Precision@ 200: 0.4400 (88 fraud in top 200)

Recall at Fixed False Positive Rates:
   Recall@FPR=0.1%: 87.8% fraud caught
   Recall@FPR=1.0%: 89.8% fraud caught
   Recall@FPR=5.0%: 93.9% fraud caught
```

---

## 🔑 Key Findings

### Top Fraud Predictors

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | V14 | 0.159 | Strongest PCA fraud signal |
| 2 | V4 | 0.085 | Second strongest PCA component |
| 3 | Amount_Zscore | 0.068 | Unusual amounts indicate fraud |
| 4 | V1_V2_interaction | 0.063 | Feature interaction adds value |
| 5 | High_Amount | 0.041 | Large transactions riskier |

### Business Impact

```
Cost Analysis (Avg fraud=$150, Review=$5):
   Loss without model: $14,700
   Loss with model:    $2,190
   Net savings:        $12,510 (85.1%)
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.10+ |
| **ML Models** | XGBoost, LightGBM, Random Forest, Scikit-learn |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Visualization** | Plotly (interactive HTML charts) |
| **Data Processing** | Pandas, NumPy |

---

## 📚 Skills Demonstrated

- **Machine Learning**: Ensemble methods, stacking, hyperparameter tuning
- **Imbalanced Classification**: SMOTE, class weights, threshold optimization
- **Feature Engineering**: Time encoding, interactions, transformations
- **Model Evaluation**: PR-AUC, Precision@K, Recall@FPR, cost analysis
- **Data Visualization**: Interactive Plotly dashboards
- **Production ML**: Model serialization, modular code structure

---

## 🔗 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Original Research Paper - ULB Machine Learning Group](https://www.researchgate.net/publication/260837261)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Imbalanced Classification Best Practices](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

---

## 📜 License

MIT License - Feel free to use for learning and portfolio purposes.

---

**Author**: Jeevan Arlagadda  
**Education**: MS Computer Science, University of Florida  
**Certification**: AWS Machine Learning Associate