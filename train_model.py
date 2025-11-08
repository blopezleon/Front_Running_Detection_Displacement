import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import numpy as np

"""
PREDICTIVE FRONT-RUNNING DETECTION MODEL

Goal: Predict if a NEW transaction will trigger a front-running auction
BEFORE the auction completes.

Key Insight: We train on past auctions (labeled by hardcoded rules), but
predict on INDIVIDUAL TRANSACTIONS as they arrive in the mempool.

Use Case: Real-time MEV detection for incoming transactions
"""

# Load labeled data
df = pd.read_csv('data/flashboys_analysis.csv')

print(f"{'='*70}")
print("PREDICTIVE FRONT-RUNNING MODEL")
print(f"{'='*70}")
print(f"Total transactions: {len(df):,}")
print(f"Total auctions: {df['auction_id'].nunique():,}")
print(f"MEV transactions: {df['is_mev_auction'].sum():,} ({df['is_mev_auction'].mean()*100:.1f}%)\n")

# Sort by timestamp to respect temporal order
df = df.sort_values('first_block').reset_index(drop=True)

# CRITICAL: Use only features available WHEN THE TRANSACTION ARRIVES
# These are observable from the transaction itself + recent history
features_per_tx = []
labels = []

print("Engineering predictive features...")

for idx, row in df.iterrows():
    # Features available IMMEDIATELY when tx arrives in mempool:
    features = {
        # Transaction characteristics
        'tx_gas_price': row['tx_gas_price'],
        'tx_gas_limit': row['tx_gas_limit'],
        'tx_value': row['tx_value'],
        
        # Position in its own auction (how quickly did it follow first tx?)
        'tx_position_in_auction': row['tx_position_in_auction'],
        
        # Ratios (gas efficiency indicators)
        'gas_price_to_limit_ratio': row['tx_gas_price'] / (row['tx_gas_limit'] + 1),
        'value_to_gas_ratio': row['tx_value'] / (row['tx_gas_price'] + 1),
        
        # Is this the first transaction in a potential auction?
        'is_first_tx': int(row['tx_position_in_auction'] == 0),
        
        # Gas price percentile relative to recent history
        # (In practice: compare to recent block gas prices)
        'gas_price_normalized': row['tx_gas_price'] / (row['avg_gas_price'] + 1),
    }
    
    # Historical context features (from previous blocks)
    # Simulate what we'd know from recent blockchain state
    if idx > 0:
        recent = df.iloc[max(0, idx-100):idx]  # Last 100 txs
        if len(recent) > 0:
            features['recent_avg_gas'] = recent['tx_gas_price'].mean()
            features['recent_max_gas'] = recent['tx_gas_price'].max()
            features['recent_mev_rate'] = recent['is_mev_auction'].mean()
            features['gas_vs_recent_avg'] = row['tx_gas_price'] / (features['recent_avg_gas'] + 1)
            features['gas_vs_recent_max'] = row['tx_gas_price'] / (features['recent_max_gas'] + 1)
        else:
            features['recent_avg_gas'] = 0
            features['recent_max_gas'] = 0
            features['recent_mev_rate'] = 0
            features['gas_vs_recent_avg'] = 1
            features['gas_vs_recent_max'] = 1
    else:
        features['recent_avg_gas'] = 0
        features['recent_max_gas'] = 0
        features['recent_mev_rate'] = 0
        features['gas_vs_recent_avg'] = 1
        features['gas_vs_recent_max'] = 1
    
    features_per_tx.append(features)
    labels.append(row['is_mev_auction'])

# Convert to DataFrame
X = pd.DataFrame(features_per_tx)
y = pd.Series(labels)

print(f"Features engineered: {len(X.columns)}")
print(f"Feature names: {list(X.columns)}\n")

# TIME-BASED SPLIT (critical for temporal prediction!)
# Train on early data, test on later data (simulates real deployment)
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training set: {len(X_train):,} transactions")
print(f"Test set: {len(X_test):,} transactions (FUTURE data)")
print(f"Train MEV rate: {y_train.mean()*100:.1f}%")
print(f"Test MEV rate: {y_test.mean()*100:.1f}%\n")

# Train with class balancing (MEV is minority class)
print("Training Gradient Boosting model with class balancing...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    subsample=0.8,
    min_samples_split=100,
)

# Calculate class weights
class_weight = len(y_train) / (2 * np.bincount(y_train))
sample_weights = np.where(y_train == 1, class_weight[1], class_weight[0])

model.fit(X_train, y_train, sample_weight=sample_weights)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
print(f"\n{'='*70}")
print("PREDICTIVE MODEL RESULTS (Time-Based Split)")
print(f"{'='*70}")
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'MEV Front-Run']))
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]:,} | False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,} | True Positives: {cm[1,1]:,}")

# Feature importance
print(f"\n{'='*70}")
print("Feature Importance (What Predicts Front-Running?):")
print(f"{'='*70}")
feature_importance = sorted(zip(X.columns, model.feature_importances_), 
                            key=lambda x: x[1], reverse=True)
for feat, importance in feature_importance[:10]:
    print(f"{feat:30s}: {importance:.4f}")

# Precision-Recall analysis for different thresholds
print(f"\n{'='*70}")
print("Threshold Analysis (For Real-Time Detection):")
print(f"{'='*70}")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
for threshold in [0.3, 0.5, 0.7, 0.9]:
    preds_at_thresh = (y_pred_proba >= threshold).astype(int)
    precision = (preds_at_thresh & y_test).sum() / (preds_at_thresh.sum() + 1e-10)
    recall = (preds_at_thresh & y_test).sum() / (y_test.sum() + 1e-10)
    print(f"Threshold {threshold:.1f}: Precision={precision:.2%}, Recall={recall:.2%}")

print(f"{'='*70}")
print("\n✅ Model trained for PREDICTIVE front-running detection!")
print("Use this to flag suspicious transactions in REAL-TIME.")
print(f"{'='*70}")