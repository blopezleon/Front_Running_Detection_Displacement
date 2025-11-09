"""
Analyze when the model detects MEV transactions within auctions.
Shows detection timing: early, middle, or late in auction sequences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_auction_timing():
    """Analyze when in an auction the model makes predictions"""
    
    # Load the trained model and scaler separately
    model_path = Path("models/mev_detector_improved.pkl")
    scaler_path = Path("models/feature_scaler_improved.pkl")
    features_path = Path("models/feature_columns_improved.txt")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(features_path, 'r') as f:
        feature_cols = [line.strip() for line in f]
    
    logger.info(f"Loaded model with {len(feature_cols)} features")
    
    # Load cached features and labels
    data_dir = Path("data/mempool")
    
    # Find the most recent feature cache
    feature_files = list(data_dir.glob("features_*_lookback50.parquet"))
    if not feature_files:
        logger.error("No cached features found")
        return
    
    feature_file = sorted(feature_files, key=lambda x: x.stat().st_mtime)[-1]
    label_file = feature_file.with_suffix('').with_suffix('').name.replace('features_', 'labels_') + '.npy'
    label_file = data_dir / label_file
    
    logger.info(f"Loading features from {feature_file.name}")
    X = pd.read_parquet(feature_file)
    y = np.load(label_file)
    
    logger.info(f"Loaded {len(X):,} samples with {y.sum():,} positive labels")
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred_proba = model.predict(X_scaled)
    
    # Filter to positive labels
    mev_mask = y == 1
    X_mev = X[mev_mask]
    y_pred_mev = y_pred_proba[mev_mask]
    
    logger.info(f"Analyzing {len(X_mev):,} MEV transactions")
    
    # Analyze same_target_count distribution (proxy for auction position)
    if 'same_target_count' in X_mev.columns:
        target_counts = X_mev['same_target_count'].values
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(target_counts, p)
            logger.info(f"Same-target count p{p}: {val:.1f}")
        
        # Categorize by position in auction
        early = target_counts <= 1
        middle = (target_counts > 1) & (target_counts <= 5)
        late = target_counts > 5
        
        logger.info(f"Early detection (≤1 prior tx): {early.sum():,} ({100*early.mean():.2f}%)")
        logger.info(f"Middle detection (2-5 prior): {middle.sum():,} ({100*middle.mean():.2f}%)")
        logger.info(f"Late detection (>5 prior): {late.sum():,} ({100*late.mean():.2f}%)")
        
        logger.info(f"Model confidence early: {y_pred_mev[early].mean():.4f} ± {y_pred_mev[early].std():.4f}")
        logger.info(f"Model confidence middle: {y_pred_mev[middle].mean():.4f} ± {y_pred_mev[middle].std():.4f}")
        logger.info(f"Model confidence late: {y_pred_mev[late].mean():.4f} ± {y_pred_mev[late].std():.4f}")
    
    # Analyze nonce replacement
    if 'sender_has_same_nonce' in X_mev.columns:
        nonce_replacement = X_mev['sender_has_same_nonce'] > 0
        
        logger.info(f"Nonce replacement txs: {nonce_replacement.sum():,} ({100*nonce_replacement.mean():.2f}%)")
        logger.info(f"Confidence with nonce replacement: {y_pred_mev[nonce_replacement].mean():.4f}")
        logger.info(f"Confidence without nonce replacement: {y_pred_mev[~nonce_replacement].mean():.4f}")
    
    # Gas escalation patterns
    if 'target_gas_escalating' in X_mev.columns:
        escalating = X_mev['target_gas_escalating'] > 0
        
        logger.info(f"Gas escalation txs: {escalating.sum():,} ({100*escalating.mean():.2f}%)")
        logger.info(f"Confidence with escalation: {y_pred_mev[escalating].mean():.4f}")
        logger.info(f"Confidence without escalation: {y_pred_mev[~escalating].mean():.4f}")
    
    # Combined early detection indicators
    if 'same_target_count' in X_mev.columns and 'sender_has_same_nonce' in X_mev.columns:
        early_detection = (X_mev['same_target_count'] <= 1) | (X_mev['sender_has_same_nonce'] > 0)
        
        logger.info(f"Early detectable txs: {early_detection.sum():,} ({100*early_detection.mean():.2f}%)")
        
        high_conf_early = early_detection & (y_pred_mev > 0.9)
        logger.info(f"High confidence (>0.9) early: {high_conf_early.sum():,} ({100*high_conf_early.mean():.2f}%)")


if __name__ == "__main__":
    analyze_auction_timing()
