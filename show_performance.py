"""
MEV Detection Model - Performance Statistics
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_against_flashboys():
    """Validate ML model predictions against FlashBoys ground truth"""
    import pandas as pd
    import numpy as np
    import pickle
    from train_mev_detector import FlashBoysLabeler, FeatureExtractor
    from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
    
    logger.info("Running validation against FlashBoys ground truth")
    
    # Load model
    logger.info("Loading trained model")
    with open('models/mev_detector.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_columns.txt', 'r') as f:
        feature_cols = [line.strip() for line in f]
    
    # Load test data
    logger.info("Loading test data: data/mempool/2025-11-05.parquet")
    df = pd.read_parquet('data/mempool/2025-11-05.parquet')
    df = df.sample(n=min(100000, len(df)), random_state=42).reset_index(drop=True)
    logger.info(f"Test data: {len(df):,} transactions")
    
    # Get FlashBoys ground truth
    logger.info("Running FlashBoys labeling for ground truth")
    labeler = FlashBoysLabeler(time_window=2.0, min_price_escalation=1.05)
    df = labeler.label_auctions(df)
    flashboys_mev = df['is_mev'].sum()
    logger.info(f"FlashBoys detected: {flashboys_mev:,} MEV ({flashboys_mev/len(df)*100:.2f}%)")
    
    # Extract features and get ML predictions
    logger.info("Extracting features and running ML predictions")
    extractor = FeatureExtractor()
    features_df, labels_array = extractor.extract_features(df)
    
    y_true = np.array(labels_array, dtype=int)
    X = features_df[feature_cols]
    
    X_scaled = scaler.transform(X)
    y_proba = model.predict(X_scaled, raw_score=False)
    threshold = 0.922
    y_pred = (y_proba >= threshold).astype(int)
    
    ml_mev = y_pred.sum()
    logger.info(f"ML Model detected: {ml_mev:,} MEV ({ml_mev/len(y_pred)*100:.2f}%)")
    
    # Calculate agreement
    both_mev = ((y_pred == 1) & (y_true == 1)).sum()
    only_flashboys = ((y_pred == 0) & (y_true == 1)).sum()
    only_ml = ((y_pred == 1) & (y_true == 0)).sum()
    both_clean = ((y_pred == 0) & (y_true == 0)).sum()
    
    logger.info(f"Agreement: Both detect MEV: {both_mev:,}")
    logger.info(f"Agreement: Only FlashBoys: {only_flashboys:,}")
    logger.info(f"Agreement: Only ML: {only_ml:,}")
    logger.info(f"Agreement: Both clean: {both_clean:,}")
    
    recall = both_mev / (both_mev + only_flashboys) if (both_mev + only_flashboys) > 0 else 0
    precision = both_mev / (both_mev + only_ml) if (both_mev + only_ml) > 0 else 0
    
    logger.info(f"ML Recall: {recall*100:.2f}% (catches {recall*100:.1f}% of FlashBoys detections)")
    logger.info(f"ML Precision: {precision*100:.2f}% (when ML says MEV, FlashBoys agrees {precision*100:.1f}%)")
    
    # ROC-AUC on test set
    if y_true.sum() > 0:
        roc_auc = roc_auc_score(y_true, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Test PR-AUC: {pr_auc:.4f}")
    
    logger.info("Validation complete")


def show_model_performance():
    """Display model performance statistics"""
    
    logger.info("Loading model performance statistics")
    
    # Parse metrics file
    metrics = {}
    feature_importance = []
    
    with open('models/training_metrics.txt', 'r') as f:
        lines = f.readlines()
        parsing_importance = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('='):
                continue
            if line.startswith('importance:'):
                parsing_importance = True
                continue
            
            if parsing_importance:
                if line and not line.startswith('Training'):
                    feature_importance.append(line)
            elif ':' in line:
                key, value = line.split(':', 1)
                metrics[key.strip()] = value.strip()
    
    # Display statistics
    logger.info("Model: LightGBM Gradient Boosting Classifier")
    logger.info(f"Training samples: 200,000 transactions")
    logger.info(f"Feature count: 32 engineered features")
    logger.info(f"Validation: 3-fold Stratified Cross-Validation")
    
    logger.info(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')} (CV mean: {metrics.get('cv_roc_auc_mean', 'N/A')} ± {metrics.get('cv_roc_auc_std', 'N/A')})")
    logger.info(f"PR-AUC: {metrics.get('pr_auc', 'N/A')} (CV mean: {metrics.get('cv_pr_auc_mean', 'N/A')} ± {metrics.get('cv_pr_auc_std', 'N/A')})")
    logger.info(f"Optimal threshold: {metrics.get('optimal_threshold', 'N/A')}")
    
    logger.info("Top 5 feature importances:")
    for i, feat_line in enumerate(feature_importance[:5], 1):
        logger.info(f"  {i}. {feat_line}")
    
    logger.info("Performance statistics loaded successfully")


if __name__ == '__main__':
    show_model_performance()
    logger.info("")
    validate_against_flashboys()
