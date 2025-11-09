#!/usr/bin/env python3

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import lightgbm as lgb
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    data_dir: str = "data/mempool"
    output_dir: str = "models"
    max_files: int = 2
    sample_size: int = 200000
    lookback: int = 30
    time_window: float = 2.0
    min_auction_size: int = 2
    min_price_escalation: float = 1.05
    test_size: float = 0.2
    n_folds: int = 3
    random_state: int = 42


class FlashBoysLabeler:
    """Labels gas auctions using FlashBoys heuristics with nonce replacement detection."""
    
    def __init__(self, time_window=2.0, min_auction_size=2, min_price_escalation=1.05):
        self.time_window = time_window
        self.min_auction_size = min_auction_size
        self.min_price_escalation = min_price_escalation
    
    def label_auctions(self, df):
        logger.info("Labeling gas auctions with nonce replacement detection")
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['is_mev'] = 0
        df['to_filled'] = df['to'].fillna('0x0')
        df['gasPrice'] = df['gasPrice'].astype(float)
        df['nonce'] = df['nonce'].astype(int)
        df['timestamp_sec'] = df['timestamp'].astype(np.int64) / 1e9
        
        to_arr = df['to_filled'].values
        from_arr = df['from'].values
        nonce_arr = df['nonce'].values
        time_arr = df['timestamp_sec'].values
        gas_arr = df['gasPrice'].values
        
        auction_count = 0
        nonce_replacement_count = 0
        processed = np.zeros(len(df), dtype=bool)
        idx = 0
        
        while idx < len(df):
            if processed[idx]:
                idx += 1
                continue
            
            target = to_arr[idx]
            current_from = from_arr[idx]
            current_nonce = nonce_arr[idx]
            
            if target == '0x0':
                idx += 1
                continue
            
            base_time = time_arr[idx]
            max_time = base_time + self.time_window
            window_end = min(idx + 150, len(df))
            
            candidates = []
            has_nonce_replacement = False
            
            for i in range(idx, window_end):
                if processed[i] or time_arr[i] > max_time:
                    continue
                
                same_target = to_arr[i] == target
                same_sender_nonce = from_arr[i] == current_from and nonce_arr[i] == current_nonce
                
                if same_target or same_sender_nonce:
                    candidates.append(i)
                    if same_sender_nonce and i != idx:
                        has_nonce_replacement = True
            
            if len(candidates) < self.min_auction_size:
                idx += 1
                continue
            
            gas_prices = gas_arr[candidates]
            is_escalating = all(
                gas_prices[k + 1] >= gas_prices[k] * self.min_price_escalation
                for k in range(len(gas_prices) - 1)
            )
            
            if not is_escalating and not has_nonce_replacement:
                idx += 1
                continue
            
            processed[candidates] = True
            df.loc[candidates, 'is_mev'] = 1
            auction_count += 1
            if has_nonce_replacement:
                nonce_replacement_count += 1
            idx = candidates[-1] + 1
        
        logger.info(f"Detected {auction_count} auctions ({nonce_replacement_count} with nonce replacement)")
        logger.info(f"MEV txs: {df['is_mev'].sum():,}/{len(df):,} ({100*df['is_mev'].mean():.2f}%)")
        return df


class FeatureExtractor:
    """Extracts causal features for MEV detection with advanced gas dynamics."""
    
    def __init__(self, lookback=30):
        self.lookback = lookback
    
    def extract_features(self, df):
        logger.info("Extracting features")
        
        features = []
        labels = []
        
        for idx in range(self.lookback, len(df)):
            window_start = max(0, idx - self.lookback)
            window = df.iloc[window_start:idx]
            
            if len(window) < 5:
                continue
            
            current_gas = float(df.loc[idx, 'gasPrice'])
            current_to = df.loc[idx, 'to']
            current_from = df.loc[idx, 'from']
            current_nonce = int(df.loc[idx, 'nonce'])
            
            gas_prices = window['gasPrice'].astype(float).values
            timestamps = window['timestamp'].values
            
            feat = {
                'gas_price': current_gas,
                'gas_limit': df.loc[idx, 'gas'],
                'tx_value': df.loc[idx, 'value'],
                'recent_gas_mean': np.mean(gas_prices),
                'recent_gas_std': np.std(gas_prices),
                'recent_gas_median': np.median(gas_prices),
                'recent_gas_max': np.max(gas_prices),
                'recent_gas_min': np.min(gas_prices),
                'gas_vs_mean': current_gas / (np.mean(gas_prices) + 1e-9),
                'gas_vs_median': current_gas / (np.median(gas_prices) + 1e-9),
                'gas_vs_max': current_gas / (np.max(gas_prices) + 1e-9),
                'tx_density': len(window) / self.lookback
            }
            
            if pd.notna(current_to):
                target_txs = window[window['to'] == current_to]
                feat['same_target_count'] = len(target_txs)
                feat['same_target_ratio'] = len(target_txs) / len(window)
                
                if len(target_txs) > 0:
                    target_max = float(target_txs['gasPrice'].max())
                    feat['target_gas_max'] = target_max
                    feat['gas_vs_target_max'] = current_gas / (target_max + 1e-9)
                    feat['beating_target_max'] = 1 if current_gas > target_max else 0
                else:
                    feat['target_gas_max'] = 0
                    feat['gas_vs_target_max'] = 1
                    feat['beating_target_max'] = 0
            else:
                feat['same_target_count'] = 0
                feat['same_target_ratio'] = 0
                feat['target_gas_max'] = 0
                feat['gas_vs_target_max'] = 1
                feat['beating_target_max'] = 0
            
            sender_txs = window[window['from'] == current_from]
            if len(sender_txs) > 0:
                feat['sender_recent_count'] = len(sender_txs)
                feat['sender_has_same_nonce'] = 1 if (sender_txs['nonce'] == current_nonce).any() else 0
                sender_max = float(sender_txs['gasPrice'].max())
                feat['sender_max_gas'] = sender_max
                feat['gas_vs_sender_max'] = current_gas / (sender_max + 1e-9)
            else:
                feat['sender_recent_count'] = 0
                feat['sender_has_same_nonce'] = 0
                feat['sender_max_gas'] = 0
                feat['gas_vs_sender_max'] = 1
            
            if len(gas_prices) >= 5:
                recent = gas_prices[-5:]
                changes = np.diff(recent)
                feat['gas_momentum'] = np.mean(changes)
                feat['gas_acceleration'] = changes[-1] - changes[0] if len(changes) >= 2 else 0
                feat['price_increase_ratio'] = (changes > 0).sum() / len(changes)
            else:
                feat['gas_momentum'] = 0
                feat['gas_acceleration'] = 0
                feat['price_increase_ratio'] = 0
            
            threshold_high = np.percentile(gas_prices, 75)
            threshold_low = np.percentile(gas_prices, 25)
            feat['high_gas_count'] = (gas_prices > threshold_high).sum()
            feat['high_gas_ratio'] = feat['high_gas_count'] / len(gas_prices)
            feat['current_is_high_gas'] = 1 if current_gas > threshold_high else 0
            
            feat['gas_volatility'] = np.std(gas_prices) / (np.mean(gas_prices) + 1e-9)
            feat['gas_range'] = (np.max(gas_prices) - np.min(gas_prices)) / (np.mean(gas_prices) + 1e-9)
            feat['gas_skew'] = (np.mean(gas_prices) - np.median(gas_prices)) / (np.std(gas_prices) + 1e-9)
            
            time_span = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, 's') if len(timestamps) > 1 else 0
            feat['time_span_sec'] = time_span
            feat['tx_rate'] = len(window) / (time_span + 1e-9)
            
            features.append(feat)
            labels.append(df.loc[idx, 'is_mev'])
        
        logger.info(f"Extracted features for {len(features):,} transactions")
        return pd.DataFrame(features), np.array(labels)


def load_parquet_data(data_dir, max_files=None):
    logger.info(f"Loading parquet files from {data_dir}")
    
    parquet_files = sorted(Path(data_dir).glob("*.parquet"))
    if max_files:
        parquet_files = parquet_files[:max_files]
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=['timestamp', 'gasPrice', 'gas', 'value', 'from', 'to', 'nonce'])
        df = df.dropna(subset=['gasPrice', 'gas', 'from'])
        logger.info(f"Loaded {pf.name}: {len(df):,} transactions")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total transactions: {len(combined):,}")
    
    return combined


def train_model_with_cv(X, y, config: TrainingConfig) -> Tuple[lgb.Booster, dict]:
    logger.info("Training LightGBM with cross-validation")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': 8,
        'verbose': -1,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum()
    }
    
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    cv_scores = []
    cv_pr_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
        )
        
        y_pred = model.predict(X_fold_val)
        roc_auc = roc_auc_score(y_fold_val, y_pred)
        
        precision, recall, _ = precision_recall_curve(y_fold_val, y_pred)
        pr_auc = auc(recall, precision)
        
        cv_scores.append(roc_auc)
        cv_pr_aucs.append(pr_auc)
        logger.info(f"Fold {fold}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    
    logger.info(f"CV ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"CV PR-AUC: {np.mean(cv_pr_aucs):.4f} ± {np.std(cv_pr_aucs):.4f}")
    
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=int(np.mean([m.best_iteration for m in [model]])) * config.n_folds // 2,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    cv_results = {
        'cv_roc_auc_mean': np.mean(cv_scores),
        'cv_roc_auc_std': np.std(cv_scores),
        'cv_pr_auc_mean': np.mean(cv_pr_aucs),
        'cv_pr_auc_std': np.std(cv_pr_aucs)
    }
    
    logger.info("Model training complete")
    return final_model, cv_results


def evaluate_model(model, X_test, y_test, feature_cols):
    logger.info("Evaluating model on holdout test set")
    
    y_pred_proba = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    optimal_idx = np.argmax(precision * recall)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    logger.info(f"Test ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
    
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 features:")
    for _, row in importance.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.0f}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'importance': importance
    }


def save_artifacts(model, scaler, feature_cols, output_dir, metrics, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "mev_detector.pkl"
    scaler_path = output_dir / "feature_scaler.pkl"
    features_path = output_dir / "feature_columns.txt"
    metrics_path = output_dir / "training_metrics.txt"
    config_path = output_dir / "training_config.txt"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    with open(metrics_path, 'w') as f:
        f.write("Training Metrics\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
    
    with open(config_path, 'w') as f:
        f.write("Training Configuration\n")
        f.write("=" * 50 + "\n")
        for k, v in config.__dict__.items():
            f.write(f"{k}: {v}\n")
    
    logger.info(f"Saved artifacts to {output_dir}")


def main():
    config = TrainingConfig()
    
    logger.info("Starting MEV detector training")
    logger.info(f"Config: sample_size={config.sample_size:,}, lookback={config.lookback}, "
                f"time_window={config.time_window}s, min_escalation={config.min_price_escalation}")
    
    df = load_parquet_data(config.data_dir, max_files=config.max_files)
    
    if len(df) > config.sample_size:
        logger.info(f"Sampling {config.sample_size:,} transactions")
        df = df.sample(n=config.sample_size, random_state=config.random_state).sort_values('timestamp').reset_index(drop=True)
    
    labeler = FlashBoysLabeler(
        time_window=config.time_window,
        min_auction_size=config.min_auction_size,
        min_price_escalation=config.min_price_escalation
    )
    df = labeler.label_auctions(df)
    
    if df['is_mev'].sum() < 100:
        logger.warning(f"Only {df['is_mev'].sum()} MEV transactions found. Consider adjusting parameters.")
    
    extractor = FeatureExtractor(lookback=config.lookback)
    X, y = extractor.extract_features(df)
    
    feature_cols = X.columns.tolist()
    logger.info(f"Extracted {len(feature_cols)} features")
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    logger.info(f"Positive rate: Train={y_train.mean():.4f}, Test={y_test.mean():.4f}")
    
    model, cv_results = train_model_with_cv(X_train, y_train, config)
    test_results = evaluate_model(model, X_test, y_test, feature_cols)
    
    all_metrics = {**cv_results, **test_results}
    save_artifacts(model, scaler, feature_cols, config.output_dir, all_metrics, config)
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
