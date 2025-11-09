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
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    data_dir: str = "data/mempool"
    output_dir: str = "models"
    max_files: int = 8  # Use ALL 8 files
    sample_size: Optional[int] = None  # Use EVERYTHING - no sampling limit
    lookback: int = 50  # Increased context window
    time_window: float = 2.5  # Slightly larger auction window
    min_auction_size: int = 2
    min_price_escalation: float = 1.03  # More sensitive than 1.05
    test_size: float = 0.2
    n_folds: int = 5  # More robust CV
    random_state: int = 42


class FlashBoysLabeler:
    """Labels gas auctions using FlashBoys heuristics with nonce replacement detection."""
    
    def __init__(self, time_window=2.5, min_auction_size=2, min_price_escalation=1.03):
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
        
        print(f"Labeling gas auctions in {len(df):,} transactions", flush=True)
        
        with tqdm(total=len(df), desc="Labeling auctions", unit="tx", smoothing=0.1) as pbar:
            idx = 0
            last_update = 0
            
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
                window_end = min(idx + 200, len(df))  # Larger search window
                
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
                
                # Update progress bar based on current position
                progress = idx - last_update
                if progress > 0:
                    pbar.update(progress)
                    last_update = idx
        
        logger.info(f"Detected {auction_count} auctions ({nonce_replacement_count} with nonce replacement)")
        logger.info(f"MEV txs: {df['is_mev'].sum():,}/{len(df):,} ({100*df['is_mev'].mean():.2f}%)")
        return df


class ImprovedFeatureExtractor:
    """
    Extract improved features for gas auction detection.
    Focus on gas dynamics, auction patterns, and temporal features.
    """
    
    def __init__(self, lookback=50):
        self.lookback = lookback
    
    def _extract_single_feature_fast(self, idx, window_start, gas_arr, to_arr, from_arr, nonce_arr, 
                                       gas_limit_arr, value_arr, timestamp_arr, labels_arr):
        """Extract features for a single transaction using pre-converted numpy arrays"""
        current_gas = gas_arr[idx]
        current_to = to_arr[idx]
        current_from = from_arr[idx]
        current_nonce = nonce_arr[idx]
        
        feat = {}
        
        # ===== Basic Transaction Features =====
        feat['gas_price'] = current_gas
        feat['gas_limit'] = gas_limit_arr[idx]
        feat['tx_value'] = value_arr[idx]
        
        # ===== Gas Dynamics (Original + Enhanced) =====
        gas_prices = gas_arr[window_start:idx]
        
        # Compute stats once
        gas_mean = np.mean(gas_prices)
        gas_std = np.std(gas_prices)
        gas_median = np.median(gas_prices)
        gas_max = np.max(gas_prices)
        gas_min = np.min(gas_prices)
        
        feat['recent_gas_mean'] = gas_mean
        feat['recent_gas_std'] = gas_std
        feat['recent_gas_median'] = gas_median
        feat['recent_gas_max'] = gas_max
        feat['recent_gas_min'] = gas_min
        feat['gas_vs_mean'] = current_gas / (gas_mean + 1e-9)
        feat['gas_vs_median'] = current_gas / (gas_median + 1e-9)
        feat['gas_vs_max'] = current_gas / (gas_max + 1e-9)
        
        # Enhanced gas features - compute percentiles once
        p75, p90, p95, p99 = np.percentile(gas_prices, [75, 90, 95, 99])
        feat['gas_percentile_75'] = p75
        feat['gas_percentile_90'] = p90
        feat['gas_percentile_95'] = p95
        feat['is_top_10pct'] = 1 if current_gas > p90 else 0
        feat['is_top_5pct'] = 1 if current_gas > p95 else 0
        feat['is_top_1pct'] = 1 if current_gas > p99 else 0
        
        # Gas volatility and dynamics
        feat['gas_volatility'] = gas_std / (gas_mean + 1e-9)
        feat['gas_range'] = (gas_max - gas_min) / (gas_mean + 1e-9)
        feat['gas_skew'] = (gas_mean - gas_median) / (gas_std + 1e-9)
        
        # Gas momentum
        if len(gas_prices) >= 10:
            recent = gas_prices[-10:]
            changes = np.diff(recent)
            feat['gas_momentum'] = np.mean(changes)
            feat['gas_acceleration'] = changes[-1] - changes[0] if len(changes) >= 2 else 0
            feat['price_increase_ratio'] = (changes > 0).sum() / len(changes)
        else:
            feat['gas_momentum'] = 0
            feat['gas_acceleration'] = 0
            feat['price_increase_ratio'] = 0
        
        # ===== Target-Based Features (Auction Detection) =====
        if pd.notna(current_to) and current_to != '0x0':
            target_mask = to_arr[window_start:idx] == current_to
            feat['same_target_count'] = target_mask.sum()
            feat['same_target_ratio'] = target_mask.sum() / (idx - window_start)
            
            if target_mask.sum() > 0:
                target_gas_prices = gas_arr[window_start:idx][target_mask]
                feat['target_gas_max'] = float(target_gas_prices.max())
                feat['target_gas_mean'] = float(target_gas_prices.mean())
                feat['target_gas_min'] = float(target_gas_prices.min())
                feat['gas_vs_target_max'] = current_gas / (feat['target_gas_max'] + 1e-9)
                feat['gas_vs_target_mean'] = current_gas / (feat['target_gas_mean'] + 1e-9)
                feat['beating_target_max'] = 1 if current_gas > feat['target_gas_max'] else 0
                
                # Target gas escalation pattern
                if len(target_gas_prices) >= 2:
                    target_diffs = np.diff(target_gas_prices)
                    feat['target_gas_escalating'] = 1 if (target_diffs > 0).sum() > len(target_diffs) * 0.6 else 0
                else:
                    feat['target_gas_escalating'] = 0
            else:
                feat['target_gas_max'] = 0
                feat['target_gas_mean'] = 0
                feat['target_gas_min'] = 0
                feat['gas_vs_target_max'] = 1
                feat['gas_vs_target_mean'] = 1
                feat['beating_target_max'] = 0
                feat['target_gas_escalating'] = 0
        else:
            feat['same_target_count'] = 0
            feat['same_target_ratio'] = 0
            feat['target_gas_max'] = 0
            feat['target_gas_mean'] = 0
            feat['target_gas_min'] = 0
            feat['gas_vs_target_max'] = 1
            feat['gas_vs_target_mean'] = 1
            feat['beating_target_max'] = 0
            feat['target_gas_escalating'] = 0
        
        # ===== Sender-Based Features (Nonce Replacement) =====
        sender_mask = from_arr[window_start:idx] == current_from
        if sender_mask.sum() > 0:
            feat['sender_recent_count'] = sender_mask.sum()
            sender_nonces = nonce_arr[window_start:idx][sender_mask]
            feat['sender_has_same_nonce'] = 1 if (sender_nonces == current_nonce).any() else 0
            sender_gas_prices = gas_arr[window_start:idx][sender_mask]
            feat['sender_max_gas'] = float(sender_gas_prices.max())
            feat['sender_mean_gas'] = float(sender_gas_prices.mean())
            feat['gas_vs_sender_max'] = current_gas / (feat['sender_max_gas'] + 1e-9)
            feat['gas_vs_sender_mean'] = current_gas / (feat['sender_mean_gas'] + 1e-9)
            
            # Sender gas escalation
            if len(sender_gas_prices) >= 2:
                sender_diffs = np.diff(sender_gas_prices)
                feat['sender_gas_escalating'] = 1 if (sender_diffs > 0).sum() > len(sender_diffs) * 0.5 else 0
            else:
                feat['sender_gas_escalating'] = 0
        else:
            feat['sender_recent_count'] = 0
            feat['sender_has_same_nonce'] = 0
            feat['sender_max_gas'] = 0
            feat['sender_mean_gas'] = 0
            feat['gas_vs_sender_max'] = 1
            feat['gas_vs_sender_mean'] = 1
            feat['sender_gas_escalating'] = 0
        
        # ===== Temporal Features =====
        timestamps = timestamp_arr[window_start:idx]
        time_span = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, 's') if len(timestamps) > 1 else 0
        feat['tx_density'] = len(timestamps) / self.lookback
        feat['time_span_sec'] = time_span
        feat['tx_rate'] = len(timestamps) / (time_span + 1e-9)
        
        # High activity periods (potential auction activity)
        feat['high_activity'] = 1 if feat['tx_rate'] > 15 else 0
        
        return feat, labels_arr[idx]
    
    def extract_features(self, df):
        logger.info(f"Extracting features (lookback={self.lookback})")
        
        # Pre-convert to numpy arrays for faster access
        gas_arr = df['gasPrice'].astype(float).values
        to_arr = df['to'].fillna('0x0').values
        from_arr = df['from'].values
        nonce_arr = df['nonce'].astype(int).values
        gas_limit_arr = df['gas'].values
        value_arr = df['value'].values
        timestamp_arr = df['timestamp'].values
        labels_arr = df['is_mev'].values
        
        features = []
        labels = []
        
        print(f"Extracting features from {len(df) - self.lookback:,} transactions", flush=True)
        
        # Process in batches for better performance
        batch_size = 10000
        n_samples = len(df) - self.lookback
        
        for batch_start in tqdm(range(0, n_samples, batch_size), desc="Feature extraction", unit=f"{batch_size}tx"):
            batch_end = min(batch_start + batch_size, n_samples)
            
            for i in range(batch_start, batch_end):
                idx = i + self.lookback
                window_start = max(0, idx - self.lookback)
                
                if idx - window_start < 10:
                    continue
                
                result = self._extract_single_feature_fast(
                    idx, window_start, 
                    gas_arr, to_arr, from_arr, nonce_arr, 
                    gas_limit_arr, value_arr, timestamp_arr, labels_arr
                )
                
                if result is not None:
                    feat, label = result
                    features.append(feat)
                    labels.append(label)
        
        logger.info(f"Extracted {len(features):,} samples with {len(features[0]) if features else 0} features")
        return pd.DataFrame(features), np.array(labels)


def load_data_memory_efficient(data_dir, max_files=8):
    """Load available data and return dataframe with file hash"""
    import hashlib
    
    logger.info(f"Loading parquet files from {data_dir}")
    
    parquet_files = sorted(Path(data_dir).glob("*.parquet"))[:max_files]
    logger.info(f"Loading {len(parquet_files)} files")
    
    # Create hash from file names
    file_names = [pf.name for pf in parquet_files]
    files_hash = hashlib.md5('|'.join(file_names).encode()).hexdigest()[:8]
    
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=[
            'timestamp', 'gasPrice', 'gas', 'value', 'from', 'to', 'nonce'
        ])
        df = df.dropna(subset=['gasPrice', 'gas', 'from'])
        
        logger.info(f"Loaded {pf.name}: {len(df):,} transactions")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total transactions: {len(combined):,}")
    logger.info(f"Data hash: {files_hash}")
    
    del dfs
    
    return combined, files_hash


def train_model_with_cv(X, y, config: TrainingConfig) -> Tuple[lgb.Booster, dict]:
    import time
    logger.info("Training LightGBM with cross-validation")
    
    # Improved hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 95,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'min_child_samples': 25,
        'min_child_weight': 0.001,
        'reg_alpha': 0.12,
        'reg_lambda': 0.12,
        'max_depth': 9,
        'verbose': -1,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'num_threads': -1,
        'force_col_wise': True
    }
    
    logger.info(f"Class balance: Positive={(y==1).sum():,}, Negative={(y==0).sum():,}")
    logger.info(f"Scale pos weight: {params['scale_pos_weight']:.2f}")
    
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    cv_scores = []
    cv_pr_aucs = []
    
    print(f"Training {config.n_folds}-fold cross-validation", flush=True)
    
    fold_times = []
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=config.n_folds, desc="CV Folds", unit="fold"), 1):
        fold_start = time.time()
        
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        print(f"  Fold {fold}: Training boosting rounds...", flush=True)
        
        # Custom callback for progress with proper timing
        progress_state = {'start_time': time.time(), 'last_iter': 0, 'last_time': time.time()}
        
        def log_progress(env):
            current_time = time.time()
            if env.iteration % 50 == 0 and env.iteration > 0:
                elapsed = current_time - progress_state['start_time']
                iters_done = env.iteration
                time_per_iter = elapsed / iters_done
                remaining = env.end_iteration - env.iteration
                eta_sec = time_per_iter * remaining
                print(f"    Round {env.iteration}/{env.end_iteration} | Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s", flush=True)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=800,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50), 
                lgb.log_evaluation(period=0),
                log_progress
            ]
        )
        
        y_pred = model.predict(X_fold_val)
        roc_auc = roc_auc_score(y_fold_val, y_pred)
        
        precision, recall, _ = precision_recall_curve(y_fold_val, y_pred)
        pr_auc = auc(recall, precision)
        
        cv_scores.append(roc_auc)
        cv_pr_aucs.append(pr_auc)
        
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        avg_fold_time = np.mean(fold_times)
        remaining_folds = config.n_folds - fold
        eta_minutes = (avg_fold_time * remaining_folds) / 60
        
        print(f"  Fold {fold}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f} | Time: {fold_time:.1f}s | ETA: {eta_minutes:.1f}m", flush=True)
    
    logger.info(f"CV ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"CV PR-AUC: {np.mean(cv_pr_aucs):.4f} ± {np.std(cv_pr_aucs):.4f}")
    
    # Train final model
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    cv_results = {
        'cv_roc_auc_mean': np.mean(cv_scores),
        'cv_roc_auc_std': np.std(cv_scores),
        'cv_pr_auc_mean': np.mean(cv_pr_aucs),
        'cv_pr_auc_std': np.std(cv_pr_aucs)
    }
    
    return final_model, cv_results


def evaluate_model(model, X_test, y_test, feature_cols):
    logger.info("Evaluating model on test set")
    
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
    
    logger.info("\nTop 20 features:")
    for _, row in importance.head(20).iterrows():
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
    
    model_path = output_dir / "mev_detector_improved.pkl"
    scaler_path = output_dir / "feature_scaler_improved.pkl"
    features_path = output_dir / "feature_columns_improved.txt"
    metrics_path = output_dir / "training_metrics_improved.txt"
    config_path = output_dir / "training_config_improved.txt"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    with open(metrics_path, 'w') as f:
        f.write("Improved MEV Detector - Training Metrics\n")
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
    print("Starting MEV Detector Training", flush=True)
    config = TrainingConfig()
    
    logger.info("MEV Detector Training - Gas Auctions")
    logger.info(f"Files: {config.max_files} | Lookback: {config.lookback}")
    logger.info(f"Time window: {config.time_window}s | Min escalation: {config.min_price_escalation}")
    
    # Load data to get hash
    df, files_hash = load_data_memory_efficient(config.data_dir, max_files=config.max_files)
    
    # Check for cached labeled data
    cache_file = Path(config.data_dir) / f"labeled_data_{files_hash}.parquet"
    
    if cache_file.exists():
        logger.info(f"Loading cached labeled data from {cache_file}")
        df = pd.read_parquet(cache_file)
        logger.info(f"Loaded {len(df):,} transactions with {df['is_mev'].sum():,} MEV transactions")
    else:
        if config.sample_size is not None and len(df) > config.sample_size:
            logger.info(f"Sampling {config.sample_size:,} transactions")
            df = df.sample(n=config.sample_size, random_state=config.random_state).sort_values('timestamp').reset_index(drop=True)
        else:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Label with FlashBoys
        labeler = FlashBoysLabeler(
            time_window=config.time_window,
            min_auction_size=config.min_auction_size,
            min_price_escalation=config.min_price_escalation
        )
        df = labeler.label_auctions(df)
        
        # Cache the labeled data
        logger.info(f"Caching labeled data to {cache_file}")
        df.to_parquet(cache_file, index=False)
    
    if df['is_mev'].sum() < 100:
        logger.warning(f"Only {df['is_mev'].sum()} MEV transactions found")
    
    # Check for cached features
    features_cache_file = Path(config.data_dir) / f"features_{files_hash}_lookback{config.lookback}.parquet"
    labels_cache_file = Path(config.data_dir) / f"labels_{files_hash}_lookback{config.lookback}.npy"
    
    if features_cache_file.exists() and labels_cache_file.exists():
        logger.info(f"Loading cached features from {features_cache_file}")
        X = pd.read_parquet(features_cache_file)
        y = np.load(labels_cache_file)
        feature_cols = X.columns.tolist()
        logger.info(f"Loaded {len(feature_cols)} features from {len(X):,} samples")
    else:
        # Extract features
        extractor = ImprovedFeatureExtractor(lookback=config.lookback)
        X, y = extractor.extract_features(df)
        
        feature_cols = X.columns.tolist()
        logger.info(f"Extracted {len(feature_cols)} features from {len(X):,} samples")
        
        # Cache the features
        logger.info(f"Caching features to {features_cache_file}")
        X.to_parquet(features_cache_file, index=False)
        np.save(labels_cache_file, y)
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"Positive rate: Train={y_train.mean():.4f}, Test={y_test.mean():.4f}\n")
    
    # Train
    model, cv_results = train_model_with_cv(X_train, y_train, config)
    test_results = evaluate_model(model, X_test, y_test, feature_cols)
    
    # Save
    all_metrics = {**cv_results, **test_results}
    save_artifacts(model, scaler, feature_cols, config.output_dir, all_metrics, config)
    
    logger.info("Training Complete")
    logger.info(f"ROC-AUC: {test_results['roc_auc']:.4f} | PR-AUC: {test_results['pr_auc']:.4f}")


if __name__ == "__main__":
    print("MEV DETECTOR - INITIALIZING", flush=True)
    main()
