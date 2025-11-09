#!/usr/bin/env python3

import logging
import pickle
from pathlib import Path
from collections import deque
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeMEVDetector:
    """Real-time MEV gas auction detector with adaptive thresholding."""
    
    def __init__(self, model_path, scaler_path, features_path, metrics_path=None, 
                 lookback=30, threshold=None):
        logger.info("Initializing MEV detector")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        
        self.threshold = threshold
        if threshold is None and metrics_path and Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                for line in f:
                    if 'optimal_threshold' in line:
                        self.threshold = float(line.split(':')[1].strip())
                        break
        
        if self.threshold is None:
            self.threshold = 0.5
        
        self.lookback = lookback
        self.tx_buffer = deque(maxlen=lookback)
        self.detection_stats = {'total': 0, 'mev_detected': 0, 'high_confidence': 0}
        
        logger.info(f"Loaded model with {len(self.feature_cols)} features, threshold={self.threshold:.3f}")
    
    def add_transaction(self, tx: Dict):
        """Add transaction to buffer."""
        self.tx_buffer.append(tx)
        self.detection_stats['total'] += 1
    
    def extract_features(self) -> Optional[Dict]:
        """Extract features from current buffer."""
        if len(self.tx_buffer) < 5:
            return None
        
        window = pd.DataFrame(list(self.tx_buffer))
        current = self.tx_buffer[-1]
        
        current_gas = float(current['gasPrice'])
        current_to = current.get('to')
        current_from = current['from']
        current_nonce = current['nonce']
        
        gas_prices = window['gasPrice'].astype(float).values
        timestamps = window.get('timestamp', pd.Series([pd.Timestamp.now()] * len(window))).values
        
        feat = {
            'gas_price': current_gas,
            'gas_limit': float(current['gas']),
            'tx_value': float(current['value']),
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
        
        if current_to is not None and current_to != '0x0':
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
        feat['high_gas_count'] = (gas_prices > threshold_high).sum()
        feat['high_gas_ratio'] = feat['high_gas_count'] / len(gas_prices)
        feat['current_is_high_gas'] = 1 if current_gas > threshold_high else 0
        
        feat['gas_volatility'] = np.std(gas_prices) / (np.mean(gas_prices) + 1e-9)
        feat['gas_range'] = (np.max(gas_prices) - np.min(gas_prices)) / (np.mean(gas_prices) + 1e-9)
        feat['gas_skew'] = (np.mean(gas_prices) - np.median(gas_prices)) / (np.std(gas_prices) + 1e-9)
        
        try:
            time_span = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, 's') if len(timestamps) > 1 else 0
        except:
            time_span = 0
        feat['time_span_sec'] = time_span
        feat['tx_rate'] = len(window) / (time_span + 1e-9) if time_span > 0 else 0
        
        return feat
    
    def predict(self) -> Tuple[float, bool, str]:
        """Predict MEV auction probability for current transaction."""
        features = self.extract_features()
        
        if features is None:
            return 0.0, False, "insufficient_data"
        
        X = pd.DataFrame([features])[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        probability = self.model.predict(X_scaled)[0]
        prediction = probability >= self.threshold
        
        if prediction:
            self.detection_stats['mev_detected'] += 1
            if probability >= 0.8:
                self.detection_stats['high_confidence'] += 1
        
        confidence = "high" if probability >= 0.8 else "medium" if probability >= 0.6 else "low"
        
        return probability, prediction, confidence
    
    def process_transaction(self, tx: Dict) -> Dict:
        """Process single transaction and return detection result."""
        self.add_transaction(tx)
        prob, pred, confidence = self.predict()
        
        return {
            'hash': tx.get('hash'),
            'from': tx['from'],
            'to': tx.get('to'),
            'gasPrice': tx['gasPrice'],
            'probability': prob,
            'is_mev': pred,
            'confidence': confidence
        }
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        stats = self.detection_stats.copy()
        if stats['total'] > 0:
            stats['detection_rate'] = stats['mev_detected'] / stats['total']
            stats['high_confidence_rate'] = stats['high_confidence'] / stats['total']
        return stats


def simulate_realtime_stream(detector: RealtimeMEVDetector, data_path: str, max_txs: int = 10000):
    """Simulate real-time processing of mempool transactions."""
    logger.info(f"Simulating real-time detection on {data_path}")
    
    df = pd.read_parquet(data_path, columns=['timestamp', 'hash', 'from', 'to', 'gasPrice', 'gas', 'value', 'nonce'])
    df = df.dropna(subset=['gasPrice', 'gas', 'from'])
    df = df.sort_values('timestamp').head(max_txs)
    
    logger.info(f"Processing {len(df):,} transactions")
    
    detections = []
    high_prob_count = 0
    log_counter = 0
    
    for idx, row in df.iterrows():
        tx = {
            'hash': row.get('hash'),
            'from': row['from'],
            'to': row.get('to'),
            'gasPrice': row['gasPrice'],
            'gas': row['gas'],
            'value': row['value'],
            'nonce': row['nonce'],
            'timestamp': row['timestamp']
        }
        
        result = detector.process_transaction(tx)
        log_counter += 1
        
        # Log every 10th transaction to avoid spam
        should_log = log_counter % 10 == 0
        
        if result['is_mev']:
            if result['probability'] >= 0.7:
                high_prob_count += 1
                logger.info(
                    f"MEV detected [{result['confidence'].upper()}]: "
                    f"hash={result['hash'][:10] if result['hash'] else 'N/A'}... "
                    f"gas={float(result['gasPrice'])/1e9:.2f}gwei "
                    f"prob={result['probability']:.3f}"
                )
        elif should_log:
            # Log some clean transactions to show normal activity
            logger.info(
                f"Clean transaction: "
                f"hash={result['hash'][:10] if result['hash'] else 'N/A'}... "
                f"gas={float(result['gasPrice'])/1e9:.2f}gwei "
                f"prob={result['probability']:.3f}"
            )
        
        detections.append(result)
    
    stats = detector.get_stats()
    
    logger.info(f"\nDetection Summary:")
    logger.info(f"  Total transactions: {stats['total']:,}")
    logger.info(f"  MEV detected: {stats['mev_detected']} ({100*stats['detection_rate']:.2f}%)")
    logger.info(f"  High confidence: {stats['high_confidence']} ({100*stats['high_confidence_rate']:.2f}%)")
    logger.info(f"  Probability ≥0.7: {high_prob_count}")
    
    return pd.DataFrame(detections)


def main():
    model_dir = Path("models")
    
    detector = RealtimeMEVDetector(
        model_path=model_dir / "mev_detector.pkl",
        scaler_path=model_dir / "feature_scaler.pkl",
        features_path=model_dir / "feature_columns.txt",
        metrics_path=model_dir / "training_metrics.txt",
        lookback=30
    )
    
    test_data = "data/mempool/2025-11-07.parquet"
    results = simulate_realtime_stream(detector, test_data, max_txs=20000)
    
    logger.info("Real-time detection simulation complete")


if __name__ == "__main__":
    main()
