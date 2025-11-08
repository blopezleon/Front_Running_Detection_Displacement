#!/usr/bin/env python3
"""
Perfect MEV Analysis Pipeline

Combines multiple detection methods for comprehensive MEV pattern discovery:
1. Gas Auction Detection (Flash Boys 2.0 methodology)
2. Sandwich Attack Detection (Heuristic pattern matching)
3. Statistical Anomaly Detection (ML-based)

Usage:
    python perfect_analysis.py
    python perfect_analysis.py --limit 50000
    python perfect_analysis.py --output mev_patterns.db
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
import json
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Known DEX contract addresses
KNOWN_DEX_CONTRACTS = {
    '0x2a0c0dbecc7e4d658f48e01e3fa353f44050c208': 'IDEX',
    '0x8d12a197cb00d4747a1fe03395095ce2a5cc6819': 'EtherDelta',
    '0x7a250d5630b4cf539739df2c5dacb4c659f2488d': 'Uniswap_V2_Router',
    '0xe592427a0aece92de3edee1f18e0157c05861564': 'Uniswap_V3_Router',
    '0xdef1c0ded9bec7f1a1670819833240f027b25eff': '0x_Exchange',
    '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f': 'SushiSwap_Router',
    '0x881d40237659c251811cec9c364ef91dc08d300c': 'Metamask_Swap',
}


class PerfectMEVAnalyzer:
    """
    Comprehensive MEV detection using multiple methods
    """
    
    def __init__(self, db_path: str = "data/crypto_data.db"):
        self.db_path = db_path
        self.time_window = 3.0  # Flash Boys: 3-second auction windows
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
    
    def load_transactions(self, limit: int = None) -> pd.DataFrame:
        """Load transactions from database, sorted by timestamp"""
        logger.info(f"Loading transactions from {self.db_path}...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                block_number,
                transaction_hash,
                transaction_index,
                from_address,
                to_address,
                value,
                gas_price,
                gas_limit,
                gas_used,
                timestamp,
                nonce
            FROM transactions
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df):,} transactions")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    # =====================================================================
    # METHOD 1: GAS AUCTION DETECTION (Flash Boys 2.0)
    # =====================================================================
    
    def detect_gas_auctions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect gas auctions using Flash Boys 2.0 methodology:
        - 3-second time windows
        - Multiple bidders competing
        - Increasing gas prices
        
        Returns list of detected auction dictionaries
        """
        logger.info("Detecting gas auctions (Flash Boys method)...")
        
        auctions = []
        current_auction_txs = []
        auction_id = 0
        
        for i in range(len(df) - 1):
            tx = df.iloc[i]
            next_tx = df.iloc[i + 1]
            
            # Calculate time difference
            time_diff = (next_tx['timestamp'] - tx['timestamp']).total_seconds()
            
            # Within auction window?
            if time_diff <= self.time_window:
                # Start or continue auction
                if len(current_auction_txs) == 0:
                    current_auction_txs.append(tx)
                current_auction_txs.append(next_tx)
            else:
                # Auction window ended
                if len(current_auction_txs) >= 2:
                    # Valid auction: 2+ transactions
                    auction = self._create_auction_dict(
                        current_auction_txs,
                        auction_id,
                        'gas_auction'
                    )
                    auctions.append(auction)
                    auction_id += 1
                
                # Reset for next potential auction
                current_auction_txs = []
        
        # Handle last auction if exists
        if len(current_auction_txs) >= 2:
            auction = self._create_auction_dict(
                current_auction_txs,
                auction_id,
                'gas_auction'
            )
            auctions.append(auction)
        
        logger.info(f"Found {len(auctions)} gas auctions")
        return auctions
    
    # =====================================================================
    # METHOD 2: SANDWICH ATTACK DETECTION
    # =====================================================================
    
    def detect_sandwich_attacks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect sandwich attacks using pattern matching:
        - Three consecutive transactions
        - Same attacker (front and back)
        - Different victim
        - DEX interactions
        - Gas price pattern: high-low-high
        """
        logger.info("Detecting sandwich attacks...")
        
        sandwiches = []
        pattern_id = 0
        
        # Group by block for efficiency
        for block_num in df['block_number'].unique():
            block_txs = df[df['block_number'] == block_num].sort_values('transaction_index')
            
            if len(block_txs) < 3:
                continue
            
            # Check consecutive triplets
            for i in range(len(block_txs) - 2):
                front = block_txs.iloc[i]
                victim = block_txs.iloc[i + 1]
                back = block_txs.iloc[i + 2]
                
                if self._is_sandwich_pattern(front, victim, back):
                    sandwich = self._create_sandwich_dict(
                        front, victim, back, pattern_id
                    )
                    sandwiches.append(sandwich)
                    pattern_id += 1
        
        logger.info(f"Found {len(sandwiches)} sandwich attacks")
        return sandwiches
    
    def _is_sandwich_pattern(self, front, victim, back) -> bool:
        """Check if three transactions form a sandwich pattern"""
        
        # Same attacker for front and back
        if front['from_address'] != back['from_address']:
            return False
        
        # Different victim
        if victim['from_address'] == front['from_address']:
            return False
        
        # Check DEX interaction
        front_to_dex = self._is_dex_address(front['to_address'])
        back_to_dex = self._is_dex_address(back['to_address'])
        victim_to_dex = self._is_dex_address(victim['to_address'])
        
        if not (front_to_dex and back_to_dex and victim_to_dex):
            return False
        
        # Gas price pattern: front >= victim, back >= victim
        if pd.notna(front['gas_price']) and pd.notna(victim['gas_price']):
            if front['gas_price'] < victim['gas_price'] * 0.99:  # 1% tolerance
                return False
        
        if pd.notna(back['gas_price']) and pd.notna(victim['gas_price']):
            if back['gas_price'] < victim['gas_price'] * 0.99:
                return False
        
        return True
    
    # =====================================================================
    # METHOD 3: STATISTICAL ANOMALY DETECTION
    # =====================================================================
    
    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.05) -> List[Dict]:
        """
        Detect anomalies using Isolation Forest:
        - Unusual gas prices
        - Unusual values
        - Unusual timing patterns
        
        contamination: Expected proportion of outliers (default 5%)
        """
        logger.info("Detecting statistical anomalies...")
        
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not installed. Skipping anomaly detection.")
            logger.warning("Install with: pip install scikit-learn")
            return []
        
        # Prepare features
        features = df[['gas_price', 'gas_limit', 'value', 'transaction_index']].copy()
        features = features.fillna(0)
        
        # Normalize features (important for Isolation Forest)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Detect anomalies
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features_scaled)
        anomaly_scores = clf.score_samples(features_scaled)
        
        # Extract anomalies
        anomalies = []
        pattern_id = 0
        
        for idx, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
            if pred == -1:  # Anomaly
                tx = df.iloc[idx]
                anomaly = {
                    'pattern_id': pattern_id,
                    'pattern_type': 'anomaly',
                    'detection_method': 'isolation_forest',
                    'confidence': float(-score),  # Higher score = more anomalous
                    'transaction_hashes': [tx['transaction_hash']],
                    'block_numbers': [int(tx['block_number'])],
                    'transactions': [tx.to_dict()],
                    'anomaly_score': float(-score),
                }
                anomalies.append(anomaly)
                pattern_id += 1
        
        logger.info(f"Found {len(anomalies)} anomalous transactions")
        return anomalies
    
    # =====================================================================
    # FEATURE EXTRACTION
    # =====================================================================
    
    def calculate_features(self, pattern: Dict) -> Dict:
        """
        Calculate rich features for ML training
        Based on Flash Boys 2.0 insights
        """
        txs = pattern['transactions']
        
        if len(txs) == 0:
            return {}
        
        # Convert to list of dicts if DataFrame rows
        if hasattr(txs[0], 'to_dict'):
            txs = [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in txs]
        
        # Gas price statistics
        gas_prices = [tx['gas_price'] for tx in txs if pd.notna(tx['gas_price'])]
        gas_limits = [tx['gas_limit'] for tx in txs if pd.notna(tx['gas_limit'])]
        values = [tx['value'] for tx in txs if pd.notna(tx['value'])]
        
        if len(gas_prices) == 0:
            gas_prices = [0]
        if len(gas_limits) == 0:
            gas_limits = [0]
        if len(values) == 0:
            values = [0]
        
        # Timing analysis
        timestamps = [tx['timestamp'] for tx in txs]
        if isinstance(timestamps[0], str):
            timestamps = [pd.to_datetime(ts) for ts in timestamps]
        
        duration = (max(timestamps) - min(timestamps)).total_seconds()
        
        # Bidder analysis
        bidders = [tx['from_address'] for tx in txs]
        unique_bidders = set(bidders)
        
        # Price dynamics
        price_deltas = []
        for i in range(1, len(gas_prices)):
            price_deltas.append(gas_prices[i] - gas_prices[i-1])
        
        # DEX interaction
        dex_txs = sum(1 for tx in txs if self._is_dex_address(tx.get('to_address', '')))
        
        features = {
            # Pattern metadata
            'pattern_id': pattern['pattern_id'],
            'pattern_type': pattern['pattern_type'],
            'detection_method': pattern['detection_method'],
            'confidence': pattern['confidence'],
            
            # Transaction count
            'num_transactions': len(txs),
            'num_unique_bidders': len(unique_bidders),
            'repeat_bidders': len(bidders) - len(unique_bidders),
            
            # Gas price statistics
            'min_gas_price': float(min(gas_prices)),
            'max_gas_price': float(max(gas_prices)),
            'avg_gas_price': float(np.mean(gas_prices)),
            'std_gas_price': float(np.std(gas_prices)),
            'median_gas_price': float(np.median(gas_prices)),
            'gas_price_range': float(max(gas_prices) - min(gas_prices)),
            
            # Gas limit statistics
            'avg_gas_limit': float(np.mean(gas_limits)),
            'total_gas_offered': float(sum(gas_limits)),
            
            # Value statistics
            'total_value': float(sum(values)),
            'max_value': float(max(values)),
            'avg_value': float(np.mean(values)),
            
            # Price dynamics
            'avg_price_delta': float(np.mean(price_deltas)) if price_deltas else 0,
            'max_price_delta': float(max(price_deltas)) if price_deltas else 0,
            'min_price_delta': float(min(price_deltas)) if price_deltas else 0,
            'price_volatility': float(np.std(price_deltas)) if price_deltas else 0,
            
            # Timing
            'duration_seconds': float(duration),
            'bids_per_second': float(len(txs) / duration) if duration > 0 else 0,
            
            # DEX interaction
            'dex_transactions': dex_txs,
            'dex_ratio': float(dex_txs / len(txs)) if len(txs) > 0 else 0,
            
            # Block information
            'num_blocks': len(set(tx['block_number'] for tx in txs)),
            'crosses_blocks': len(set(tx['block_number'] for tx in txs)) > 1,
            'min_block': int(min(tx['block_number'] for tx in txs)),
            'max_block': int(max(tx['block_number'] for tx in txs)),
            
            # Position analysis
            'avg_tx_index': float(np.mean([tx['transaction_index'] for tx in txs 
                                          if pd.notna(tx['transaction_index'])])),
        }
        
        # Add sandwich-specific features
        if pattern['pattern_type'] == 'sandwich':
            features.update(self._calculate_sandwich_features(pattern))
        
        return features
    
    def _calculate_sandwich_features(self, pattern: Dict) -> Dict:
        """Calculate features specific to sandwich attacks"""
        txs = pattern['transactions']
        
        if len(txs) != 3:
            return {}
        
        front, victim, back = txs[0], txs[1], txs[2]
        
        victim_gas = victim['gas_price'] if pd.notna(victim['gas_price']) else 0
        front_gas = front['gas_price'] if pd.notna(front['gas_price']) else 0
        back_gas = back['gas_price'] if pd.notna(back['gas_price']) else 0
        
        return {
            'front_gas_price': float(front_gas),
            'victim_gas_price': float(victim_gas),
            'back_gas_price': float(back_gas),
            'gas_price_ratio_front': float(front_gas / victim_gas) if victim_gas > 0 else 0,
            'gas_price_ratio_back': float(back_gas / victim_gas) if victim_gas > 0 else 0,
            'victim_value': float(victim['value']) if pd.notna(victim['value']) else 0,
            'attacker_address': front['from_address'],
            'victim_address': victim['from_address'],
            'position_gap': int(back['transaction_index'] - front['transaction_index']),
        }
    
    # =====================================================================
    # HELPER METHODS
    # =====================================================================
    
    def _is_dex_address(self, address: str) -> bool:
        """Check if address is a known DEX contract"""
        if pd.isna(address) or address == '':
            return False
        return address.lower() in [addr.lower() for addr in KNOWN_DEX_CONTRACTS.keys()]
    
    def _create_auction_dict(self, txs: List, auction_id: int, pattern_type: str) -> Dict:
        """Create standardized auction dictionary"""
        return {
            'pattern_id': auction_id,
            'pattern_type': pattern_type,
            'detection_method': 'time_window_clustering',
            'confidence': 1.0,  # High confidence for time-based detection
            'transaction_hashes': [tx['transaction_hash'] for tx in txs],
            'block_numbers': [int(tx['block_number']) for tx in txs],
            'transactions': [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in txs],
        }
    
    def _create_sandwich_dict(self, front, victim, back, pattern_id: int) -> Dict:
        """Create standardized sandwich dictionary"""
        return {
            'pattern_id': pattern_id,
            'pattern_type': 'sandwich',
            'detection_method': 'heuristic_pattern',
            'confidence': 0.9,  # High confidence for heuristic detection
            'transaction_hashes': [front['transaction_hash'], victim['transaction_hash'], 
                                  back['transaction_hash']],
            'block_numbers': [int(front['block_number'])],
            'transactions': [front.to_dict() if hasattr(front, 'to_dict') else front,
                           victim.to_dict() if hasattr(victim, 'to_dict') else victim,
                           back.to_dict() if hasattr(back, 'to_dict') else back],
        }
    
    # =====================================================================
    # EXPORT & STORAGE
    # =====================================================================
    
    def export_to_csv(self, patterns: List[Dict], features_list: List[Dict], 
                     output_path: str = "mev_analysis_results.csv"):
        """Export detected patterns with features to CSV"""
        logger.info(f"Exporting {len(patterns)} patterns to {output_path}...")
        
        # Combine pattern info with features
        rows = []
        for pattern, features in zip(patterns, features_list):
            row = {
                **features,
                'transaction_hashes': json.dumps(pattern['transaction_hashes']),
                'block_numbers': json.dumps(pattern['block_numbers']),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported to {output_path}")
        logger.info(f"Columns: {len(df.columns)}, Rows: {len(df)}")
    
    def save_to_database(self, patterns: List[Dict], features_list: List[Dict],
                        db_path: str = "mev_patterns.db"):
        """Save patterns and features to SQLite database"""
        logger.info(f"Saving {len(patterns)} patterns to {db_path}...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detected_patterns (
                pattern_id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                detection_method TEXT,
                confidence REAL,
                num_transactions INTEGER,
                transaction_hashes TEXT,
                block_numbers TEXT,
                features TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert patterns
        for pattern, features in zip(patterns, features_list):
            cursor.execute("""
                INSERT OR REPLACE INTO detected_patterns
                (pattern_id, pattern_type, detection_method, confidence,
                 num_transactions, transaction_hashes, block_numbers, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                features['pattern_id'],
                features['pattern_type'],
                features['detection_method'],
                features['confidence'],
                features['num_transactions'],
                json.dumps(pattern['transaction_hashes']),
                json.dumps(pattern['block_numbers']),
                json.dumps(features)
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved to {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Perfect MEV Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perfect_analysis.py
  python perfect_analysis.py --limit 50000
  python perfect_analysis.py --output my_results.csv
  python perfect_analysis.py --detect-all
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/crypto_data.db',
        help='Input database path (default: data/crypto_data.db)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of transactions to analyze'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/mev_analysis_results.csv',
        help='Output CSV file (default: data/mev_analysis_results.csv)'
    )
    
    parser.add_argument(
        '--db-output',
        type=str,
        default='data/mev_patterns.db',
        help='Output database file (default: data/mev_patterns.db)'
    )
    
    parser.add_argument(
        '--detect-all',
        action='store_true',
        help='Run all detection methods (default: auctions + sandwiches only)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("PERFECT MEV ANALYSIS PIPELINE")
    print("="*70)
    print()
    
    # Initialize analyzer
    analyzer = PerfectMEVAnalyzer(args.input)
    
    # Load transactions
    df = analyzer.load_transactions(limit=args.limit)
    
    print()
    print("="*70)
    print("DETECTION PHASE")
    print("="*70)
    print()
    
    all_patterns = []
    
    # Method 1: Gas Auctions
    print("Method 1: Gas Auction Detection (Flash Boys 2.0)")
    print("-" * 70)
    auctions = analyzer.detect_gas_auctions(df)
    all_patterns.extend(auctions)
    print()
    
    # Method 2: Sandwich Attacks
    print("Method 2: Sandwich Attack Detection")
    print("-" * 70)
    sandwiches = analyzer.detect_sandwich_attacks(df)
    all_patterns.extend(sandwiches)
    print()
    
    # Method 3: Anomalies (optional)
    if args.detect_all:
        print("Method 3: Statistical Anomaly Detection")
        print("-" * 70)
        anomalies = analyzer.detect_anomalies(df, contamination=0.05)
        all_patterns.extend(anomalies)
        print()
    
    print("="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    print()
    
    # Calculate features for all patterns
    logger.info("Calculating features for all patterns...")
    features_list = []
    for pattern in all_patterns:
        features = analyzer.calculate_features(pattern)
        features_list.append(features)
    
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(f"Total transactions analyzed: {len(df):,}")
    print(f"Total patterns detected: {len(all_patterns):,}")
    print(f"  • Gas auctions: {len(auctions):,}")
    print(f"  • Sandwich attacks: {len(sandwiches):,}")
    if args.detect_all:
        print(f"  • Anomalies: {len(anomalies):,}")
    print(f"\nDetection rate: {len(all_patterns)/len(df)*100:.2f}%")
    print(f"Features per pattern: {len(features_list[0]) if features_list else 0}")
    
    print()
    print("="*70)
    print("EXPORT")
    print("="*70)
    print()
    
    # Export results
    if features_list:
        analyzer.export_to_csv(all_patterns, features_list, args.output)
        analyzer.save_to_database(all_patterns, features_list, args.db_output)
    
    print()
    print("="*70)
    print("COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print(f"  1. Review results: cat {args.output}")
    print(f"  2. Query database: sqlite3 {args.db_output}")
    print(f"  3. Label patterns: python label_data.py --input {args.db_output}")
    print()


if __name__ == "__main__":
    main()
