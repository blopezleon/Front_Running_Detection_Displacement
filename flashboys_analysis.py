#!/usr/bin/env python3
"""
Flash Boys 2.0 Analysis - Official Implementation

Uses the ACTUAL algorithms from the Flash Boys 2.0 paper researchers.
This script:
1. Converts your collected data to their format
2. Applies their gas auction detection algorithm
3. Outputs ML-ready labeled dataset

Based on: flashboys2/read_csv.py (official implementation)
"""

import sqlite3
import pandas as pd
import csv
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FlashBoysAnalyzer:
    """
    Official Flash Boys 2.0 gas auction detection
    
    From the paper: Detects gas auctions using 3-second time windows
    where multiple bidders compete with increasing gas prices.
    """
    
    def __init__(self, db_path: str = "data/crypto_data.db"):
        self.db_path = db_path
        self.time_window = 3.0  # 3-second auction window (from paper)
        
    def load_transactions_from_db(self) -> list:
        """Load transactions from your database in Flash Boys format"""
        logger.info(f"Loading transactions from {self.db_path}...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get transactions sorted by timestamp
        cursor.execute("""
            SELECT 
                transaction_hash as hash,
                from_address as sender,
                nonce as account_nonce,
                gas_price,
                gas_limit,
                timestamp,
                block_number,
                transaction_index,
                to_address,
                value,
                input_data as payload
            FROM transactions
            ORDER BY timestamp, transaction_index
        """)
        
        transactions = []
        for row in cursor.fetchall():
            # Convert to Flash Boys format
            tx = {
                'hash': row[0],
                'sender': row[1].lower() if row[1] else '',
                'account_nonce': str(row[2]) if row[2] is not None else '0',
                'gas_price': int(row[3]) if row[3] else 0,
                'gas_limit': int(row[4]) if row[4] else 0,
                'timestamp': row[5],
                'time_seen': self._to_nanoseconds(row[5]),  # Convert to nanoseconds
                'block_number': row[6],
                'transaction_index': row[7],
                'to_address': row[8].lower() if row[8] else '',
                'value': float(row[9]) if row[9] else 0.0,
                'payload': row[10] if row[10] else ''
            }
            transactions.append(tx)
        
        conn.close()
        logger.info(f"Loaded {len(transactions):,} transactions")
        return transactions
    
    def _to_nanoseconds(self, timestamp_str):
        """Convert timestamp to nanoseconds (Flash Boys format)"""
        if isinstance(timestamp_str, str):
            dt = datetime.fromisoformat(timestamp_str)
        else:
            dt = timestamp_str
        return int(dt.timestamp() * 1e9)
    
    # =========================================================================
    # OFFICIAL FLASH BOYS AUCTION DETECTION ALGORITHM
    # =========================================================================
    
    def get_bidder(self, item):
        """Get unique bidder identifier (address + nonce)"""
        return (item['sender'], item['account_nonce'])
    
    def add_bidder_to(self, auction_participation, bidder, auction_id):
        """Track bidder participation in auctions"""
        if bidder in auction_participation:
            if auction_id in auction_participation[bidder]:
                auction_participation[bidder][auction_id] += 1
            else:
                auction_participation[bidder][auction_id] = 1
        else:
            auction_participation[bidder] = {auction_id: 1}
    
    def get_individual_auctions(self, seen_list):
        """
        CORE ALGORITHM FROM FLASH BOYS 2.0 PAPER
        
        Detects gas auctions using 3-second time windows.
        If transactions arrive within 3 seconds of each other,
        they are part of the same auction.
        
        Returns:
            auctions: List of auction transaction lists
            non_auctions: Transactions not in auctions
            bidders: Set of bidders per auction
            participation: Bidder->auction mapping
        """
        auctions = []
        auction_bidders = []
        non_auction_txs = []
        auction_participation = {}
        
        curr_auction = []
        curr_bidders = set()
        auction_id = 0
        
        for i in range(len(seen_list) - 1):
            prev_item = seen_list[i]
            item = seen_list[i+1]
            
            # Calculate time difference in seconds
            time_difference = (item['time_seen'] - prev_item['time_seen']) / 1e9
            
            if time_difference < self.time_window:
                # This tx is part of the auction
                bidder_id = self.get_bidder(item)
                
                if len(curr_auction) == 0:
                    # New auction; previous tx must have triggered it
                    curr_auction = [prev_item, item]
                    # Previous tx actually isn't non-auction
                    if non_auction_txs:
                        non_auction_txs = non_auction_txs[:-1]
                    
                    original_bidder_id = self.get_bidder(prev_item)
                    curr_bidders.add(original_bidder_id)
                    curr_bidders.add(bidder_id)
                    self.add_bidder_to(auction_participation, original_bidder_id, auction_id)
                else:
                    curr_auction.append(item)
                    curr_bidders.add(bidder_id)
                
                self.add_bidder_to(auction_participation, bidder_id, auction_id)
            else:
                # Transaction is not part of an auction
                if len(curr_auction) != 0:
                    # Previous auction ended; log and reset
                    auctions.append(curr_auction)
                    auction_bidders.append(curr_bidders)
                    curr_auction = []
                    curr_bidders = set()
                    auction_id += 1
                non_auction_txs.append(item)
        
        # Handle last auction if exists
        if len(curr_auction) != 0:
            auctions.append(curr_auction)
            auction_bidders.append(curr_bidders)
        
        return auctions, non_auction_txs, auction_bidders, auction_participation
    
    def analyze_auctions(self, auctions, auction_bidders):
        """
        Analyze detected auctions and extract features for ML
        """
        auction_features = []
        
        for auction_id, (auction_txs, bidders) in enumerate(zip(auctions, auction_bidders)):
            if len(auction_txs) < 2:
                continue
            
            # Calculate auction statistics
            gas_prices = [tx['gas_price'] for tx in auction_txs]
            timestamps = [tx['time_seen'] for tx in auction_txs]
            
            # Auction features
            features = {
                'auction_id': auction_id,
                'num_bids': len(auction_txs),
                'num_bidders': len(bidders),
                'min_gas_price': min(gas_prices),
                'max_gas_price': max(gas_prices),
                'avg_gas_price': sum(gas_prices) / len(gas_prices),
                'gas_price_range': max(gas_prices) - min(gas_prices),
                'gas_price_std': pd.Series(gas_prices).std(),
                'duration_seconds': (timestamps[-1] - timestamps[0]) / 1e9,
                'first_block': auction_txs[0]['block_number'],
                'last_block': auction_txs[-1]['block_number'],
                'blocks_spanned': auction_txs[-1]['block_number'] - auction_txs[0]['block_number'] + 1,
            }
            
            # Price escalation (indicator of competitive bidding)
            price_increases = sum(1 for i in range(len(gas_prices)-1) 
                                 if gas_prices[i+1] > gas_prices[i])
            features['price_escalation_ratio'] = price_increases / (len(gas_prices) - 1) if len(gas_prices) > 1 else 0
            
            # Winner analysis (highest gas price)
            winner_idx = gas_prices.index(max(gas_prices))
            winner_tx = auction_txs[winner_idx]
            features['winner_address'] = winner_tx['sender']
            features['winner_gas_price'] = winner_tx['gas_price']
            features['winner_position'] = winner_idx
            features['winner_block'] = winner_tx['block_number']
            
            # Label: Is this MEV-related?
            # High gas price + multiple bidders + price escalation = likely MEV
            features['is_mev_auction'] = int(
                features['num_bidders'] >= 2 and
                features['gas_price_range'] > 1e9 and  # >1 Gwei difference
                features['price_escalation_ratio'] > 0.3
            )
            
            # Store individual transactions for detailed analysis
            for idx, tx in enumerate(auction_txs):
                tx_features = features.copy()
                tx_features.update({
                    'tx_hash': tx['hash'],
                    'tx_position_in_auction': idx,
                    'tx_sender': tx['sender'],
                    'tx_gas_price': tx['gas_price'],
                    'tx_gas_limit': tx['gas_limit'],
                    'tx_to_address': tx['to_address'],
                    'tx_value': tx['value'],
                    'tx_block': tx['block_number'],
                    'tx_index': tx['transaction_index'],
                    'is_winner': int(idx == winner_idx),
                    'is_first_bid': int(idx == 0),
                    'is_last_bid': int(idx == len(auction_txs) - 1),
                })
                auction_features.append(tx_features)
        
        return auction_features
    
    def export_to_csv(self, features: list, output_path: str = "data/flashboys_analysis.csv"):
        """Export analysis results to CSV for ML training"""
        if not features:
            logger.warning("No features to export")
            return
        
        df = pd.DataFrame(features)
        
        # Ensure data directory exists
        Path(output_path).parent.mkdir(exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df):,} labeled transactions to {output_path}")
        
        # Print summary stats
        logger.info(f"\n{'='*70}")
        logger.info("FLASH BOYS ANALYSIS RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Total auctions detected: {df['auction_id'].nunique():,}")
        logger.info(f"Total transactions in auctions: {len(df):,}")
        logger.info(f"MEV auctions detected: {df['is_mev_auction'].sum():,}")
        logger.info(f"Average bids per auction: {df.groupby('auction_id')['num_bids'].first().mean():.1f}")
        logger.info(f"Average bidders per auction: {df.groupby('auction_id')['num_bidders'].first().mean():.1f}")
        logger.info(f"{'='*70}\n")


def main():
    """Run Flash Boys analysis on your collected data"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Flash Boys 2.0 Analysis - Official Implementation'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/crypto_data.db',
        help='Input database path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/flashboys_analysis.csv',
        help='Output CSV file'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = FlashBoysAnalyzer(db_path=args.input)
    
    # Load your data
    transactions = analyzer.load_transactions_from_db()
    
    if not transactions:
        logger.error("No transactions found in database!")
        return
    
    logger.info(f"\n{'='*70}")
    logger.info("RUNNING FLASH BOYS 2.0 AUCTION DETECTION")
    logger.info(f"{'='*70}\n")
    
    # Apply official Flash Boys algorithm
    auctions, non_auctions, bidders, participation = analyzer.get_individual_auctions(transactions)
    
    logger.info(f"Detected {len(auctions):,} gas auctions")
    logger.info(f"Non-auction transactions: {len(non_auctions):,}")
    
    # Analyze auctions and extract ML features
    features = analyzer.analyze_auctions(auctions, bidders)
    
    # Export to CSV
    analyzer.export_to_csv(features, args.output)
    
    logger.info(f"\n✅ Analysis complete!")
    logger.info(f"   Results: {args.output}")
    logger.info(f"   Ready for ML training!\n")


if __name__ == "__main__":
    main()
