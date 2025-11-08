#!/usr/bin/env python3
"""
Enhanced Auction Detection - Combining Flash Boys 2.0 with Game Theory

This integrates the auction detection methodology from the Flash Boys 2.0
paper with your game-theoretic Exec() algorithm for richer labeling.
"""

import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add flashboys2 to path for their utilities
sys.path.append(str(Path(__file__).parent / 'flashboys2'))


class FlashBoysAuctionDetector:
    """
    Auction detection using Flash Boys 2.0 methodology
    
    Based on flashboys2/read_csv.py but adapted for our database schema
    """
    
    def __init__(self, db_path: str = "crypto_data.db"):
        self.db_path = db_path
        self.time_window = 3.0  # seconds (from paper)
    
    def get_transactions_time_ordered(self, start_block: int = None, 
                                     end_block: int = None, 
                                     limit: int = 10000) -> List[Dict]:
        """
        Get transactions ordered by timestamp
        
        Returns transactions with all fields needed for auction detection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
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
                timestamp,
                nonce,
                input_data
            FROM transactions
            WHERE 1=1
        """
        
        params = []
        if start_block:
            query += " AND block_number >= ?"
            params.append(start_block)
        if end_block:
            query += " AND block_number <= ?"
            params.append(end_block)
        
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        # Convert to list of dicts
        result = []
        for row in cursor.fetchall():
            result.append(dict(row))
        
        conn.close()
        return result
    
    def should_filter_frontier(self, frontier: Dict, bidder_id: Tuple[str, int]) -> bool:
        """
        Filter out-of-order transactions (from Flash Boys)
        
        This prevents including transactions that were seen late due to
        network propagation delays or node sync issues.
        
        Args:
            frontier: Dict mapping address to latest known nonce
            bidder_id: Tuple of (address, nonce)
        
        Returns:
            True if transaction should be filtered out
        """
        bid_addr, bid_nonce = bidder_id
        
        if bid_addr in frontier:
            # If this nonce is more than 2 behind frontier, filter it
            if frontier[bid_addr] > bid_nonce + 2:
                return True
            frontier[bid_addr] = max(frontier[bid_addr], bid_nonce)
        else:
            frontier[bid_addr] = bid_nonce
        
        return False
    
    def detect_auctions(self, transactions: List[Dict]) -> Tuple[List[List[Dict]], Dict]:
        """
        Detect gas auctions from transaction stream (Flash Boys methodology)
        
        An auction is a series of transactions within a 3-second window
        where multiple actors are competing by increasing gas prices.
        
        Returns:
            (auctions, metadata) where:
            - auctions: List of auction lists, each auction is list of tx dicts
            - metadata: Dict with statistics about detection
        """
        auctions = []
        current_auction = []
        current_bidders = set()
        auction_id = 0
        auction_participation = {}  # Maps (address, nonce) to auction_id
        non_auction_txs = []
        
        # Frontier tracking to filter sync issues
        frontier = {}
        
        for i in range(len(transactions) - 1):
            prev_tx = transactions[i]
            tx = transactions[i+1]
            
            # Filter out-of-order transactions
            bidder_id = (tx['from_address'], tx['nonce'])
            if self.should_filter_frontier(frontier, bidder_id):
                continue
            
            # Calculate time difference
            prev_time = datetime.fromisoformat(str(prev_tx['timestamp']))
            curr_time = datetime.fromisoformat(str(tx['timestamp']))
            time_diff = (curr_time - prev_time).total_seconds()
            
            if time_diff < self.time_window:
                # This tx is part of an auction
                if len(current_auction) == 0:
                    # New auction starting
                    current_auction = [prev_tx, tx]
                    non_auction_txs = non_auction_txs[:-1]  # prev_tx was actually in auction
                    
                    prev_bidder = (prev_tx['from_address'], prev_tx['nonce'])
                    current_bidders.add(prev_bidder)
                    current_bidders.add(bidder_id)
                    
                    auction_participation[prev_bidder] = auction_id
                else:
                    current_auction.append(tx)
                    current_bidders.add(bidder_id)
                
                auction_participation[bidder_id] = auction_id
            else:
                # Auction ended (or no auction)
                if len(current_auction) >= 2:  # Only save auctions with 2+ bids
                    auctions.append({
                        'id': auction_id,
                        'transactions': current_auction,
                        'bidders': current_bidders,
                        'start_time': current_auction[0]['timestamp'],
                        'end_time': current_auction[-1]['timestamp']
                    })
                    auction_id += 1
                    current_auction = []
                    current_bidders = set()
                
                non_auction_txs.append(tx)
        
        # Handle last auction
        if len(current_auction) >= 2:
            auctions.append({
                'id': auction_id,
                'transactions': current_auction,
                'bidders': current_bidders,
                'start_time': current_auction[0]['timestamp'],
                'end_time': current_auction[-1]['timestamp']
            })
        
        metadata = {
            'total_transactions': len(transactions),
            'num_auctions': len(auctions),
            'non_auction_txs': len(non_auction_txs),
            'auction_participation': auction_participation
        }
        
        return auctions, metadata
    
    def calculate_auction_features(self, auction: Dict) -> Dict:
        """
        Calculate rich features for an auction (beyond simple gas prices)
        
        These features capture the dynamics of bidding behavior
        """
        txs = auction['transactions']
        
        # Basic statistics
        gas_prices = [tx['gas_price'] for tx in txs]
        gas_limits = [tx['gas_limit'] for tx in txs]
        values = [tx['value'] for tx in txs]
        
        # Timing analysis
        start_time = datetime.fromisoformat(str(txs[0]['timestamp']))
        end_time = datetime.fromisoformat(str(txs[-1]['timestamp']))
        duration = (end_time - start_time).total_seconds()
        
        # Bid dynamics (price changes)
        price_deltas = []
        time_deltas = []
        for i in range(1, len(txs)):
            price_delta = txs[i]['gas_price'] - txs[i-1]['gas_price']
            price_deltas.append(price_delta)
            
            t_curr = datetime.fromisoformat(str(txs[i]['timestamp']))
            t_prev = datetime.fromisoformat(str(txs[i-1]['timestamp']))
            time_deltas.append((t_curr - t_prev).total_seconds())
        
        # Bidder analysis
        unique_bidders = len(auction['bidders'])
        bidder_addrs = [tx['from_address'] for tx in txs]
        repeat_bidders = len(bidder_addrs) - len(set(bidder_addrs))
        
        return {
            'auction_id': auction['id'],
            'num_bids': len(txs),
            'num_unique_bidders': unique_bidders,
            'repeat_bidders': repeat_bidders,
            
            # Gas price statistics
            'min_gas_price': min(gas_prices),
            'max_gas_price': max(gas_prices),
            'avg_gas_price': np.mean(gas_prices),
            'std_gas_price': np.std(gas_prices),
            'price_range': max(gas_prices) - min(gas_prices),
            
            # Bid dynamics
            'avg_price_delta': np.mean(price_deltas) if price_deltas else 0,
            'max_price_delta': max(price_deltas) if price_deltas else 0,
            'avg_time_delta': np.mean(time_deltas) if time_deltas else 0,
            
            # Gas limits
            'avg_gas_limit': np.mean(gas_limits),
            'total_gas_offered': sum(gas_limits),
            
            # Value transferred
            'total_value': sum(values),
            'avg_value': np.mean(values),
            
            # Timing
            'duration_seconds': duration,
            'bids_per_second': len(txs) / duration if duration > 0 else 0,
            
            # Block information
            'block_numbers': list(set(tx['block_number'] for tx in txs)),
            'crosses_blocks': len(set(tx['block_number'] for tx in txs)) > 1,
        }
    
    def export_auctions_to_csv(self, auctions: List[Dict], 
                               output_path: str = "detected_auctions.csv"):
        """
        Export detected auctions to CSV for analysis
        
        Similar format to Flash Boys auctions.csv
        """
        rows = []
        
        for auction in auctions:
            features = self.calculate_auction_features(auction)
            
            for tx in auction['transactions']:
                row = {
                    'auction_id': auction['id'],
                    'block_number': tx['block_number'],
                    'transaction_hash': tx['transaction_hash'],
                    'transaction_index': tx['transaction_index'],
                    'from_address': tx['from_address'],
                    'to_address': tx['to_address'],
                    'value': tx['value'],
                    'gas_price': tx['gas_price'],
                    'gas_limit': tx['gas_limit'],
                    'nonce': tx['nonce'],
                    'timestamp': tx['timestamp'],
                    **features  # Include auction-level features
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(auctions)} auctions ({len(rows)} transactions) to {output_path}")
        return df


def main():
    """
    Example usage: Detect auctions from your database
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect gas auctions using Flash Boys methodology')
    parser.add_argument('--start-block', type=int, help='Start block number')
    parser.add_argument('--end-block', type=int, help='End block number')
    parser.add_argument('--limit', type=int, default=10000, help='Max transactions to analyze')
    parser.add_argument('--output', default='detected_auctions.csv', help='Output CSV file')
    args = parser.parse_args()
    
    print("="*70)
    print("FLASH BOYS AUCTION DETECTION")
    print("="*70)
    
    detector = FlashBoysAuctionDetector()
    
    # 1. Load transactions
    print(f"\nLoading transactions from database...")
    transactions = detector.get_transactions_time_ordered(
        start_block=args.start_block,
        end_block=args.end_block,
        limit=args.limit
    )
    print(f"  → Loaded {len(transactions):,} transactions")
    
    # 2. Detect auctions
    print(f"\nDetecting auctions (time window = {detector.time_window}s)...")
    auctions, metadata = detector.detect_auctions(transactions)
    
    print(f"\nResults:")
    print(f"  • Total transactions analyzed: {metadata['total_transactions']:,}")
    print(f"  • Auctions detected: {metadata['num_auctions']:,}")
    print(f"  • Non-auction transactions: {metadata['non_auction_txs']:,}")
    print(f"  • Auction rate: {metadata['num_auctions']/metadata['total_transactions']*100:.2f}%")
    
    # 3. Calculate features for each auction
    print(f"\nCalculating auction features...")
    for auction in auctions[:5]:  # Show first 5
        features = detector.calculate_auction_features(auction)
        print(f"\n  Auction {features['auction_id']}:")
        print(f"    Bids: {features['num_bids']}, Bidders: {features['num_unique_bidders']}")
        print(f"    Gas Price Range: {features['min_gas_price']/1e9:.2f} - {features['max_gas_price']/1e9:.2f} gwei")
        print(f"    Duration: {features['duration_seconds']:.3f}s")
        print(f"    Crosses blocks: {features['crosses_blocks']}")
    
    if len(auctions) > 5:
        print(f"\n  ... and {len(auctions)-5} more auctions")
    
    # 4. Export to CSV
    print(f"\nExporting to {args.output}...")
    detector.export_auctions_to_csv(auctions, args.output)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review {args.output} for detected auctions")
    print(f"  2. Run game-theoretic labeling on these auctions")
    print(f"  3. Combine with DEX profit calculation")
    print()


if __name__ == "__main__":
    main()
