"""
Front-Running Detection: Data Labeling via Auction Execution Algorithm

This module implements the Exec(S0, Δ0, S1, Δ1, [D, ℓ()]) algorithm for labeling
transaction sequences as part of a game-theoretic front-running detection framework.

The algorithm simulates a two-player auction where:
- S0, S1: Strategy functions for players 0 and 1
- Δ0, Δ1: Time delays (network latency) for each player
- D: Duration of the auction
- ℓ(): Loss function for the losing bidder

Output: ($r0, $r1) - rewards for each player based on auction outcome
"""

import sqlite3
import json
import heapq
from typing import List, Tuple, Optional, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from pathlib import Path
import numpy as np


@dataclass
class Bid:
    """Represents a single bid in the auction"""
    timestamp: float  # ˆt: when bid arrives at mempool
    amount: float  # $b: bid amount (gas price)
    player_id: int  # i ∈ {0, 1}: which player made the bid
    transaction_hash: str = ""  # optional: actual tx hash
    
    def __lt__(self, other):
        """For heap ordering by timestamp"""
        return self.timestamp < other.timestamp


@dataclass
class DelayedEvent:
    """Events scheduled with network delay"""
    time: float
    event_type: str  # "bid" or "observation"
    data: Any = None
    
    def __lt__(self, other):
        return self.time < other.time


@dataclass
class StrategyState:
    """Maintains state for a player's strategy"""
    player_id: int
    sigma: Dict[str, Any]  # Internal state (e.g., observed history, beliefs)
    next_wakeup: float = float('inf')  # t*_i: next time strategy wants to act
    
    
class PriorityQueue:
    """Min-heap priority queue with peek capability"""
    def __init__(self):
        self.heap = []
    
    def push(self, item):
        heapq.heappush(self.heap, item)
    
    def pop(self):
        return heapq.heappop(self.heap) if self.heap else None
    
    def peek_time(self):
        """Get time of next item without removing"""
        return self.heap[0].timestamp if self.heap else float('inf')
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def __len__(self):
        return len(self.heap)


class AuctionExecutor:
    """
    Implements the Exec() algorithm for simulating front-running auctions
    and labeling transaction sequences for ML training.
    """
    
    def __init__(self, 
                 strategy_0: Callable,
                 strategy_1: Callable,
                 delta_0: float,
                 delta_1: float,
                 duration: float,
                 loss_function: Callable[[float], float]):
        """
        Initialize auction executor.
        
        Args:
            strategy_0: Strategy function S0(observations, state, time) -> (action, new_state, wakeup_time)
            strategy_1: Strategy function S1(observations, state, time) -> (action, new_state, wakeup_time)
            delta_0: Network delay for player 0 (in seconds or blocks)
            delta_1: Network delay for player 1
            duration: Auction duration D (sample from distribution or fixed)
            loss_function: ℓ(bid) -> loss value for losing bidder
        """
        self.strategies = {0: strategy_0, 1: strategy_1}
        self.deltas = {0: delta_0, 1: delta_1}
        self.duration = duration
        self.loss_fn = loss_function
        
    def execute(self, 
                initial_state_0: Dict[str, Any] = None,
                initial_state_1: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the auction algorithm (lines 1-27).
        
        Returns:
            Dictionary containing:
                - rewards: ($r0, $r1)
                - winner: player id
                - winning_bid: Bid object
                - losing_bid: Bid object
                - all_bids: chronological list of all bids
                - execution_trace: detailed event log
        """
        # Line 1: Initialize
        b_star = []  # All bids that have arrived (accepted into mempool)
        p = PriorityQueue()  # Priority queue of pending bids (not yet in mempool)
        d0 = PriorityQueue()  # Delayed observations for player 0
        d1 = PriorityQueue()  # Delayed observations for player 1
        
        t_star = 0.0  # Current simulation time
        t_star_0 = 0.0  # Player 0's next wakeup time
        t_star_1 = 0.0  # Player 1's next wakeup time
        
        # Initialize strategy states
        sigma_0 = initial_state_0 or {}
        sigma_1 = initial_state_1 or {}
        
        # Line 2: Sample auction end time
        t_end = self.duration
        
        execution_trace = []
        
        # Initialize strategies at t=0 to get initial bids
        action_0, sigma_0, t_star_0 = self.strategies[0]([], sigma_0, 0.0)
        if action_0 is not None:
            p.push(action_0)
            execution_trace.append({
                'time': 0.0,
                'event': 'initial_action',
                'player': 0,
                'amount': action_0.amount
            })
        
        action_1, sigma_1, t_star_1 = self.strategies[1]([], sigma_1, 0.0)
        if action_1 is not None:
            p.push(action_1)
            execution_trace.append({
                'time': 0.0,
                'event': 'initial_action',
                'player': 1,
                'amount': action_1.amount
            })
        
        # Line 3-20: Main simulation loop
        while True:
            # Determine next event time
            next_bid_time = p.peek_time()
            next_d0_time = d0.heap[0].time if d0.heap else float('inf')
            next_d1_time = d1.heap[0].time if d1.heap else float('inf')
            
            # Line 4-8: Process arriving bid (earliest event)
            if (next_bid_time <= min(t_star_0, t_star_1, next_d0_time, next_d1_time)):
                # Line 5
                t_star = next_bid_time
                
                # Line 6: Pop bid from priority queue
                bid = p.pop()
                
                # Line 7: Add to accepted bids
                b_star.append(bid)
                
                execution_trace.append({
                    'time': t_star,
                    'event': 'bid_arrival',
                    'player': bid.player_id,
                    'amount': bid.amount
                })
                
                # Line 8: Schedule observation for other player (with delay)
                other_player = 1 - bid.player_id
                delayed_time = t_star + self.deltas[other_player]
                
                if other_player == 0:
                    d0.push(DelayedEvent(delayed_time, 'observation', bid))
                else:
                    d1.push(DelayedEvent(delayed_time, 'observation', bid))
            
            # Line 9-14: Process delayed observations
            elif self._check_delayed_observation(0, d0, next_bid_time, t_star_0, t_star_1, next_d0_time, next_d1_time):
                # Player 0 processes observation
                t_star = d0.pop().time
                
                # Line 11: Strategy updates based on observations up to t* - Δ0
                observations = self._get_observations(b_star, t_star, self.deltas[0])
                action, sigma_0, t_star_0 = self.strategies[0](observations, sigma_0, t_star)
                
                execution_trace.append({
                    'time': t_star,
                    'event': 'strategy_update',
                    'player': 0,
                    'observations': len(observations),
                    'next_wakeup': t_star_0
                })
                
                # Line 13-14: Submit action (bid) if not null
                if action is not None:
                    p.push(action)
                    
            elif self._check_delayed_observation(1, d1, next_bid_time, t_star_0, t_star_1, next_d1_time, next_d0_time):
                # Player 1 processes observation (symmetric to above)
                t_star = d1.pop().time
                
                observations = self._get_observations(b_star, t_star, self.deltas[1])
                action, sigma_1, t_star_1 = self.strategies[1](observations, sigma_1, t_star)
                
                execution_trace.append({
                    'time': t_star,
                    'event': 'strategy_update',
                    'player': 1,
                    'observations': len(observations),
                    'next_wakeup': t_star_1
                })
                
                if action is not None:
                    p.push(action)
            
            # Line 15-19: Process strategy wakeups (proactive actions)
            elif self._check_wakeup(0, t_star_0, next_bid_time, next_d0_time, next_d1_time, t_star_1):
                # Player 0 wakeup
                t_star = t_star_0
                
                observations = self._get_observations(b_star, t_star, self.deltas[0])
                action, sigma_0, t_star_0 = self.strategies[0](observations, sigma_0, t_star)
                
                execution_trace.append({
                    'time': t_star,
                    'event': 'wakeup',
                    'player': 0,
                    'next_wakeup': t_star_0
                })
                
                if action is not None:
                    p.push(action)
                    
            elif self._check_wakeup(1, t_star_1, next_bid_time, next_d0_time, next_d1_time, t_star_0):
                # Player 1 wakeup
                t_star = t_star_1
                
                observations = self._get_observations(b_star, t_star, self.deltas[1])
                action, sigma_1, t_star_1 = self.strategies[1](observations, sigma_1, t_star)
                
                execution_trace.append({
                    'time': t_star,
                    'event': 'wakeup',
                    'player': 1,
                    'next_wakeup': t_star_1
                })
                
                if action is not None:
                    p.push(action)
            
            # Line 20: Check termination condition
            if next_bid_time > t_end:
                break
                
            # Safety: prevent infinite loop if no events
            if all(x == float('inf') for x in [next_bid_time, next_d0_time, next_d1_time, t_star_0, t_star_1]):
                break
        
        # Line 21-27: Compute outcomes and rewards
        b = b_star  # All accepted bids
        
        # Line 22: Find winning bid (highest bid amount)
        if not b:
            # No bids - tie or no activity
            return {
                'rewards': (0.0, 0.0),
                'winner': None,
                'winning_bid': None,
                'losing_bid': None,
                'all_bids': [],
                'execution_trace': execution_trace
            }
        
        winning_bid = max(b, key=lambda x: x.amount)
        winner_id = winning_bid.player_id
        
        # Line 23: Find highest bid from losing player
        losing_bids = [bid for bid in b if bid.player_id != winner_id]
        losing_bid = max(losing_bids, key=lambda x: x.amount) if losing_bids else None
        
        # Line 24-27: Compute rewards based on winner
        if winner_id == 0:
            # Player 0 wins
            r0 = 1.0 - winning_bid.amount  # Winner pays their bid, gets unit value
            r1 = self.loss_fn(losing_bid.amount) if losing_bid else 0.0
        else:
            # Player 1 wins
            r0 = self.loss_fn(losing_bid.amount) if losing_bid else 0.0
            r1 = 1.0 - winning_bid.amount
        
        return {
            'rewards': (r0, r1),
            'winner': winner_id,
            'winning_bid': winning_bid,
            'losing_bid': losing_bid,
            'all_bids': b,
            'execution_trace': execution_trace,
            'total_bids': len(b),
            'player_0_bids': len([x for x in b if x.player_id == 0]),
            'player_1_bids': len([x for x in b if x.player_id == 1])
        }
    
    def _check_delayed_observation(self, player_id: int, d_queue: PriorityQueue,
                                   next_bid_time: float, t_star_0: float, t_star_1: float,
                                   d_this_time: float, d_other_time: float) -> bool:
        """Line 9 condition check"""
        if d_queue.is_empty():
            return False
        return (d_this_time < next_bid_time and 
                d_this_time <= min(t_star_0, t_star_1, d_other_time))
    
    def _check_wakeup(self, player_id: int, t_star_i: float,
                     next_bid_time: float, next_d0_time: float, next_d1_time: float,
                     t_star_other: float) -> bool:
        """Line 15 condition check"""
        return (t_star_i < min(next_bid_time, next_d0_time, next_d1_time) and
                t_star_i <= t_star_other)
    
    def _get_observations(self, all_bids: List[Bid], current_time: float, 
                         delay: float) -> List[Bid]:
        """
        Get observations visible to a player at current_time given their delay.
        Returns bids from b*[t* - Δ_i] (all bids up to time minus delay)
        """
        cutoff_time = current_time - delay
        return [bid for bid in all_bids if bid.timestamp <= cutoff_time]


class TransactionLabeler:
    """
    Labels real blockchain transaction data using the auction execution model.
    Converts raw transactions into labeled training examples for ML models.
    """
    
    def __init__(self, db_path: str = "crypto_data.db"):
        self.db_path = db_path
        
    def label_block_transactions(self, 
                                 block_number: int,
                                 base_strategy_0: Callable = None,
                                 base_strategy_1: Callable = None,
                                 delta_0: float = 0.1,
                                 delta_1: float = 0.1) -> List[Dict[str, Any]]:
        """
        Label all transactions in a block as potential front-running scenarios.
        
        Args:
            block_number: Block to analyze
            base_strategy_0: Strategy for honest user (victim)
            base_strategy_1: Strategy for front-runner (attacker)
            delta_0: Network delay for honest user
            delta_1: Network delay for front-runner (often lower)
            
        Returns:
            List of labeled examples with features and target labels
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch all transactions in block, ordered by transaction_index
        cursor.execute("""
            SELECT transaction_hash, transaction_index, from_address, to_address,
                   value, gas_price, gas_limit, input_data, timestamp
            FROM transactions
            WHERE block_number = ?
            ORDER BY transaction_index
        """, (block_number,))
        
        transactions = cursor.fetchall()
        conn.close()
        
        if not transactions:
            return []
        
        # Default strategies if not provided
        if base_strategy_0 is None:
            base_strategy_0 = self._create_simple_strategy(0)
        if base_strategy_1 is None:
            base_strategy_1 = self._create_simple_strategy(1)
        
        labeled_examples = []
        
        # Identify potential front-running pairs
        for i, tx in enumerate(transactions):
            # Look for sandwich pattern: tx[i-1] (front), tx[i] (victim), tx[i+1] (back)
            if i > 0 and i < len(transactions) - 1:
                prev_tx = transactions[i - 1]
                next_tx = transactions[i + 1]
                
                # Check if sandwich: similar DEX interactions with victim in middle
                if self._is_potential_sandwich(prev_tx, tx, next_tx):
                    # Create auction simulation
                    labeled_example = self._label_sandwich_pattern(
                        prev_tx, tx, next_tx, block_number,
                        base_strategy_0, base_strategy_1,
                        delta_0, delta_1
                    )
                    labeled_examples.append(labeled_example)
        
        return labeled_examples
    
    def _is_potential_sandwich(self, front_tx, victim_tx, back_tx) -> bool:
        """
        Heuristic: check if three consecutive transactions form sandwich pattern.
        """
        # Same attacker address for front and back
        if front_tx[2] != back_tx[2]:  # from_address
            return False
        
        # Different victim
        if victim_tx[2] == front_tx[2]:
            return False
        
        # Similar DEX target (to_address)
        dex_addresses = self._get_known_dex_addresses()
        if front_tx[3] in dex_addresses and back_tx[3] in dex_addresses:
            return True
        
        # Gas price pattern: front >= victim, back can be similar
        front_gas = front_tx[5]
        victim_gas = victim_tx[5]
        if front_gas is not None and victim_gas is not None:
            if front_gas >= victim_gas * 1.01:  # 1% higher
                return True
        
        return False
    
    def _label_sandwich_pattern(self, front_tx, victim_tx, back_tx, block_number,
                                strategy_0, strategy_1, delta_0, delta_1) -> Dict[str, Any]:
        """
        Create labeled training example from sandwich pattern using auction model.
        """
        # Extract features
        victim_gas_price = victim_tx[5] if victim_tx[5] else 0
        front_gas_price = front_tx[5] if front_tx[5] else 0
        back_gas_price = back_tx[5] if back_tx[5] else 0
        
        victim_value = victim_tx[4] if victim_tx[4] else 0
        
        # Normalize bids (gas prices) to [0, 1] range for auction
        max_gas = max(front_gas_price, victim_gas_price, back_gas_price, 1)
        norm_victim_bid = victim_gas_price / max_gas
        norm_front_bid = front_gas_price / max_gas
        
        # Create simple loss function (linear or quadratic)
        def loss_fn(bid):
            return -bid  # Loser's loss proportional to their bid
        
        # Create auction executor
        executor = AuctionExecutor(
            strategy_0=strategy_0,
            strategy_1=strategy_1,
            delta_0=delta_0,
            delta_1=delta_1,
            duration=1.0,  # Single block duration
            loss_function=loss_fn
        )
        
        # Execute auction with initial bids
        initial_state_0 = {'initial_bid': norm_victim_bid, 'tx_hash': victim_tx[0]}
        initial_state_1 = {'initial_bid': norm_front_bid, 'tx_hash': front_tx[0]}
        
        result = executor.execute(initial_state_0, initial_state_1)
        
        # Create labeled example
        return {
            'block_number': block_number,
            'victim_tx_hash': victim_tx[0],
            'front_tx_hash': front_tx[0],
            'back_tx_hash': back_tx[0],
            
            # Features
            'victim_gas_price': victim_gas_price,
            'front_gas_price': front_gas_price,
            'back_gas_price': back_gas_price,
            'victim_value': victim_value,
            'gas_price_ratio': front_gas_price / victim_gas_price if victim_gas_price > 0 else 0,
            'tx_position_victim': victim_tx[1],
            'tx_position_front': front_tx[1],
            
            # Labels from auction model
            'is_frontrun': 1,  # Detected sandwich
            'winner_player': result['winner'],
            'victim_reward': result['rewards'][0],
            'attacker_reward': result['rewards'][1],
            'total_bids': result['total_bids'],
            
            # Execution trace (for analysis)
            'execution_trace': json.dumps(result['execution_trace'])
        }
    
    def _create_simple_strategy(self, player_id: int) -> Callable:
        """
        Create a simple baseline strategy that submits one bid at start.
        Real strategies would be more sophisticated (reinforcement learning, etc.)
        """
        def strategy(observations: List[Bid], state: Dict[str, Any], 
                    current_time: float) -> Tuple[Optional[Bid], Dict[str, Any], float]:
            # Submit initial bid if not done
            if 'bid_submitted' not in state:
                bid_amount = state.get('initial_bid', 0.5)
                bid = Bid(
                    timestamp=current_time,
                    amount=bid_amount,
                    player_id=player_id,
                    transaction_hash=state.get('tx_hash', '')
                )
                state['bid_submitted'] = True
                return bid, state, float('inf')  # No more wakeups
            else:
                # Already bid, no more actions
                return None, state, float('inf')
        
        return strategy
    
    def _get_known_dex_addresses(self) -> set:
        """Return known DEX router addresses on Ethereum"""
        return {
            '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',  # Uniswap V2 Router
            '0xe592427a0aece92de3edee1f18e0157c05861564',  # Uniswap V3 Router
            '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f',  # Sushiswap Router
            '0x1111111254fb6c44bac0bed2854e76f90643097d',  # 1inch V4 Router
        }
    
    def export_labels_to_csv(self, labeled_examples: List[Dict[str, Any]], 
                            output_path: str = "labeled_training_data.csv",
                            append: bool = False):
        """
        Export labeled examples to CSV for model training
        
        Args:
            labeled_examples: List of labeled examples
            output_path: Output CSV file path
            append: If True, append to existing file instead of overwriting
        """
        import pandas as pd
        
        if not labeled_examples:
            print("No labeled examples to export")
            return
        
        # Flatten execution trace if needed
        for example in labeled_examples:
            if 'execution_trace' in example and isinstance(example['execution_trace'], str):
                # Keep as string or could parse
                pass
        
        df_new = pd.DataFrame(labeled_examples)
        
        if append and Path(output_path).exists():
            # Append to existing file
            df_existing = pd.read_csv(output_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates based on transaction hashes
            df_combined = df_combined.drop_duplicates(
                subset=['victim_tx_hash', 'front_tx_hash'], 
                keep='last'
            )
            df_combined.to_csv(output_path, index=False)
            print(f"Appended {len(df_new)} new labels to {output_path} (total: {len(df_combined)})")
        else:
            # Create new file
            df_new.to_csv(output_path, index=False)
            print(f"Exported {len(labeled_examples)} labeled examples to {output_path}")
        
        return df_new
    
    def get_labeled_blocks(self, output_path: str = "labeled_training_data.csv") -> Set[int]:
        """Get set of blocks that have already been labeled"""
        import pandas as pd
        
        if not Path(output_path).exists():
            return set()
        
        try:
            df = pd.read_csv(output_path)
            return set(df['block_number'].unique())
        except Exception:
            return set()
    
    def monitor_and_label(self, 
                         output_path: str = "labeled_training_data.csv",
                         check_interval: int = 30,
                         delta_0: float = 0.15,
                         delta_1: float = 0.03,
                         max_iterations: Optional[int] = None):
        """
        Continuously monitor database for new blocks and label them
        
        Args:
            output_path: CSV file to append labels to
            check_interval: Seconds between database checks
            delta_0: Victim network delay
            delta_1: Attacker network delay
            max_iterations: Max number of check cycles (None = infinite)
        """
        import time
        
        print("="*70)
        print("CONTINUOUS LABELING MODE")
        print("="*70)
        print(f"Output file: {output_path}")
        print(f"Check interval: {check_interval}s")
        print(f"Network delays: victim={delta_0}s, attacker={delta_1}s")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        iteration = 0
        total_labeled = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                
                # Get already labeled blocks
                labeled_blocks = self.get_labeled_blocks(output_path)
                
                # Get all collected blocks from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT block_number 
                    FROM transactions 
                    ORDER BY block_number DESC
                """)
                all_blocks = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                # Find unlabeled blocks
                unlabeled = [b for b in all_blocks if b not in labeled_blocks]
                
                if unlabeled:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(unlabeled)} unlabeled blocks")
                    
                    # Label new blocks
                    batch_labeled = []
                    for block_num in unlabeled:
                        labeled = self.label_block_transactions(
                            block_number=block_num,
                            delta_0=delta_0,
                            delta_1=delta_1
                        )
                        if labeled:
                            print(f"  Block {block_num}: {len(labeled)} patterns detected")
                            batch_labeled.extend(labeled)
                        else:
                            print(f"  Block {block_num}: clean")
                    
                    # Export/append to CSV
                    if batch_labeled:
                        self.export_labels_to_csv(
                            batch_labeled, 
                            output_path=output_path,
                            append=True
                        )
                        total_labeled += len(batch_labeled)
                    
                    print(f"Session total: {total_labeled} patterns labeled")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No new blocks. "
                          f"Waiting {check_interval}s...")
                
                # Wait before next check
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n\nStopping labeling monitor...")
        
        print("\n" + "="*70)
        print("LABELING SESSION COMPLETE")
        print("="*70)
        print(f"Iterations: {iteration}")
        print(f"Total patterns labeled: {total_labeled}")
        print(f"Output: {output_path}")
        print("="*70)


def example_usage():
    """
    Example: Label transactions from collected blocks
    """
    print("=" * 60)
    print("Front-Running Detection: Data Labeling")
    print("=" * 60)
    
    labeler = TransactionLabeler()
    
    # Label transactions from recent blocks
    conn = sqlite3.connect("crypto_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT block_number FROM transactions ORDER BY block_number DESC LIMIT 10")
    blocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not blocks:
        print("⚠️  No blocks found in database. Run data collection first.")
        return
    
    print(f"\n📊 Labeling transactions from {len(blocks)} blocks...")
    
    all_labeled = []
    for block_num in blocks:
        labeled = labeler.label_block_transactions(
            block_number=block_num,
            delta_0=0.1,  # Honest user: 100ms delay
            delta_1=0.05  # Attacker: 50ms delay (faster)
        )
        all_labeled.extend(labeled)
        if labeled:
            print(f"  Block {block_num}: found {len(labeled)} potential front-running patterns")
    
    print(f"\n✅ Total labeled examples: {len(all_labeled)}")
    
    if all_labeled:
        # Export to CSV
        labeler.export_labels_to_csv(all_labeled)
        
        # Print sample
        print("\n📋 Sample labeled example:")
        sample = all_labeled[0]
        for key, value in sample.items():
            if key != 'execution_trace':
                print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Front-Running Detection Data Labeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python label_data.py                    # Label recent blocks once
  python label_data.py --monitor          # Continuous monitoring mode
  python label_data.py --monitor --interval 60  # Check every 60 seconds
        """
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Run in continuous monitoring mode (waits for new data)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        metavar='SECONDS',
        help='Check interval in seconds for monitoring mode (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='labeled_training_data.csv',
        metavar='FILE',
        help='Output CSV file (default: labeled_training_data.csv)'
    )
    
    parser.add_argument(
        '--delta-victim',
        type=float,
        default=0.15,
        metavar='SECONDS',
        help='Network delay for victim in seconds (default: 0.15)'
    )
    
    parser.add_argument(
        '--delta-attacker',
        type=float,
        default=0.03,
        metavar='SECONDS',
        help='Network delay for attacker in seconds (default: 0.03)'
    )
    
    args = parser.parse_args()
    
    labeler = TransactionLabeler()
    
    if args.monitor:
        # Continuous monitoring mode
        print("Starting continuous labeling monitor...")
        print("This will wait for new blocks to be added to the database")
        print("Run quick_start.py in another terminal to collect data\n")
        
        labeler.monitor_and_label(
            output_path=args.output,
            check_interval=args.interval,
            delta_0=args.delta_victim,
            delta_1=args.delta_attacker,
            max_iterations=None
        )
    else:
        # One-time labeling
        example_usage()
