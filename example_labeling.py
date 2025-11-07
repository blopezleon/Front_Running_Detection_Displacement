"""
Example: How to use the labeling algorithm on collected data

This script demonstrates:
1. Loading transactions from the database
2. Applying the auction execution algorithm
3. Generating labeled training data
4. Exporting to CSV for ML training
"""

from label_data import TransactionLabeler, AuctionExecutor, Bid
import sqlite3
import pandas as pd


def main():
    print("=" * 70)
    print("Front-Running Detection: Data Labeling Example")
    print("=" * 70)
    
    # Initialize labeler
    labeler = TransactionLabeler(db_path="crypto_data.db")
    
    # Check database
    conn = sqlite3.connect("crypto_data.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM transactions")
    tx_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT block_number) FROM transactions")
    block_count = cursor.fetchone()[0]
    
    print(f"\n📊 Database Statistics:")
    print(f"  Total transactions: {tx_count:,}")
    print(f"  Total blocks: {block_count:,}")
    
    if tx_count == 0:
        print("\n⚠️  No data found. Please run data collection first:")
        print("  python quick_start.py")
        conn.close()
        return
    
    # Get recent blocks for labeling
    cursor.execute("""
        SELECT DISTINCT block_number 
        FROM transactions 
        ORDER BY block_number DESC 
        LIMIT 20
    """)
    blocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"\n🏷️  Labeling {len(blocks)} most recent blocks...")
    print("-" * 70)
    
    # Label each block
    all_labeled = []
    sandwich_count = 0
    
    for i, block_num in enumerate(blocks, 1):
        # Label with different latencies for victim vs attacker
        labeled = labeler.label_block_transactions(
            block_number=block_num,
            delta_0=0.15,  # Victim: 150ms network delay (normal user)
            delta_1=0.03   # Attacker: 30ms delay (co-located with validator)
        )
        
        if labeled:
            sandwich_count += len(labeled)
            print(f"  [{i:2d}/{len(blocks)}] Block {block_num}: "
                  f"🚨 {len(labeled)} sandwich attack(s) detected")
            all_labeled.extend(labeled)
        else:
            print(f"  [{i:2d}/{len(blocks)}] Block {block_num}: ✅ No attacks detected")
    
    print("-" * 70)
    print(f"\n✅ Labeling complete!")
    print(f"  Total labeled examples: {len(all_labeled)}")
    print(f"  Sandwich attacks detected: {sandwich_count}")
    
    if not all_labeled:
        print("\n💡 Tip: Try collecting more blocks to find front-running patterns")
        print("  Sandwich attacks are more common on blocks with high DEX activity")
        return
    
    # Export to CSV
    print(f"\n💾 Exporting labeled data...")
    df = labeler.export_labels_to_csv(all_labeled, "labeled_training_data.csv")
    
    # Show statistics
    print(f"\n📈 Training Data Summary:")
    print(f"  Features: {len(df.columns)} columns")
    print(f"  Examples: {len(df)} rows")
    
    if len(df) > 0:
        print(f"\n🎯 Label Distribution:")
        print(f"  Attacker wins: {(df['winner_player'] == 1).sum()} ({(df['winner_player'] == 1).sum() / len(df) * 100:.1f}%)")
        print(f"  Victim wins: {(df['winner_player'] == 0).sum()} ({(df['winner_player'] == 0).sum() / len(df) * 100:.1f}%)")
        
        print(f"\n💰 Average Rewards:")
        print(f"  Victim avg reward: ${df['victim_reward'].mean():.4f}")
        print(f"  Attacker avg reward: ${df['attacker_reward'].mean():.4f}")
        
        print(f"\n⚡ Gas Price Statistics:")
        print(f"  Avg victim gas: {df['victim_gas_price'].mean() / 1e9:.2f} Gwei")
        print(f"  Avg front-run gas: {df['front_gas_price'].mean() / 1e9:.2f} Gwei")
        print(f"  Avg gas ratio: {df['gas_price_ratio'].mean():.2f}x")
        
        # Sample example
        print(f"\n📋 Sample Labeled Example:")
        sample = df.iloc[0].to_dict()
        print(f"  Block: {sample['block_number']}")
        print(f"  Victim TX: {sample['victim_tx_hash'][:16]}...")
        print(f"  Front TX: {sample['front_tx_hash'][:16]}...")
        print(f"  Gas Ratio: {sample['gas_price_ratio']:.2f}x")
        print(f"  Winner: Player {sample['winner_player']} ({'Attacker' if sample['winner_player'] == 1 else 'Victim'})")
        print(f"  Victim Reward: ${sample['victim_reward']:.4f}")
        print(f"  Attacker Reward: ${sample['attacker_reward']:.4f}")
    
    print(f"\n✅ Next steps:")
    print(f"  1. Review labeled data: labeled_training_data.csv")
    print(f"  2. Train ML model using these features")
    print(f"  3. Deploy model for real-time detection")
    print()


def demo_custom_strategy():
    """
    Advanced: Demonstrate custom strategy implementation
    """
    print("\n" + "=" * 70)
    print("Advanced Example: Custom Strategy")
    print("=" * 70)
    
    def adaptive_strategy(observations, state, current_time):
        """
        Adaptive strategy that reacts to observed competition
        """
        # First bid: submit conservative bid
        if 'initial_bid' not in state:
            initial_bid = Bid(
                timestamp=current_time,
                amount=0.6,  # Start at 60% of max
                player_id=state.get('player_id', 0)
            )
            state['initial_bid'] = True
            state['my_max_bid'] = 0.9
            return initial_bid, state, current_time + 0.2  # Wake up in 200ms
        
        # React to observations
        if observations:
            max_opponent = max(obs.amount for obs in observations 
                             if obs.player_id != state.get('player_id', 0))
            
            # Counter-bid if opponent is bidding high and we haven't exceeded budget
            if max_opponent > 0.65 and 'counter_bid' not in state:
                counter_amount = min(max_opponent * 1.15, state['my_max_bid'])
                counter_bid = Bid(
                    timestamp=current_time,
                    amount=counter_amount,
                    player_id=state.get('player_id', 0)
                )
                state['counter_bid'] = True
                return counter_bid, state, float('inf')
        
        # No more actions
        return None, state, float('inf')
    
    # Loss function with risk aversion
    def quadratic_loss(bid_amount):
        return -(bid_amount ** 2) if bid_amount else 0
    
    # Run simulation
    executor = AuctionExecutor(
        strategy_0=adaptive_strategy,
        strategy_1=adaptive_strategy,
        delta_0=0.1,
        delta_1=0.05,
        duration=1.0,
        loss_function=quadratic_loss
    )
    
    result = executor.execute(
        initial_state_0={'player_id': 0},
        initial_state_1={'player_id': 1}
    )
    
    print(f"\n🎮 Auction Simulation Results:")
    print(f"  Winner: Player {result['winner']}")
    print(f"  Winning Bid: ${result['winning_bid'].amount:.3f}")
    print(f"  Rewards: Player 0 = ${result['rewards'][0]:.4f}, Player 1 = ${result['rewards'][1]:.4f}")
    print(f"  Total Bids: {len(result['all_bids'])}")
    print(f"  Events Logged: {len(result['execution_trace'])}")
    
    print(f"\n📜 Execution Trace:")
    for event in result['execution_trace'][:10]:  # Show first 10 events
        print(f"  t={event['time']:.3f}s: {event['event']} "
              f"(player {event.get('player', 'N/A')})")


if __name__ == "__main__":
    # Run basic labeling
    main()
    
    # Optionally run advanced demo
    # Uncomment to see custom strategy in action:
    # demo_custom_strategy()
