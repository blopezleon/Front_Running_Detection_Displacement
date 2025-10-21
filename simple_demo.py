#!/usr/bin/env python3
"""
Simple Demo - Quick front-running detection without heavy model training
Shows the data collection and analysis capabilities
"""

import asyncio
import logging
import sqlite3
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_collected_data():
    """Analyze the collected blockchain data"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🔍 FRONT-RUNNING DETECTION - QUICK ANALYSIS 🔍                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    db_path = Path("crypto_data.db")
    if not db_path.exists():
        logger.error("❌ No database found. Please run data collection first.")
        return False
    
    conn = sqlite3.connect(db_path)
    
    # Load data
    logger.info("📊 Loading transaction data from database...")
    tx_df = pd.read_sql_query("SELECT * FROM transactions", conn)
    mev_df = pd.read_sql_query("SELECT * FROM mev_opportunities", conn)
    
    if tx_df.empty:
        logger.error("❌ No transactions found in database.")
        conn.close()
        return False
    
    logger.info(f"✅ Loaded {len(tx_df):,} transactions from {tx_df['block_number'].nunique()} blocks")
    
    # Basic statistics
    print("\n" + "="*80)
    print("📈 TRANSACTION STATISTICS")
    print("="*80)
    
    print(f"\n💰 Transaction Overview:")
    print(f"   Total Transactions: {len(tx_df):,}")
    print(f"   Unique Blocks: {tx_df['block_number'].nunique():,}")
    print(f"   Block Range: {tx_df['block_number'].min():,} to {tx_df['block_number'].max():,}")
    print(f"   Total Value: {tx_df['value'].sum():.4f} ETH")
    print(f"   Average TX per Block: {len(tx_df) / tx_df['block_number'].nunique():.1f}")
    
    print(f"\n⛽ Gas Analysis:")
    print(f"   Average Gas Price: {tx_df['gas_price'].mean()/1e9:.2f} Gwei")
    print(f"   Min Gas Price: {tx_df['gas_price'].min()/1e9:.2f} Gwei")
    print(f"   Max Gas Price: {tx_df['gas_price'].max()/1e9:.2f} Gwei")
    print(f"   Std Dev Gas Price: {tx_df['gas_price'].std()/1e9:.2f} Gwei")
    
    # Front-running pattern detection
    print("\n" + "="*80)
    print("🔍 FRONT-RUNNING PATTERN DETECTION")
    print("="*80)
    
    logger.info("\n🔎 Analyzing transaction patterns...")
    
    # Group by block and analyze
    patterns_found = []
    
    for block_num in tx_df['block_number'].unique():
        block_txs = tx_df[tx_df['block_number'] == block_num].sort_values('transaction_index')
        
        if len(block_txs) < 3:
            continue
        
        # Look for sandwich attack patterns: high gas -> normal gas -> high gas
        for i in range(len(block_txs) - 2):
            tx1 = block_txs.iloc[i]
            tx2 = block_txs.iloc[i + 1]
            tx3 = block_txs.iloc[i + 2]
            
            # Check for gas price sandwich pattern
            if (tx1['gas_price'] > tx2['gas_price'] * 1.5 and 
                tx3['gas_price'] > tx2['gas_price'] * 1.5):
                
                patterns_found.append({
                    'block': block_num,
                    'victim_tx': tx2['transaction_hash'],
                    'front_tx': tx1['transaction_hash'],
                    'back_tx': tx3['transaction_hash'],
                    'victim_gas': tx2['gas_price'] / 1e9,
                    'front_gas': tx1['gas_price'] / 1e9,
                    'back_gas': tx3['gas_price'] / 1e9,
                    'gas_ratio': tx1['gas_price'] / tx2['gas_price'],
                    'victim_value': tx2['value']
                })
    
    if patterns_found:
        print(f"\n🚨 POTENTIAL FRONT-RUNNING DETECTED: {len(patterns_found)} patterns")
        print("\nTop 5 suspicious patterns:")
        print("-" * 80)
        
        for i, pattern in enumerate(sorted(patterns_found, key=lambda x: x['gas_ratio'], reverse=True)[:5], 1):
            print(f"\n   Pattern {i}:")
            print(f"   Block: {pattern['block']}")
            print(f"   Victim TX: {pattern['victim_tx'][:20]}...")
            print(f"   Front-run gas: {pattern['front_gas']:.2f} Gwei ({pattern['gas_ratio']:.2f}x victim)")
            print(f"   Victim gas: {pattern['victim_gas']:.2f} Gwei")
            print(f"   Back-run gas: {pattern['back_gas']:.2f} Gwei")
            print(f"   Victim value: {pattern['victim_value']:.4f} ETH")
    else:
        print("\n✅ No obvious sandwich attack patterns detected")
    
    # High gas price transactions (potential MEV)
    high_gas_threshold = tx_df['gas_price'].quantile(0.95)
    high_gas_txs = tx_df[tx_df['gas_price'] > high_gas_threshold]
    
    print(f"\n⚡ High Priority Transactions (Top 5% gas price):")
    print(f"   Count: {len(high_gas_txs):,}")
    print(f"   Average gas: {high_gas_txs['gas_price'].mean()/1e9:.2f} Gwei")
    print(f"   These might indicate MEV activity")
    
    # MEV opportunities
    if not mev_df.empty:
        print("\n" + "="*80)
        print("💎 MEV OPPORTUNITIES DETECTED")
        print("="*80)
        
        print(f"\n   Total MEV Opportunities: {len(mev_df):,}")
        print(f"   Total Profit: ${mev_df['profit_usd'].sum():,.2f}")
        print(f"   Average Profit: ${mev_df['profit_usd'].mean():,.2f}")
        
        mev_types = mev_df['mev_type'].value_counts()
        print(f"\n   MEV Types:")
        for mev_type, count in mev_types.items():
            print(f"      {mev_type}: {count}")
    else:
        print("\n" + "="*80)
        print("ℹ️  NO MEV OPPORTUNITIES DETECTED YET")
        print("="*80)
        print("\n   Note: MEV detection requires specific transaction patterns.")
        print("   Collect more blocks to increase chances of finding MEV activity.")
    
    # Block-level analysis
    print("\n" + "="*80)
    print("📦 BLOCK-LEVEL ANALYSIS")
    print("="*80)
    
    block_stats = tx_df.groupby('block_number').agg({
        'gas_price': ['mean', 'std', 'max'],
        'value': 'sum',
        'transaction_hash': 'count'
    }).reset_index()
    
    block_stats.columns = ['block', 'avg_gas', 'std_gas', 'max_gas', 'total_value', 'tx_count']
    block_stats['gas_volatility'] = block_stats['std_gas'] / block_stats['avg_gas']
    
    # Find most suspicious blocks (high gas volatility)
    suspicious_blocks = block_stats.nlargest(5, 'gas_volatility')
    
    print(f"\n🎯 Most Suspicious Blocks (high gas price volatility):")
    print("-" * 80)
    for idx, row in suspicious_blocks.iterrows():
        print(f"\n   Block {row['block']:,}")
        print(f"      Transactions: {int(row['tx_count'])}")
        print(f"      Avg Gas: {row['avg_gas']/1e9:.2f} Gwei")
        print(f"      Max Gas: {row['max_gas']/1e9:.2f} Gwei")
        print(f"      Gas Volatility: {row['gas_volatility']:.2f}")
        print(f"      Total Value: {row['total_value']:.4f} ETH")
    
    conn.close()
    
    # Generate summary report
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    print(f"\n✅ Analysis Complete!")
    print(f"   • Analyzed {len(tx_df):,} transactions")
    print(f"   • Found {len(patterns_found)} potential front-running patterns")
    print(f"   • Identified {len(high_gas_txs):,} high-priority transactions")
    print(f"   • Detected {len(suspicious_blocks)} suspicious blocks")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Collect more blocks for better pattern detection")
    print(f"   2. Monitor blocks with high gas volatility")
    print(f"   3. Investigate transactions with unusually high gas prices")
    
    print(f"\n📊 Data files available:")
    print(f"   • crypto_data.db - SQLite database")
    print(f"   • training_data/ - Exported CSV/Parquet files")
    
    return True


async def collect_more_data(num_blocks=10):
    """Quick data collection"""
    from get_data import CryptoDataCollector
    
    logger.info(f"\n🔄 Collecting {num_blocks} more blocks...")
    
    collector = CryptoDataCollector()
    await collector.collect_latest_data(num_blocks=num_blocks, chain_id=1)
    collector.export_data_for_training()
    await collector.close()
    
    logger.info("✅ Data collection complete!")


async def main():
    """Main demo function"""
    import sys
    
    print("\nWhat would you like to do?\n")
    print("1. Analyze existing data")
    print("2. Collect more data (10 blocks)")
    print("3. Collect data + analyze")
    print("4. Exit\n")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        analyze_collected_data()
    elif choice == '2':
        await collect_more_data(10)
    elif choice == '3':
        await collect_more_data(10)
        analyze_collected_data()
    elif choice == '4':
        logger.info("👋 Goodbye!")
    else:
        logger.warning("Invalid choice")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

