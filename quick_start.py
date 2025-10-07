#!/usr/bin/env python3
"""
Quick Start Script - No API Keys Required!
This script collects Ethereum data using free public RPC endpoints.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_welcome():
    """Print welcome message"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 QUICK START - NO API KEYS NEEDED! 🚀                  ║
║                                                                              ║
║          Collecting Ethereum front-running data using free sources          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def check_dependencies():
    """Check if required packages are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = {
        'web3': 'Web3',
        'pandas': 'pandas',
        'aiohttp': 'aiohttp',
        'sqlite3': 'sqlite3 (built-in)',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"  ✅ {name}")
        except ImportError:
            logger.error(f"  ❌ {name} - MISSING")
            missing.append(package)
    
    if missing:
        logger.error(f"\n⚠️  Missing packages: {', '.join(missing)}")
        logger.info("\n📦 To install missing packages, run:")
        logger.info("   pip install " + " ".join(missing))
        return False
    
    logger.info("✅ All core dependencies installed!\n")
    return True

async def collect_ethereum_data(num_blocks=10):
    """Collect Ethereum data using free public RPC"""
    logger.info(f"📊 Starting collection of {num_blocks} Ethereum blocks...")
    logger.info("⏳ This will take a few minutes...\n")
    
    try:
        from get_data import CryptoDataCollector
        
        # Initialize collector (uses free public RPC by default)
        collector = CryptoDataCollector()
        
        # Collect data from latest blocks
        await collector.collect_latest_data(num_blocks=num_blocks, chain_id=1)
        
        # Export data for training
        logger.info("\n💾 Exporting data for analysis...")
        file_paths = collector.export_data_for_training()
        
        # Clean up
        await collector.close()
        
        logger.info("\n✅ Data collection completed successfully!")
        logger.info(f"📊 Files created:")
        for key, path in file_paths.items():
            logger.info(f"   • {path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during data collection: {e}")
        logger.error(f"   Details: {str(e)}")
        return False

def check_collected_data():
    """Check what data has been collected"""
    import sqlite3
    
    db_path = Path("crypto_data.db")
    if not db_path.exists():
        logger.warning("⚠️  No database found yet.")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get transaction count
        cursor.execute("SELECT COUNT(*) FROM transactions")
        tx_count = cursor.fetchone()[0]
        
        # Get block count
        cursor.execute("SELECT COUNT(DISTINCT block_number) FROM transactions")
        block_count = cursor.fetchone()[0]
        
        # Get MEV count
        cursor.execute("SELECT COUNT(*) FROM mev_opportunities")
        mev_count = cursor.fetchone()[0]
        
        # Get total ETH value
        cursor.execute("SELECT SUM(value) FROM transactions")
        total_eth = cursor.fetchone()[0] or 0
        
        conn.close()
        
        logger.info("\n📈 Current Database Stats:")
        logger.info(f"   📦 Blocks collected: {block_count:,}")
        logger.info(f"   💰 Transactions: {tx_count:,}")
        logger.info(f"   🎯 MEV opportunities: {mev_count:,}")
        logger.info(f"   💎 Total value: {total_eth:.4f} ETH")
        
        # Get file size
        size_mb = db_path.stat().st_size / (1024 * 1024)
        logger.info(f"   💾 Database size: {size_mb:.2f} MB\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading database: {e}")
        return False

def analyze_data():
    """Run analysis on collected data"""
    logger.info("📊 Running analysis on collected data...")
    
    try:
        from analyze_data import DataAnalyzer
        
        analyzer = DataAnalyzer()
        
        if analyzer.tx_df.empty:
            logger.warning("⚠️  No data found. Please collect data first.")
            return False
        
        # Generate statistics
        stats = analyzer.generate_summary_stats()
        
        logger.info("\n📈 ANALYSIS RESULTS:\n")
        
        if 'transactions' in stats:
            tx_stats = stats['transactions']
            logger.info("💰 TRANSACTION STATISTICS:")
            logger.info(f"   Total Transactions: {tx_stats['total_transactions']:,}")
            logger.info(f"   Unique Blocks: {tx_stats['unique_blocks']:,}")
            logger.info(f"   Average Gas Price: {tx_stats['avg_gas_price']/1e9:.2f} Gwei")
            logger.info(f"   Total Value: {tx_stats['total_value_eth']:.4f} ETH")
            logger.info(f"   Avg TX per Block: {tx_stats['avg_tx_per_block']:.1f}\n")
        
        if 'mev' in stats:
            mev_stats = stats['mev']
            logger.info("🎯 MEV STATISTICS:")
            logger.info(f"   MEV Opportunities: {mev_stats['total_mev_opportunities']:,}")
            logger.info(f"   Total Profit: ${mev_stats['total_profit_usd']:,.2f}")
            logger.info(f"   Average Profit: ${mev_stats['avg_profit_usd']:,.2f}")
            logger.info(f"   MEV Percentage: {mev_stats['mev_percentage']:.2f}%\n")
        
        # Detect patterns
        logger.info("🔍 Detecting front-running patterns...")
        patterns_df = analyzer.detect_front_running_patterns()
        
        if not patterns_df.empty:
            logger.info(f"\n🚨 FRONT-RUNNING PATTERNS DETECTED: {len(patterns_df)}")
            logger.info(f"   Average gas ratio: {patterns_df['gas_ratio'].mean():.2f}x")
            logger.info(f"   Total estimated profit: {patterns_df['estimated_profit'].sum():.4f} ETH\n")
        else:
            logger.info("   ✅ No obvious front-running patterns detected\n")
        
        # Generate report
        logger.info("📋 Generating HTML report...")
        analyzer.generate_report("ethereum_analysis_report.html")
        logger.info("✅ Report saved: ethereum_analysis_report.html\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        return False

def show_menu():
    """Show interactive menu"""
    print("""
What would you like to do?

1️⃣  Collect Data (10 blocks - Quick, ~2-3 minutes)
2️⃣  Collect Data (50 blocks - Standard, ~10-15 minutes)
3️⃣  Collect Data (100 blocks - Extended, ~20-30 minutes)
4️⃣  Analyze Collected Data
5️⃣  Show Current Database Stats
6️⃣  Run Full Demo (Collect + Analyze)
7️⃣  Exit

""")

async def main():
    """Main function"""
    print_welcome()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Please install missing dependencies first.")
        sys.exit(1)
    
    while True:
        show_menu()
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            success = await collect_ethereum_data(num_blocks=10)
            if success:
                check_collected_data()
        
        elif choice == '2':
            success = await collect_ethereum_data(num_blocks=50)
            if success:
                check_collected_data()
        
        elif choice == '3':
            success = await collect_ethereum_data(num_blocks=100)
            if success:
                check_collected_data()
        
        elif choice == '4':
            analyze_data()
        
        elif choice == '5':
            check_collected_data()
        
        elif choice == '6':
            logger.info("🚀 Running full demo...\n")
            success = await collect_ethereum_data(num_blocks=20)
            if success:
                check_collected_data()
                analyze_data()
        
        elif choice == '7':
            logger.info("👋 Goodbye!")
            break
        
        else:
            logger.warning("⚠️  Invalid choice. Please enter 1-7.")
        
        input("\n⏸️  Press Enter to continue...")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        logger.error("Please check your installation and try again.")
