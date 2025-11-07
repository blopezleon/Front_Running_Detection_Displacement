#!/usr/bin/env python3
"""
Comprehensive Guide: Running Continuous Collection + Labeling

This script shows you how to run both processes for hours/days.
"""

import sys
import os

def print_guide():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    CONTINUOUS DATA COLLECTION + LABELING                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

OVERVIEW:
---------
This system allows you to:
  1. Collect Ethereum blockchain data continuously (hours/days)
  2. Label transactions for MEV/front-running patterns simultaneously
  3. Use the exact algorithm from the research paper (Exec auction)


SETUP:
------
1. Make sure you have all dependencies:
   
   pip install web3 aiohttp pandas numpy


OPTION 1: Run Both Manually (Recommended)
------------------------------------------

Terminal 1 - Data Collection:
    python collect_data.py
    
    Options:
      --batch-size N    Blocks per batch (default: 10)
      --delay S         Seconds between batches (default: 5.0)
    
    Examples:
      python collect_data.py                    # Default settings
      python collect_data.py --batch-size 20    # Larger batches
      python collect_data.py --delay 10         # Slower rate


Terminal 2 - Labeling Monitor:
    python label_data.py --monitor
    
    Options:
      --interval S          Check for new blocks every S seconds (default: 30)
      --output FILE         CSV output file (default: labeled_training_data.csv)
      --delta-victim D      Victim network delay in seconds (default: 0.15)
      --delta-attacker D    Attacker network delay in seconds (default: 0.03)
    
    Examples:
      python label_data.py --monitor                      # Default settings
      python label_data.py --monitor --interval 60        # Check every minute
      python label_data.py --monitor --output my_data.csv # Custom output


OPTION 2: Run Both Automatically
---------------------------------

Single command that starts both processes:
    python run_both.py

This will:
  - Start collection in background
  - Start labeling monitor in background
  - Show interleaved output from both
  - Stop both processes when you press Ctrl+C


UNDERSTANDING THE OUTPUT:
-------------------------

Collection Output:
  - "Collecting blocks X to Y..." - Current batch being fetched
  - "Block NNNNN: M transactions" - Individual block saved
  - "Progress: X blocks collected (Y.Z blocks/min)" - Rate stats

Labeling Output:
  - "Found N unlabeled blocks" - New blocks to process
  - "Block NNNNN: M patterns detected" - Sandwich attacks found
  - "Block NNNNN: clean" - No MEV patterns detected
  - "Session total: X patterns labeled" - Running count


WHAT IT DOES:
-------------

Data Collection (collect_data.py):
  • Fetches blocks from Ethereum blockchain continuously
  • Stores blocks and transactions in SQLite database (crypto_data.db)
  • Automatically resumes from last collected block
  • Catches up to chain tip, then follows new blocks
  • Graceful shutdown with Ctrl+C

Labeling (label_data.py --monitor):
  • Monitors database for new blocks every 30s (configurable)
  • Applies Exec(S₀, Δ₀, S₁, Δ₁, [D, ℓ()]) algorithm from paper
  • Simulates two-player auction with network delays
  • Detects sandwich attacks and assigns rewards
  • Exports labeled patterns to CSV for ML training
  • Skips already-labeled blocks (deduplication)


ALGORITHM DETAILS:
------------------

Based on: https://arxiv.org/pdf/1904.05234
Section: 3.2 "Modeling MEV as an Auction"

The Exec() algorithm:
  1. Takes ordered sequence of transactions S₀ (victim first)
  2. Simulates arrival delays: Δ₀=0.15s (victim), Δ₁=0.03s (attacker)
  3. Uses priority queue to resolve who wins block inclusion
  4. Calculates rewards for victim and attacker
  5. Labels pattern as sandwich attack if attacker has advantage

Network delay assumptions:
  • Victim: 150ms delay (normal user, public RPC)
  • Attacker: 30ms delay (MEV bot, optimized infrastructure)
  • These reflect real-world latency differences


MONITORING LONG RUNS:
---------------------

Check progress anytime:
  - Database: crypto_data.db (SQLite, query with sqlite3 or Python)
  - Labels: labeled_training_data.csv (pandas, Excel, etc)

Database stats:
    python -c "import sqlite3; c=sqlite3.connect('crypto_data.db').cursor(); \\
    c.execute('SELECT COUNT(DISTINCT block_number) FROM transactions'); \\
    print(f'Blocks: {c.fetchone()[0]}'); \\
    c.execute('SELECT COUNT(*) FROM transactions'); \\
    print(f'Transactions: {c.fetchone()[0]}')"

Label stats:
    python -c "import pandas as pd; df=pd.read_csv('labeled_training_data.csv'); \\
    print(f'Patterns: {len(df)}'); \\
    print(f'Blocks with MEV: {df.block_number.nunique()}')"


STOPPING:
---------

Both scripts handle Ctrl+C gracefully:
  1. Press Ctrl+C in the terminal
  2. Current operations finish cleanly
  3. Final statistics are printed
  4. All data is saved

If something goes wrong:
  - Database is transaction-safe (won't corrupt)
  - CSV has unique transaction hashes (no duplicates)
  - Just restart the scripts - they resume automatically


PERFORMANCE EXPECTATIONS:
--------------------------

Data Collection:
  • Speed: ~10-30 blocks/minute (depends on RPC rate limits)
  • Storage: ~50-100 MB per 1,000 blocks
  • Can run for days continuously

Labeling:
  • Speed: ~1-5 blocks/second (depends on transaction count)
  • Pattern detection rate: ~5-15% of blocks (varies by period)
  • CPU usage: Low (single-threaded, I/O bound)


TROUBLESHOOTING:
----------------

"Failed to connect to Ethereum RPC":
  → Check internet connection
  → Public RPC might be down, script will retry automatically

"Missing required package":
  → pip install web3 aiohttp pandas numpy

"Database locked":
  → SQLite handles concurrent reads safely
  → Only one writer (collector) at a time
  → Labeler only reads - no conflicts

"No patterns detected":
  → Normal! MEV is relatively rare
  → Most blocks are clean
  → Try collecting 100+ blocks for good sample

"Same blocks being re-labeled":
  → Fixed in latest version
  → CSV tracks labeled blocks to skip them
  → Delete CSV to re-label everything


RECOMMENDED SETTINGS:
---------------------

For fast catch-up (historical data):
    python collect_data.py --batch-size 20 --delay 2

For long-term monitoring (real-time):
    python collect_data.py --batch-size 5 --delay 10

For frequent labeling (real-time):
    python label_data.py --monitor --interval 15

For batch labeling (historical):
    python label_data.py --monitor --interval 60


NEXT STEPS:
-----------

After collecting labeled data:
  1. Load labeled_training_data.csv into pandas
  2. Train ML model (classification/regression)
  3. Predict MEV patterns on new transactions
  4. Build real-time detection system

The labeled CSV contains:
  • block_number: Ethereum block number
  • transaction_hash: Unique transaction ID
  • transaction_index: Position in block
  • from_address: Sender address
  • to_address: Recipient address
  • value: ETH transferred
  • gas_price: Transaction fee
  • is_front_running: Binary label (1=sandwich, 0=clean)
  • victim_reward: V's reward in auction
  • attacker_reward: A's reward in auction
  • displacement: |R₀ - R₁| (auction displacement metric)


═══════════════════════════════════════════════════════════════════════════

Ready to start? Choose your option above!

═══════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    print_guide()
